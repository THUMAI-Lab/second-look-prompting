"""Merge complete LogicVista shards and call the native evaluator."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

from vlmeval.probes.readout_v2 import sha256_file, write_json

CONDITIONS = ['slp', 'no_write', 'vtt']


def merge_records(groups, expected_ids):
    records = {}
    for group in groups:
        for record in group:
            key = str(record['index'])
            if key in records:
                raise ValueError(f'Duplicate original index: {key}')
            records[key] = record
    if set(records) != set(expected_ids):
        raise ValueError('Merged coverage differs from frozen dataset indices')
    return [records[key] for key in expected_ids]


def run(root, nproc, dataset_name='LogicVista'):
    if dataset_name != 'LogicVista':
        raise ValueError('The no-read scorer supports LogicVista only')
    from vlmeval.dataset.image_vqa import LogicVista
    from vlmeval.smp import dump
    if os.environ.get('LOCAL_LLM') or os.environ.get('VLMEVAL_OPENAI_TEMPERATURE'):
        raise ValueError('Use the existing gpt-4o-mini judge recipe without model/temperature overrides')
    dataset = LogicVista(dataset_name)
    ids = dataset.data['index'].astype(str).tolist()
    expected_size = 447
    assert len(ids) == expected_size and len(set(ids)) == expected_size
    shards = sorted(p for p in root.glob('shard-*') if p.is_dir())
    if not shards:
        raise ValueError('No generation shards found')
    manifests = [json.loads((p/'manifest.json').read_text()) for p in shards]
    for path, manifest in zip(shards, manifests):
        assert manifest['dataset'] == dataset_name
        complete = json.loads((path/'generation-complete.json').read_text())
        assert complete['conditions'] == CONDITIONS
        assert complete['samples'] == manifest['args']['limit']
        for field in ['dataset_sha256', 'versions', 'source_sha256', 'flex_backend_sha256',
                      'model_metadata_sha256']:
            assert manifest[field] == manifests[0][field], field
        sampling = {k:v for k,v in manifest['sampling'].items() if k != 'sample_seeds'}
        assert sampling == {k:v for k,v in manifests[0]['sampling'].items() if k != 'sample_seeds'}
        for field in ['model', 'tp', 'batch_size', 'max_model_len', 'max_batched_tokens',
                      'max_tokens', 'seed', 'gpu_memory_utilization',
                      'min_pixels', 'max_pixels', 'use_custom_prompt']:
            assert manifest['args'][field] == manifests[0]['args'][field], field
    assert manifests[0]['dataset_sha256'] == sha256_file(Path(os.environ['LMUData'])/f'{dataset_name}.tsv')
    evaluation = {}
    for condition in CONDITIONS:
        groups = []
        for path, manifest in zip(shards, manifests):
            records = json.loads((path/f'{condition}.json').read_text())['records']
            assert [str(r['index']) for r in records] == manifest['sample_indices']
            groups.append(records)
        records = merge_records(groups, ids)
        frame = dataset.data.drop(columns=['image', 'image_path'], errors='ignore').copy()
        assert frame['answer'].astype(str).tolist() == [r['answer'] for r in records]
        frame['prediction'] = [r['prediction'] for r in records]
        dest = root/'merged'/condition
        dest.mkdir(parents=True, exist_ok=True)
        prediction_file = dest/f'{dataset_name}.xlsx'
        dump(frame, str(prediction_file))
        write_json(dest/'generation-audit.json', dict(
            samples=len(records), truncated=sum(r['finish_reason']=='length' for r in records),
            generated_tokens=sum(len(r['token_ids']) for r in records),
            prediction_sha256=sha256_file(prediction_file),
            shard_sources=[str(p/f'{condition}.json') for p in shards]))
        with warnings.catch_warnings():
            # Preserve the declared native judge contract instead of accepting its
            # emergency exact-matching fallback when credentials/service fail.
            warnings.filterwarnings('error', message='.*will use exact matching.*')
            score = dataset.evaluate(str(prediction_file), model='gpt-4o-mini', nproc=nproc)
        if score is None:
            raise RuntimeError(f'Native judge did not produce scores for {condition}')
        evaluation[condition] = score if isinstance(score, dict) else score.to_dict(orient='records')
        write_json(dest/'evaluation-complete.json', dict(
            judge='gpt-4o-mini-2024-07-18',
            samples=len(ids), dataset=dataset_name, score=evaluation[condition],
            scored_prediction_sha256=sha256_file(prediction_file),
            scorer_sha256=sha256_file(Path(__file__).resolve().parents[1]/'dataset/image_vqa.py'),
            parser_sha256=sha256_file(Path(__file__).resolve().parents[1]/'dataset/utils/logicvista.py')))
    write_json(root/'evaluation-complete.json', evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--nproc', type=int, default=512)
    parser.add_argument('--dataset', choices=['LogicVista'], default='LogicVista')
    args = parser.parse_args()
    run(args.root, args.nproc, args.dataset)
