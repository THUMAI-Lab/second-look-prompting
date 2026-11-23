"""Check unsupported no-read datasets fail before loading models or querying GPUs."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REMOVED = ['OCRBench', 'AI2D_TEST', 'MMBench_DEV_EN_V11', 'WeMath']


def isolated_function(relative, name, **namespace):
    tree = ast.parse((ROOT / relative).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]


class LogicVistaScopeTests(unittest.TestCase):
    def test_cli_only_accepts_logicvista(self):
        calls = []
        main = isolated_function('vlmeval/probes/run_generation_attention.py', 'main',
                                 argparse=argparse, run=calls.append, __doc__='no-read')
        with patch.object(sys, 'argv', ['generator', '--output', 'unused', '--dataset', 'LogicVista']):
            main()
        self.assertEqual(calls[0].dataset, 'LogicVista')
        for dataset in REMOVED:
            with self.subTest(dataset=dataset), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(sys, 'argv', ['generator', '--output', 'unused', '--dataset', dataset]):
                    with self.assertRaises(SystemExit) as result:
                        main()
                    self.assertEqual(result.exception.code, 2)
        self.assertEqual(len(calls), 1)

    def test_direct_entries_reject_before_dependency_or_gpu_access(self):
        prepare = isolated_function('vlmeval/probes/run_generation_attention.py', 'prepare')
        score = isolated_function('vlmeval/probes/score_generation_attention.py', 'run')
        driver = isolated_function('scripts/run_generation_attention_batch.py', 'run', json=json)
        with tempfile.TemporaryDirectory() as temp:
            temporary_root = Path(temp)
            config = temporary_root / 'config.json'
            for dataset in REMOVED:
                config.write_text(json.dumps({'dataset': dataset}))
                with self.subTest(dataset=dataset):
                    with patch('builtins.__import__', side_effect=AssertionError('unexpected dependency load')):
                        with self.assertRaisesRegex(ValueError, 'LogicVista only'):
                            prepare(SimpleNamespace(dataset=dataset))
                        with self.assertRaisesRegex(ValueError, 'LogicVista only'):
                            score(temporary_root, 1, dataset)
                        with self.assertRaisesRegex(ValueError, 'LogicVista only'):
                            driver(config)


if __name__ == '__main__':
    unittest.main()
