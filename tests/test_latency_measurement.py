import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('latency', Path(__file__).resolve().parents[1]/'scripts/latency_test.py')
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


class Engine:
    def __init__(self, counts=(1, 3), fail=False):
        self.counts = list(counts)
        self.active = False
        self.calls = 0
        self.fail = fail

    def has_unfinished_requests(self):
        return self.active

    def add_request(self, request_id, request, params):
        self.request_id = request_id
        self.active = True
        self.calls += 1

    def step(self):
        if self.fail:
            raise RuntimeError('engine failure')
        n = self.counts.pop(0)
        self.active = bool(self.counts)
        return [NS(request_id=self.request_id, finished=not self.active, prompt_token_ids=[1, 2],
                   outputs=[NS(token_ids=list(range(n)), text='native output', finish_reason='stop')])]


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.vllm = patch.dict(sys.modules, {'vllm.sampling_params':NS(RequestOutputKind=NS(CUMULATIVE=1))})
        self.vllm.start()
        self.params = NS(n=1, output_kind=1, **dict.fromkeys([
            "temperature", "top_p", "top_k", "min_p", "max_tokens", "repetition_penalty",
            "presence_penalty", "frequency_penalty", "seed", "stop", "stop_token_ids"], 0))

    def tearDown(self):
        self.vllm.stop()

    def test_step_timestamps_and_native_response(self):
        model = NS(use_vllm=True, llm=NS(llm_engine=Engine()))
        latency.apply_strict_latency_patch(model)
        with patch.object(latency.time, 'perf_counter', side_effect=[10., 11., 13., 14.]):
            result = model.llm.generate([{'prompt':'native prompt'}], sampling_params=self.params)
        self.assertEqual(result[0].outputs[0].text, 'native output')
        self.assertEqual(model._last_ttft, 1.)
        self.assertEqual(model._last_ttpt, 1.)
        self.assertEqual(model._last_timing_metrics['engine_wall_s'], 4.)
        self.assertEqual(model._last_num_completion_tokens, 3)

    def test_failure_has_no_second_request_or_estimated_result(self):
        engine = Engine(fail=True)
        model = NS(use_vllm=True, llm=NS(llm_engine=engine))
        latency.apply_strict_latency_patch(model)
        with self.assertRaisesRegex(RuntimeError, 'engine failure'):
            model.llm.generate([{}], sampling_params=self.params)
        self.assertEqual(engine.calls, 1)
        self.assertFalse(hasattr(model, '_last_ttft'))

    def test_empty_output_has_no_invented_first_token(self):
        with patch.object(latency.time, 'perf_counter', side_effect=[10., 11., 12.]):
            _, m = latency.step_timed_request(NS(llm_engine=Engine(counts=[0])), {}, self.params)
        self.assertEqual(m['num_output_tokens'], 0)
        self.assertTrue(math.isnan(m['ttft_s']))
        self.assertEqual(m['timing_source'], 'no_output_token')

    def test_busy_engine_and_multi_request_rejected(self):
        engine = Engine(); engine.active = True
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            latency.step_timed_request(NS(llm_engine=engine), {}, self.params)
        model = NS(use_vllm=True, llm=NS(llm_engine=Engine()))
        latency.apply_strict_latency_patch(model)
        with self.assertRaisesRegex(ValueError, 'batch size one'):
            model.llm.generate([{}, {}], sampling_params=self.params)

    def test_ordered_allowlist_and_missing_identity(self):
        import pandas as pd
        def build(_):
            return NS(data=pd.DataFrame(dict(index=[3, 1, 2])))
        with patch.dict(sys.modules, {'vlmeval.dataset':NS(build_dataset=build)}):
            actual = latency.build_latency_dataset('test', allowlist=[2, 3], ordered=True)
            self.assertEqual(actual.data['index'].tolist(), [2, 3])
            with self.assertRaises(KeyError):
                latency.build_latency_dataset('test', allowlist=[4], ordered=True)
            with self.assertRaises(ValueError):
                latency.build_latency_dataset('test', allowlist=[2, 2], ordered=True)

    def test_old_matrix_and_four_recipes(self):
        old = latency.latency_recipes(dict(replay_modes=['image_text'], cot_modes=['CoT','noCoT']))
        self.assertEqual([r['name'] for r in old], ['image_text_CoT','image_text_noCoT'])
        new = [dict(name=n, replay_mode='image_text_image_text') for n in ['VT','SLP','half','quarter']]
        self.assertEqual(latency.latency_recipes(dict(recipes=new)), new)
        with self.assertRaises(ValueError):
            latency.latency_recipes(dict(recipes=[new[0], new[0]]))


if __name__ == '__main__':
    unittest.main()
