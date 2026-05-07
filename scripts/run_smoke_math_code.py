#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentcontrol.providers import CachedProvider, DummyProvider, ProviderRequest
from agentcontrol.tasks_math import load_math_tasks, format_dummy_math_prompt
from agentcontrol.tasks_code import load_code_tasks, format_dummy_code_prompt
from agentcontrol.trace_store import TraceEvent, TraceStore, state_hash
from agentcontrol.verifiers import verify_math_answer, run_python_unit_tests
from agentcontrol.utils import read_json, write_json


def call(provider: CachedProvider, model: str, prompt: str):
    return provider.generate(ProviderRequest(provider='dummy', model=model, messages=[{'role':'user','content':prompt}], params={'temperature':0}))


def main() -> None:
    trace = TraceStore('traces/smoke_math_code.jsonl')
    provider = CachedProvider(DummyProvider(), cache_dir='cache/provider', ledger_path='cache/cost_ledger.jsonl')
    outcomes = read_json('experiments/smoke_outcomes.json', {})
    for task in load_math_tasks(20):
        tid = task['id']; outcomes.setdefault(tid, {}); prompt = format_dummy_math_prompt(task)
        cheap = call(provider, 'dummy-cheap', prompt); cheap_ver = verify_math_answer(cheap.text, task['answer'])
        trace.append(TraceEvent(tid, 'math', 0, 'cheap_answer', state_hash({'task':tid}), cheap.request_hash, cheap.model, cheap.input_tokens, cheap.output_tokens, cheap.cache_hit, cheap.cost_usd, cheap.latency_ms, cheap.text, cheap_ver, 20.0, cheap_ver['pass']))
        outcomes[tid]['cheap_answer'] = {'success': cheap_ver['pass'], 'cost': 1.0, 'latency_ms': cheap.latency_ms, 'unsupported_risk': 0.0}
        repair = call(provider, 'dummy-cheap', prompt + '\nREPAIR_WITH_HINT'); repair_ver = verify_math_answer(repair.text, task['answer'])
        outcomes[tid]['cheap_repair'] = {'success': repair_ver['pass'], 'cost': 1.0, 'latency_ms': repair.latency_ms, 'unsupported_risk': 0.0}
        outcomes[tid]['cheap_repair_after_hint'] = {'success': repair_ver['pass'], 'cost': 1.0, 'latency_ms': repair.latency_ms, 'unsupported_risk': 0.0}
        hint = call(provider, 'dummy-pro', prompt + '\nSTRONG_HINT')
        outcomes[tid]['strong_hint'] = {'success': False, 'cost': 2.0, 'latency_ms': hint.latency_ms, 'unsupported_risk': 0.0}
        strong = call(provider, 'dummy-pro', prompt); strong_ver = verify_math_answer(strong.text, task['answer'])
        trace.append(TraceEvent(tid, 'math', 1, 'strong_answer', state_hash({'task':tid}), strong.request_hash, strong.model, strong.input_tokens, strong.output_tokens, strong.cache_hit, strong.cost_usd, strong.latency_ms, strong.text, strong_ver, 20.0, strong_ver['pass']))
        outcomes[tid]['strong_answer'] = {'success': strong_ver['pass'], 'cost': 10.0, 'latency_ms': strong.latency_ms, 'unsupported_risk': 0.0}
    for task in load_code_tasks():
        tid = task['id']; outcomes.setdefault(tid, {}); prompt = format_dummy_code_prompt(task)
        cheap = call(provider, 'dummy-cheap', prompt); cheap_ver = run_python_unit_tests(cheap.text, task['tests'])
        trace.append(TraceEvent(tid, 'code', 0, 'cheap_answer', state_hash({'task':tid}), cheap.request_hash, cheap.model, cheap.input_tokens, cheap.output_tokens, cheap.cache_hit, cheap.cost_usd, cheap.latency_ms, cheap.text, cheap_ver, 20.0, cheap_ver['pass']))
        outcomes[tid]['cheap_answer'] = {'success': cheap_ver['pass'], 'cost': 1.0, 'latency_ms': cheap.latency_ms, 'unsupported_risk': 0.0}
        repair = call(provider, 'dummy-cheap', prompt + '\nREPAIR_WITH_HINT'); repair_ver = run_python_unit_tests(repair.text, task['tests'])
        outcomes[tid]['cheap_repair'] = {'success': False, 'cost': 1.0, 'latency_ms': repair.latency_ms, 'unsupported_risk': 0.0}
        outcomes[tid]['cheap_repair_after_hint'] = {'success': repair_ver['pass'], 'cost': 1.0, 'latency_ms': repair.latency_ms, 'unsupported_risk': 0.0}
        hint = call(provider, 'dummy-pro', prompt + '\nSTRONG_HINT')
        outcomes[tid]['strong_hint'] = {'success': False, 'cost': 2.0, 'latency_ms': hint.latency_ms, 'unsupported_risk': 0.0}
        strong = call(provider, 'dummy-pro', prompt); strong_ver = run_python_unit_tests(strong.text, task['tests'])
        trace.append(TraceEvent(tid, 'code', 1, 'strong_answer', state_hash({'task':tid}), strong.request_hash, strong.model, strong.input_tokens, strong.output_tokens, strong.cache_hit, strong.cost_usd, strong.latency_ms, strong.text, strong_ver, 20.0, strong_ver['pass']))
        outcomes[tid]['strong_answer'] = {'success': strong_ver['pass'], 'cost': 10.0, 'latency_ms': strong.latency_ms, 'unsupported_risk': 0.0}
    write_json('experiments/smoke_outcomes.json', outcomes)
    print('wrote traces/smoke_math_code.jsonl and experiments/smoke_outcomes.json')

if __name__ == '__main__': main()
