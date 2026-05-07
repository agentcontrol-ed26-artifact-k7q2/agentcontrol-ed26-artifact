from agentcontrol.providers import CachedProvider, DummyProvider, ProviderRequest, total_actual_spend_usd


def test_cost_accounting_exactness(tmp_path):
    dummy = DummyProvider()
    provider = CachedProvider(dummy, cache_dir=tmp_path / 'cache', ledger_path=tmp_path / 'ledger.jsonl')
    req = ProviderRequest(provider='dummy', model='dummy-cheap', messages=[{'role':'user','content':'DUMMY_CHEAP: 5'}])
    r1 = provider.generate(req); r2 = provider.generate(req)
    assert dummy.call_count == 1
    assert r1.cost_usd > 0
    assert r2.cost_usd == 0
    assert total_actual_spend_usd(tmp_path / 'ledger.jsonl') == r1.cost_usd
