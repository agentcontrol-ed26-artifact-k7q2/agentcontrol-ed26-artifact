from agentcontrol.providers import CachedProvider, DummyProvider, ProviderRequest


def test_no_duplicate_api_calls_when_cache_exists(tmp_path):
    dummy = DummyProvider()
    cached = CachedProvider(dummy, cache_dir=tmp_path / 'cache', ledger_path=tmp_path / 'ledger.jsonl')
    req = ProviderRequest(provider='dummy', model='dummy-cheap', messages=[{'role':'user','content':'DUMMY_CHEAP: 1'}])
    cached.generate(req); cached.generate(req); cached.generate(req)
    assert dummy.call_count == 1
