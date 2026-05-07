from agentcontrol.providers import ProviderRequest, deterministic_cache_key


def test_provider_cache_key_stability():
    req1 = ProviderRequest(provider='dummy', model='m', messages=[{'role':'user','content':'hello'}], params={'temperature':0})
    req2 = ProviderRequest(provider='dummy', model='m', messages=[{'role':'user','content':'hello'}], params={'temperature':0})
    assert deterministic_cache_key(req1) == deterministic_cache_key(req2)


def test_provider_cache_key_changes_on_prompt():
    req1 = ProviderRequest(provider='dummy', model='m', messages=[{'role':'user','content':'hello'}])
    req2 = ProviderRequest(provider='dummy', model='m', messages=[{'role':'user','content':'goodbye'}])
    assert deterministic_cache_key(req1) != deterministic_cache_key(req2)
