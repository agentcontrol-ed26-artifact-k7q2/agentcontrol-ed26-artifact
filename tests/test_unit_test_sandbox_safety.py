from agentcontrol.verifiers import run_python_unit_tests


def test_unit_test_sandbox_blocks_forbidden_import():
    code = "import os\ndef f():\n    return os.listdir('.')\n"
    result = run_python_unit_tests(code, ['assert True'])
    assert result['pass'] is False
    assert 'forbidden' in result['error']
