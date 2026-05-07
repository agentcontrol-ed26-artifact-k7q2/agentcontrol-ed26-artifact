from agentcontrol.verifiers import run_python_unit_tests


def test_code_unit_tests_pass():
    result = run_python_unit_tests('def add(a, b):\n    return a + b\n', ['assert add(1, 2) == 3'])
    assert result['pass'] is True


def test_code_unit_tests_fail():
    result = run_python_unit_tests('def add(a, b):\n    return a - b\n', ['assert add(1, 2) == 3'])
    assert result['pass'] is False
