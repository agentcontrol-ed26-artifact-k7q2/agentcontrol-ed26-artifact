from agentcontrol.verifiers import math_exact_match, verify_math_answer


def test_math_exact_comparator():
    assert math_exact_match('The answer is 42.', '42')
    assert math_exact_match('x = 7', '7')
    assert not math_exact_match('41', '42')


def test_verify_math_answer():
    out = verify_math_answer('The answer is 20', '20')
    assert out['pass'] is True
    assert out['score'] == 1.0
