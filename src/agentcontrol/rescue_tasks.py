"""Expanded rescue task pool with real problem text and deterministic verifiers.

Goal: enough lightweight, locally-verifiable tasks across math/code/evidence
that the cheap arm does not always saturate, exposing cheap-vs-strong-vs-graph
gaps under real models. Tasks are simple and unambiguous so verification is
robust; difficulty comes from breadth and a few tougher entries, not trickery.

NOT real benchmarks (no MATH / HumanEval download). Provenance is documented.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Math: 30 deterministic arithmetic / algebra / word problems with clean answers
# ---------------------------------------------------------------------------
_MATH = [
    ("rm_001", "What is 7 + 6?", "13", "easy"),
    ("rm_002", "What is 12 - 5?", "7", "easy"),
    ("rm_003", "What is 8 * 9?", "72", "easy"),
    ("rm_004", "What is 81 / 9?", "9", "easy"),
    ("rm_005", "What is 50 - 17?", "33", "easy"),
    ("rm_006", "What is 13 + 28?", "41", "easy"),
    ("rm_007", "What is 6 * 7?", "42", "easy"),
    ("rm_008", "What is 100 / 4?", "25", "easy"),
    ("rm_009", "What is 144 / 12?", "12", "easy"),
    ("rm_010", "What is 15 + 27?", "42", "easy"),
    ("rm_011", "If 4 notebooks cost $20, what is the cost of 7 notebooks? Answer in dollars.", "35", "medium"),
    ("rm_012", "A train travels 90 miles in 3 hours. How many miles in 7 hours?", "210", "medium"),
    ("rm_013", "What is 35% of 200?", "70", "medium"),
    ("rm_014", "Solve 3x + 5 = 20 for x.", "5", "medium"),
    ("rm_015", "A rectangle has area 84 and width 7. What is its length?", "12", "medium"),
    ("rm_016", "What is the sum of the integers from 1 to 20?", "210", "medium"),
    ("rm_017", "What is the next number in the sequence 3, 6, 12, 24, ?", "48", "medium"),
    ("rm_018", "If x/4 + 3 = 9, what is x?", "24", "medium"),
    ("rm_019", "A number is tripled and then increased by 4 to get 31. What is the number?", "9", "medium"),
    ("rm_020", "What is 18% of 250?", "45", "medium"),
    ("rm_021", "What is 11 squared minus 7 squared?", "72", "hard"),
    ("rm_022", "What is the least common multiple of 9 and 12?", "36", "hard"),
    ("rm_023", "A price increases from 80 to 100. What is the percent increase?", "25", "hard"),
    ("rm_024", "If 6 workers build 6 chairs in 6 hours, how many chairs do 12 workers build in 12 hours?", "24", "hard"),
    ("rm_025", "What is the greatest common divisor of 48 and 60?", "12", "hard"),
    ("rm_026", "Solve 2(x - 3) = 14 for x.", "10", "hard"),
    ("rm_027", "What is 7! divided by 5!?", "42", "hard"),
    ("rm_028", "A bag has 3 red, 4 blue, and 5 green marbles. The probability of drawing a blue marble is what fraction with denominator 12?", "4", "hard"),
    ("rm_029", "What is the sum of the first 10 odd positive integers?", "100", "hard"),
    ("rm_030", "If a 60% solution is mixed with a 40% solution in equal parts, what is the resulting percentage?", "50", "hard"),
]


# ---------------------------------------------------------------------------
# Code: 15 small Python function-completion tasks with deterministic unit tests
# ---------------------------------------------------------------------------
_CODE = [
    ("rc_001", "add", "Write add(a, b) returning the sum of two integers.",
     ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"], "easy"),
    ("rc_002", "is_even", "Write is_even(n) returning True iff n is even.",
     ["assert is_even(2) is True", "assert is_even(3) is False", "assert is_even(0) is True"], "easy"),
    ("rc_003", "max_of_three", "Write max_of_three(a, b, c) returning the maximum of three integers.",
     ["assert max_of_three(1, 2, 3) == 3", "assert max_of_three(5, 2, 3) == 5", "assert max_of_three(-1, -2, -3) == -1"], "easy"),
    ("rc_004", "factorial", "Write factorial(n) for nonnegative integer n.",
     ["assert factorial(0) == 1", "assert factorial(5) == 120", "assert factorial(1) == 1"], "medium"),
    ("rc_005", "fib", "Write fib(n) returning the nth Fibonacci number with fib(0)=0, fib(1)=1.",
     ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(7) == 13"], "medium"),
    ("rc_006", "is_prime", "Write is_prime(n) returning True iff n is a prime number greater than 1.",
     ["assert is_prime(2) is True", "assert is_prime(4) is False", "assert is_prime(13) is True", "assert is_prime(1) is False"], "medium"),
    ("rc_007", "reverse_words", "Write reverse_words(s) reversing the order of words in s, not characters.",
     ["assert reverse_words('one two three') == 'three two one'", "assert reverse_words('hello') == 'hello'"], "medium"),
    ("rc_008", "count_vowels", "Write count_vowels(s) counting vowels (aeiou, lowercase) in s.",
     ["assert count_vowels('hello') == 2", "assert count_vowels('rhythm') == 0", "assert count_vowels('aeiou') == 5"], "medium"),
    ("rc_009", "gcd", "Write gcd(a, b) returning the greatest common divisor of two positive integers.",
     ["assert gcd(12, 18) == 6", "assert gcd(7, 13) == 1", "assert gcd(100, 75) == 25"], "medium"),
    ("rc_010", "is_palindrome", "Write is_palindrome(s) returning True iff s reads the same forward and backward.",
     ["assert is_palindrome('racecar') is True", "assert is_palindrome('hello') is False", "assert is_palindrome('') is True"], "medium"),
    ("rc_011", "sum_digits", "Write sum_digits(n) returning the sum of the decimal digits of nonnegative integer n.",
     ["assert sum_digits(123) == 6", "assert sum_digits(0) == 0", "assert sum_digits(99) == 18"], "hard"),
    ("rc_012", "merge_sorted", "Write merge_sorted(a, b) merging two sorted lists into one sorted list.",
     ["assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]", "assert merge_sorted([], [1, 2]) == [1, 2]"], "hard"),
    ("rc_013", "binary_search", "Write binary_search(a, target) returning the index of target in sorted list a, or -1 if absent.",
     ["assert binary_search([1, 3, 5, 7, 9], 5) == 2", "assert binary_search([1, 3, 5, 7, 9], 6) == -1", "assert binary_search([], 1) == -1"], "hard"),
    ("rc_014", "rle_encode", "Write rle_encode(s) returning the run-length encoding as a list of (char, count) tuples.",
     ["assert rle_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]", "assert rle_encode('') == []"], "hard"),
    ("rc_015", "matrix_transpose", "Write matrix_transpose(m) returning the transpose of a 2D list m.",
     ["assert matrix_transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]", "assert matrix_transpose([[1, 2, 3]]) == [[1], [2], [3]]"], "hard"),
]


# ---------------------------------------------------------------------------
# Evidence: 15 cited-answer QA tasks over a small local corpus.
# ---------------------------------------------------------------------------
_EVIDENCE_CORPUS = {
    "doc_curie": "Marie Curie discovered polonium and radium and won Nobel Prizes in physics and chemistry.",
    "doc_lovelace": "Ada Lovelace wrote notes on the Analytical Engine and is often described as an early computer programmer.",
    "doc_moon": "The Apollo 11 mission landed humans on the Moon in 1969.",
    "doc_python": "Python was created by Guido van Rossum and first released in 1991.",
    "doc_everest": "Mount Everest is the highest mountain above sea level.",
    "doc_einstein": "Albert Einstein developed the theory of general relativity, published in 1915.",
    "doc_wright": "The Wright brothers achieved the first sustained powered flight in 1903 at Kitty Hawk.",
    "doc_dna": "The double helix structure of DNA was elucidated by James Watson and Francis Crick in 1953.",
    "doc_war": "World War II ended in 1945.",
    "doc_telephone": "Alexander Graham Bell patented the telephone in 1876.",
    "doc_evolution": "Charles Darwin published On the Origin of Species in 1859.",
    "doc_amazon": "The Amazon River is the second longest river in the world after the Nile.",
}
_EVIDENCE = [
    ("re_001", "Who discovered radium?", "Marie Curie", ["doc_curie"]),
    ("re_002", "Which mission landed humans on the Moon in 1969?", "Apollo 11", ["doc_moon"]),
    ("re_003", "Who created Python?", "Guido van Rossum", ["doc_python"]),
    ("re_004", "What is the highest mountain above sea level?", "Mount Everest", ["doc_everest"]),
    ("re_005", "Who developed the theory of general relativity?", "Albert Einstein", ["doc_einstein"]),
    ("re_006", "Where did the Wright brothers achieve their first flight?", "Kitty Hawk", ["doc_wright"]),
    ("re_007", "Who elucidated the double helix structure of DNA?", "Watson and Crick", ["doc_dna"]),
    ("re_008", "In what year did World War II end?", "1945", ["doc_war"]),
    ("re_009", "Who patented the telephone in 1876?", "Alexander Graham Bell", ["doc_telephone"]),
    ("re_010", "Who wrote On the Origin of Species?", "Charles Darwin", ["doc_evolution"]),
    ("re_011", "Who is described as an early computer programmer for the Analytical Engine?", "Ada Lovelace", ["doc_lovelace"]),
    ("re_012", "Which river is the second longest in the world?", "Amazon", ["doc_amazon"]),
    ("re_013", "What did Marie Curie discover besides radium?", "polonium", ["doc_curie"]),
    ("re_014", "In what year was Python first released?", "1991", ["doc_python"]),
    ("re_015", "In what year did the Wright brothers achieve their first sustained powered flight?", "1903", ["doc_wright"]),
]


def get_pool() -> dict[str, dict[str, Any]]:
    pool = {}
    for tid, q, a, d in _MATH:
        pool[tid] = {"family": "math", "id": tid, "question": q, "answer": a, "difficulty": d}
    for tid, ep, prompt, tests, d in _CODE:
        pool[tid] = {"family": "code", "id": tid, "entry_point": ep, "prompt": prompt, "tests": tests, "difficulty": d}
    for tid, q, a, citations in _EVIDENCE:
        pool[tid] = {
            "family": "evidence", "id": tid, "question": q, "answer": a,
            "citations": citations, "evidence": {c: _EVIDENCE_CORPUS[c] for c in citations},
        }
    return pool


# ---------------------------------------------------------------------------
# Verifiers — deterministic, no model calls.
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.strip().lower()


def verify_math(task: dict, output: str) -> bool:
    """Accept if the gold answer appears as a token in the model output.
    Tolerates final-answer prose like 'The answer is 42.'.
    """
    if output is None:
        return False
    out = output.strip()
    answer = task["answer"]
    # Look for the LAST numeric/word match. For numeric answers, prefer the last
    # number in the output. For non-numeric, allow case-insensitive substring.
    if answer.lstrip("-").isdigit():
        nums = re.findall(r"-?\d+", out.replace(",", ""))
        return bool(nums) and nums[-1] == answer
    return _normalize(answer) in _normalize(out)


def _extract_python_code(output: str) -> str:
    if not output:
        return ""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return output.strip()


def verify_code(task: dict, output: str) -> bool:
    """Run the model output as Python code, then execute the deterministic
    unit tests in a sandboxed namespace. No filesystem, no network access used.
    """
    code = _extract_python_code(output)
    if not code:
        return False
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<rescue>", "exec"), ns, ns)  # noqa: S102
    except Exception:
        return False
    if task["entry_point"] not in ns:
        return False
    try:
        for t in task["tests"]:
            exec(compile(t, "<rescue_test>", "exec"), ns, ns)  # noqa: S102
    except Exception:
        return False
    return True


def verify_evidence(task: dict, output: str) -> tuple[bool, float]:
    """Return (success, unsupported_risk).

    Success requires the gold answer phrase AND a citation token referencing
    the correct doc id in brackets. Unsupported risk is 1.0 when the answer
    cites a doc but does not match an authorized citation.
    """
    if output is None:
        return False, 0.0
    out_norm = _normalize(output)
    gold = _normalize(task["answer"])
    citations = task["citations"]
    # citation tokens look like [doc_xxx]; strict bracketed form
    cited = re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", output or "")
    has_authorized_citation = any(c in citations for c in cited)
    has_unauthorized_citation = bool(cited) and not has_authorized_citation
    answer_present = gold in out_norm
    success = bool(answer_present and has_authorized_citation)
    risk = 1.0 if (cited and not has_authorized_citation) else 0.0
    return success, risk


def verify(task: dict, output: str) -> tuple[bool, float]:
    """Family-dispatched verifier. Returns (success, unsupported_risk)."""
    fam = task["family"]
    if fam == "math":
        return verify_math(task, output), 0.0
    if fam == "code":
        return verify_code(task, output), 0.0
    if fam == "evidence":
        return verify_evidence(task, output)
    raise ValueError(f"unknown family: {fam}")
