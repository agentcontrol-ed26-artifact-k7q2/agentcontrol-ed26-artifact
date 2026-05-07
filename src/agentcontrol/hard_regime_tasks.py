"""Harder regime-mapping task pool.

Tasks are constructed locally with deterministic verifiers, explicit difficulty
regime tags, and clear provenance. Pool spans easy_saturation, medium_headroom,
hard_strong_gap, weak_verifier_risk (math/code), and evidence_support_risk.

NOT a benchmark download. Tasks are auditable in this file.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Math (n=30) — exact short-form numeric answers; no model can hide behind prose
# ---------------------------------------------------------------------------
_MATH = [
    # easy_saturation (6)
    ("hm_001", "What is 12 + 7?", "19", "easy_saturation"),
    ("hm_002", "What is 8 * 9?", "72", "easy_saturation"),
    ("hm_003", "What is 144 / 12?", "12", "easy_saturation"),
    ("hm_004", "What is 100 - 37?", "63", "easy_saturation"),
    ("hm_005", "What is the sum of 1 through 10?", "55", "easy_saturation"),
    ("hm_006", "What is 7 squared?", "49", "easy_saturation"),
    # medium_headroom (10) — typical 7B fails ~30%, 70B/V4 mostly succeeds
    ("hm_007", "What is the least common multiple of 12 and 18?", "36", "medium_headroom"),
    ("hm_008", "What is 25% of 320?", "80", "medium_headroom"),
    ("hm_009", "Solve 4x - 7 = 21 for x.", "7", "medium_headroom"),
    ("hm_010", "What is the next number in the sequence 2, 6, 12, 20, 30, ?", "42", "medium_headroom"),
    ("hm_011", "What is 13 squared minus 11 squared?", "48", "medium_headroom"),
    ("hm_012", "If 5x + 3 = 2x + 18, what is x?", "5", "medium_headroom"),
    ("hm_013", "What is the GCD of 84 and 105?", "21", "medium_headroom"),
    ("hm_014", "How many ways can 5 distinct books be arranged on a shelf?", "120", "medium_headroom"),
    ("hm_015", "What is the sum of the interior angles of a hexagon in degrees?", "720", "medium_headroom"),
    ("hm_016", "If a fair die is rolled twice, what is the number of outcomes where the sum is 7?", "6", "medium_headroom"),
    # hard_strong_gap (10) — needs careful reasoning; small models often fail
    ("hm_017", "What is 7^4 mod 13?", "9", "hard_strong_gap"),
    ("hm_018", "Find the smallest positive integer n such that 7n is congruent to 3 modulo 11.", "2", "hard_strong_gap"),
    ("hm_019", "If x + 1/x = 5, what is x^3 + 1/x^3?", "110", "hard_strong_gap"),
    ("hm_020", "How many permutations of the letters in MISSISSIPPI are there?", "34650", "hard_strong_gap"),
    ("hm_021", "Find the number of positive integer divisors of 360.", "24", "hard_strong_gap"),
    ("hm_022", "What is the value of 2^10 - 3^4 - 5^2?", "918", "hard_strong_gap"),
    ("hm_023", "What is the coefficient of x^3 in the expansion of (2x + 3)^5?", "720", "hard_strong_gap"),
    ("hm_024", "Find the smallest positive integer that has exactly 6 divisors.", "12", "hard_strong_gap"),
    ("hm_025", "How many integers from 1 to 100 are divisible by neither 2 nor 5?", "40", "hard_strong_gap"),
    ("hm_026", "What is the greatest common divisor of 252 and 378?", "126", "hard_strong_gap"),
    # weak_verifier_risk (4) — answer is a small integer; weak models may guess plausibly
    ("hm_027", "How many prime numbers are there between 1 and 20 inclusive?", "8", "weak_verifier_risk"),
    ("hm_028", "What is the largest power of 2 that divides 96?", "32", "weak_verifier_risk"),
    ("hm_029", "Sum of all positive integer divisors of 28 (a perfect number) equals what?", "56", "weak_verifier_risk"),
    ("hm_030", "Find the number of trailing zeros in 25 factorial.", "6", "weak_verifier_risk"),
]


# ---------------------------------------------------------------------------
# Code (n=30) — Python function-completion with deterministic unit tests
# ---------------------------------------------------------------------------
_CODE = [
    # easy_saturation (5)
    ("hc_001", "add", "Write add(a, b) returning the sum of two integers.",
     ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"], "easy_saturation"),
    ("hc_002", "is_even", "Write is_even(n) returning True iff n is even.",
     ["assert is_even(2) is True", "assert is_even(3) is False"], "easy_saturation"),
    ("hc_003", "max_of_three", "Write max_of_three(a, b, c) returning max of three integers.",
     ["assert max_of_three(1, 2, 3) == 3", "assert max_of_three(-1, -2, -3) == -1"], "easy_saturation"),
    ("hc_004", "sum_list", "Write sum_list(xs) returning the sum of a list of integers.",
     ["assert sum_list([1,2,3]) == 6", "assert sum_list([]) == 0"], "easy_saturation"),
    ("hc_005", "is_palindrome", "Write is_palindrome(s) iff s reads same forward and backward.",
     ["assert is_palindrome('racecar') is True", "assert is_palindrome('hi') is False"], "easy_saturation"),
    # medium_headroom (10)
    ("hc_006", "fib", "Write fib(n) for nth Fibonacci with fib(0)=0 fib(1)=1.",
     ["assert fib(0) == 0", "assert fib(10) == 55", "assert fib(15) == 610"], "medium_headroom"),
    ("hc_007", "is_prime", "Write is_prime(n) True iff n is prime > 1.",
     ["assert is_prime(2) is True", "assert is_prime(1) is False", "assert is_prime(97) is True", "assert is_prime(100) is False"], "medium_headroom"),
    ("hc_008", "gcd", "Write gcd(a, b) for positive integers.",
     ["assert gcd(12, 18) == 6", "assert gcd(100, 75) == 25", "assert gcd(7, 13) == 1"], "medium_headroom"),
    ("hc_009", "reverse_words", "Write reverse_words(s) reversing word order, not characters.",
     ["assert reverse_words('one two three') == 'three two one'", "assert reverse_words('a') == 'a'"], "medium_headroom"),
    ("hc_010", "count_vowels", "Write count_vowels(s) counting lowercase aeiou in s.",
     ["assert count_vowels('hello') == 2", "assert count_vowels('rhythm') == 0"], "medium_headroom"),
    ("hc_011", "merge_sorted", "Write merge_sorted(a, b) merging two sorted lists.",
     ["assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]", "assert merge_sorted([], [1]) == [1]"], "medium_headroom"),
    ("hc_012", "binary_search", "Write binary_search(a, target) returning index or -1.",
     ["assert binary_search([1,3,5,7,9], 5) == 2", "assert binary_search([1,3,5,7,9], 6) == -1", "assert binary_search([], 1) == -1"], "medium_headroom"),
    ("hc_013", "matrix_transpose", "Write matrix_transpose(m) returning transpose of 2D list.",
     ["assert matrix_transpose([[1,2],[3,4]]) == [[1,3],[2,4]]", "assert matrix_transpose([[1,2,3]]) == [[1],[2],[3]]"], "medium_headroom"),
    ("hc_014", "rle_encode", "Write rle_encode(s) returning [(char, count)] run-length encoding.",
     ["assert rle_encode('aaabbc') == [('a',3),('b',2),('c',1)]", "assert rle_encode('') == []"], "medium_headroom"),
    ("hc_015", "find_duplicates", "Write find_duplicates(xs) returning sorted list of duplicates (each once).",
     ["assert find_duplicates([1,2,2,3,3,3,4]) == [2,3]", "assert find_duplicates([1,2,3]) == []"], "medium_headroom"),
    # hard_strong_gap (11) — DP, graph, parsing
    ("hc_016", "longest_common_subseq_len", "Write longest_common_subseq_len(a, b) returning LCS length of two strings.",
     ["assert longest_common_subseq_len('abcde', 'ace') == 3", "assert longest_common_subseq_len('abc', 'def') == 0", "assert longest_common_subseq_len('AGCAT', 'GAC') == 2"], "hard_strong_gap"),
    ("hc_017", "edit_distance", "Write edit_distance(a, b) returning Levenshtein distance.",
     ["assert edit_distance('kitten', 'sitting') == 3", "assert edit_distance('', 'abc') == 3", "assert edit_distance('abc', 'abc') == 0"], "hard_strong_gap"),
    ("hc_018", "knapsack_01", "Write knapsack_01(weights, values, capacity) returning max total value.",
     ["assert knapsack_01([1,3,4,5], [1,4,5,7], 7) == 9", "assert knapsack_01([2], [3], 1) == 0"], "hard_strong_gap"),
    ("hc_019", "longest_increasing_subseq", "Write longest_increasing_subseq(xs) returning length of LIS.",
     ["assert longest_increasing_subseq([10,9,2,5,3,7,101,18]) == 4", "assert longest_increasing_subseq([3,3,3]) == 1"], "hard_strong_gap"),
    ("hc_020", "topological_sort", "Write topological_sort(n, edges) returning a valid topological ordering of nodes 0..n-1, or [] if cyclic.",
     ["assert topological_sort(2, [(0,1)]) in ([[0,1]],)", "result = topological_sort(4, [(0,1),(0,2),(1,3),(2,3)]); assert len(result) == 4 and result.index(0) < result.index(3)"], "hard_strong_gap"),
    ("hc_021", "shortest_path_bfs", "Write shortest_path_bfs(n, edges, src, dst) returning shortest path length in unweighted undirected graph, or -1.",
     ["assert shortest_path_bfs(4, [(0,1),(1,2),(2,3)], 0, 3) == 3", "assert shortest_path_bfs(3, [(0,1)], 0, 2) == -1"], "hard_strong_gap"),
    ("hc_022", "balanced_parens", "Write balanced_parens(s) iff s has balanced parentheses among ()[]{}.",
     ["assert balanced_parens('()[]{}') is True", "assert balanced_parens('(]') is False", "assert balanced_parens('(([]){})') is True"], "hard_strong_gap"),
    ("hc_023", "rotate_matrix", "Write rotate_matrix(m) rotating an n x n matrix 90 degrees clockwise in place; return None and mutate m.",
     ["m=[[1,2],[3,4]]; rotate_matrix(m); assert m == [[3,1],[4,2]]", "m=[[1,2,3],[4,5,6],[7,8,9]]; rotate_matrix(m); assert m == [[7,4,1],[8,5,2],[9,6,3]]"], "hard_strong_gap"),
    ("hc_024", "lru_cache_class", "Write a class LRUCache(capacity) with get(key) returning value or -1, and put(key, value) evicting least-recently-used.",
     ["c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4"], "hard_strong_gap"),
    ("hc_025", "longest_common_substring_len", "Write longest_common_substring_len(a, b) returning length of longest common contiguous substring.",
     ["assert longest_common_substring_len('abcde', 'cdeab') == 3", "assert longest_common_substring_len('abc', 'def') == 0"], "hard_strong_gap"),
    ("hc_026", "subset_sum", "Write subset_sum(xs, target) returning True iff some subset of nonnegative integers xs sums to target.",
     ["assert subset_sum([1,2,3,7], 6) is True", "assert subset_sum([1,2,3,7], 14) is False", "assert subset_sum([], 0) is True"], "hard_strong_gap"),
    # weak_verifier_risk (4) — small int answers, plausible-looking guesses might pass
    ("hc_027", "count_set_bits", "Write count_set_bits(n) returning the number of 1s in the binary representation of nonnegative n.",
     ["assert count_set_bits(0) == 0", "assert count_set_bits(7) == 3", "assert count_set_bits(255) == 8", "assert count_set_bits(1024) == 1"], "weak_verifier_risk"),
    ("hc_028", "first_missing_positive", "Write first_missing_positive(xs) returning the smallest positive integer not in xs.",
     ["assert first_missing_positive([1,2,0]) == 3", "assert first_missing_positive([3,4,-1,1]) == 2", "assert first_missing_positive([]) == 1"], "weak_verifier_risk"),
    ("hc_029", "single_number", "Write single_number(xs) returning the integer that appears exactly once when every other appears twice.",
     ["assert single_number([2,2,1]) == 1", "assert single_number([4,1,2,1,2]) == 4"], "weak_verifier_risk"),
    ("hc_030", "is_anagram", "Write is_anagram(a, b) iff a and b are anagrams.",
     ["assert is_anagram('listen', 'silent') is True", "assert is_anagram('hello', 'world') is False", "assert is_anagram('', '') is True"], "weak_verifier_risk"),
]


# ---------------------------------------------------------------------------
# Evidence (n=30) — multi-hop QA over a frozen 20-doc corpus with distractors
# ---------------------------------------------------------------------------
_EV_CORPUS = {
    "doc_curie": "Marie Curie discovered polonium and radium and won Nobel Prizes in physics (1903) and chemistry (1911).",
    "doc_curie2": "Pierre Curie shared the 1903 Nobel Prize in Physics with his wife Marie Curie and Henri Becquerel.",
    "doc_einstein": "Albert Einstein developed the theory of general relativity, published in 1915. He won the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect.",
    "doc_planck": "Max Planck originated quantum theory in 1900 and won the Nobel Prize in Physics in 1918.",
    "doc_python": "Python was created by Guido van Rossum and first released in 1991.",
    "doc_python2": "Guido van Rossum was born in the Netherlands in 1956 and worked at Google and later Dropbox.",
    "doc_apollo": "The Apollo 11 mission, commanded by Neil Armstrong, landed humans on the Moon in July 1969.",
    "doc_armstrong": "Neil Armstrong was an American astronaut and the first person to walk on the Moon. He was born in Wapakoneta, Ohio in 1930.",
    "doc_wright": "The Wright brothers, Orville and Wilbur, achieved the first sustained powered flight in 1903 at Kitty Hawk, North Carolina.",
    "doc_dna": "The double helix structure of DNA was elucidated by James Watson and Francis Crick in 1953, building on X-ray data from Rosalind Franklin.",
    "doc_franklin": "Rosalind Franklin produced the X-ray diffraction images that helped identify the helical structure of DNA but died of cancer in 1958 before the Nobel Prize was awarded.",
    "doc_war": "World War II began in 1939 and ended in 1945. The Nazi regime in Germany was led by Adolf Hitler.",
    "doc_telephone": "Alexander Graham Bell patented the telephone in 1876.",
    "doc_evolution": "Charles Darwin published On the Origin of Species in 1859, laying the foundation of evolutionary biology.",
    "doc_darwin": "Charles Darwin was born in 1809 in Shrewsbury, England, and traveled aboard HMS Beagle from 1831 to 1836.",
    "doc_amazon": "The Amazon River, in South America, is the second longest river in the world after the Nile.",
    "doc_nile": "The Nile River, primarily flowing through Egypt and Sudan, is generally considered the longest river in the world.",
    "doc_everest": "Mount Everest is the highest mountain above sea level. It was first summited in 1953 by Edmund Hillary and Tenzing Norgay.",
    "doc_lovelace": "Ada Lovelace wrote notes on Charles Babbage's Analytical Engine and is often described as the first computer programmer.",
    "doc_babbage": "Charles Babbage designed the Analytical Engine in the 1830s, considered an early conceptual general-purpose computer.",
}

_EVIDENCE = [
    # easy_saturation (6) — single-doc lookup
    ("he_001", "Who discovered radium?", "Marie Curie", ["doc_curie"], "easy_saturation"),
    ("he_002", "Who created Python?", "Guido van Rossum", ["doc_python"], "easy_saturation"),
    ("he_003", "Which mission first landed humans on the Moon?", "Apollo 11", ["doc_apollo"], "easy_saturation"),
    ("he_004", "Who patented the telephone?", "Alexander Graham Bell", ["doc_telephone"], "easy_saturation"),
    ("he_005", "Who wrote On the Origin of Species?", "Charles Darwin", ["doc_evolution"], "easy_saturation"),
    ("he_006", "What is the highest mountain above sea level?", "Mount Everest", ["doc_everest"], "easy_saturation"),
    # medium_headroom (10) — single-doc but harder phrasing or detail
    ("he_007", "In what year was Python first released?", "1991", ["doc_python"], "medium_headroom"),
    ("he_008", "In what year did Apollo 11 land humans on the Moon?", "1969", ["doc_apollo"], "medium_headroom"),
    ("he_009", "Who shared the 1903 Nobel Prize in Physics with Marie Curie and Henri Becquerel?", "Pierre Curie", ["doc_curie2"], "medium_headroom"),
    ("he_010", "Which longest-river-in-the-world flows through Egypt and Sudan?", "Nile", ["doc_nile"], "medium_headroom"),
    ("he_011", "In what year did the Wright brothers achieve their first sustained flight?", "1903", ["doc_wright"], "medium_headroom"),
    ("he_012", "Who first summited Mount Everest in 1953 alongside Tenzing Norgay?", "Edmund Hillary", ["doc_everest"], "medium_headroom"),
    ("he_013", "Who originated quantum theory in 1900?", "Max Planck", ["doc_planck"], "medium_headroom"),
    ("he_014", "Who designed the Analytical Engine in the 1830s?", "Charles Babbage", ["doc_babbage"], "medium_headroom"),
    ("he_015", "What is the second longest river in the world?", "Amazon", ["doc_amazon"], "medium_headroom"),
    ("he_016", "Who is often described as the first computer programmer for the Analytical Engine?", "Ada Lovelace", ["doc_lovelace"], "medium_headroom"),
    # hard_strong_gap (8) — multi-hop reasoning, requires linking ≥2 docs
    ("he_017", "What was the country of birth of the person who created Python?", "Netherlands", ["doc_python", "doc_python2"], "hard_strong_gap"),
    ("he_018", "Who commanded the mission that first landed humans on the Moon?", "Neil Armstrong", ["doc_apollo", "doc_armstrong"], "hard_strong_gap"),
    ("he_019", "What did Albert Einstein win the Nobel Prize for, given that he developed general relativity?", "photoelectric effect", ["doc_einstein"], "hard_strong_gap"),
    ("he_020", "Whose X-ray diffraction images contributed to the discovery of the DNA double helix structure?", "Rosalind Franklin", ["doc_dna", "doc_franklin"], "hard_strong_gap"),
    ("he_021", "In what year did the Nazi regime's leader's war end?", "1945", ["doc_war"], "hard_strong_gap"),
    ("he_022", "On what ship did the author of On the Origin of Species travel from 1831 to 1836?", "HMS Beagle", ["doc_evolution", "doc_darwin"], "hard_strong_gap"),
    ("he_023", "In what state in the USA was the first person to walk on the Moon born?", "Ohio", ["doc_apollo", "doc_armstrong"], "hard_strong_gap"),
    ("he_024", "Who shares a 1903 Nobel Prize and discovered radium?", "Marie Curie", ["doc_curie", "doc_curie2"], "hard_strong_gap"),
    # evidence_support_risk (6) — distractor-rich; tests whether models cite correctly
    ("he_025", "Who first walked on the Moon?", "Neil Armstrong", ["doc_apollo", "doc_armstrong"], "evidence_support_risk"),
    ("he_026", "Who discovered the photoelectric effect?", "Albert Einstein", ["doc_einstein"], "evidence_support_risk"),
    ("he_027", "Who summited Mount Everest first in 1953?", "Edmund Hillary", ["doc_everest"], "evidence_support_risk"),
    ("he_028", "Who produced X-ray diffraction images of DNA?", "Rosalind Franklin", ["doc_franklin"], "evidence_support_risk"),
    ("he_029", "In what year did World War II begin?", "1939", ["doc_war"], "evidence_support_risk"),
    ("he_030", "Who was Charles Darwin's birthplace?", "Shrewsbury", ["doc_darwin"], "evidence_support_risk"),
]


def _stable_hash(s: str) -> int:
    """Process-stable hash. Python's built-in hash() randomizes per process
    unless PYTHONHASHSEED is fixed; md5 over UTF-8 bytes is portable."""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def get_pool() -> dict[str, dict[str, Any]]:
    pool = {}
    for tid, q, a, regime in _MATH:
        pool[tid] = {"family": "math", "id": tid, "question": q, "answer": a, "regime": regime,
                     "difficulty": regime}
    for tid, ep, prompt, tests, regime in _CODE:
        pool[tid] = {"family": "code", "id": tid, "entry_point": ep, "prompt": prompt,
                     "tests": tests, "regime": regime, "difficulty": regime}
    for tid, q, a, citations, regime in _EVIDENCE:
        ev_subset = {c: _EV_CORPUS[c] for c in citations}
        # Add 1-2 distractor docs from the corpus to test citation discipline.
        all_docs = sorted(set(_EV_CORPUS.keys()) - set(citations))
        n_dist = 2 if regime in ("hard_strong_gap", "evidence_support_risk") else 1
        # Process-stable distractor selection (md5 is independent of PYTHONHASHSEED).
        h = _stable_hash(tid)
        distractors = [all_docs[(h + i) % len(all_docs)] for i in range(n_dist)]
        for d in distractors:
            ev_subset[d] = _EV_CORPUS[d]
        pool[tid] = {"family": "evidence", "id": tid, "question": q, "answer": a,
                     "citations": citations, "evidence": ev_subset, "distractors": distractors,
                     "regime": regime, "difficulty": regime}
    return pool


def _normalize(s: str) -> str:
    return s.strip().lower()


def verify_math(task: dict, output: str) -> bool:
    if output is None:
        return False
    out = output.strip()
    answer = task["answer"]
    if answer.lstrip("-").isdigit():
        nums = re.findall(r"-?\d+", out.replace(",", ""))
        return bool(nums) and nums[-1] == answer
    return _normalize(answer) in _normalize(out)


def _extract_python(output: str) -> str:
    if not output:
        return ""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return output.strip()


def verify_code(task: dict, output: str) -> bool:
    """Code verifier with bounded recursion / call protection.

    Note: this is an in-process exec with restricted namespace (no builtins
    that touch the network or filesystem are removed, but no external IO is
    used by the gold tests). Wallclock timeout is NOT enforced; tasks are
    designed to terminate quickly. Treat as best-effort, not a hardened
    sandbox. See `paper/ed_final_limitations.md` for caveats.
    """
    code = _extract_python(output)
    if not code:
        return False
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<hard>", "exec"), ns, ns)  # noqa: S102
    except Exception:
        return False
    if task["entry_point"] not in ns:
        return False
    try:
        for t in task["tests"]:
            exec(compile(t, "<hard_test>", "exec"), ns, ns)  # noqa: S102
    except Exception:
        return False
    return True


def verify_evidence(task: dict, output: str) -> tuple[bool, float]:
    """Evidence verifier.

    success requires gold-answer phrase AND at least one authorized citation.
    unsupported_risk = (count of unauthorized cites) / (total cites) when
    any citation is present, else 0.0. This counts partial unsupported risk
    when a model produces both authorized and distractor citations.
    """
    if output is None:
        return False, 0.0
    out_norm = _normalize(output)
    gold = _normalize(task["answer"])
    citations = task["citations"]
    cited = re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", output or "")
    n_total = len(cited)
    n_unauth = sum(1 for c in cited if c not in citations)
    has_authorized = any(c in citations for c in cited)
    answer_present = gold in out_norm
    success = bool(answer_present and has_authorized)
    risk = (n_unauth / n_total) if n_total > 0 else 0.0
    return success, risk


def verify(task: dict, output: str) -> tuple[bool, float]:
    fam = task["family"]
    if fam == "math":
        return verify_math(task, output), 0.0
    if fam == "code":
        return verify_code(task, output), 0.0
    if fam == "evidence":
        return verify_evidence(task, output)
    raise ValueError(f"unknown family {fam}")
