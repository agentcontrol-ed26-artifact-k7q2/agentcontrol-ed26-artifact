"""Apollo Phase 3B — build n=60 adversarial verifier-risk pool.

Design principle: tasks where a *weak verifier* (loose pattern match) can
false-accept a plausible wrong answer, while an *exact verifier* (audit
checker) catches it. Includes abstain action.

Families:
  - math_traps (n=20): common-error wrong answers (off-by-one, swapped sign,
    wrong order of ops, sqrt vs square, percent vs proportion, base case mistakes).
  - code_traps (n=15): hidden edge cases (n=0, n=1, empty input, negative,
    duplicates) where a weak unit-test passes but an exact audit suite fails.
  - evidence_traps (n=15): frozen corpus where distractor docs contain the
    gold-answer phrase verbatim but in the wrong context; cheap model may
    cite the distractor with the correct phrase, fooling a weak verifier.
  - data_analysis_traps (n=10): tasks where a common error produces a
    misleading intermediate that prints plausibly but is wrong.

Each task has:
  - task_id, family, question, gold_answer
  - weak_verifier_accepts: list of strings the weak verifier would FALSE-accept
    (the wrong answer is in this list)
  - exact_verifier_audit: a deterministic predicate (function name or arithmetic
    check) that catches the wrong answer
  - common_wrong_answer: the answer a typical small/cheap model produces
  - difficulty, provenance
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
APOLLO = REPO / "main_apollo"


_MATH_TRAPS = [
    # (id, question, gold, common_wrong, weak_accepts (regex/substring), audit_kind)
    ("vm_001", "What is 12 * 13?", "156", "144",
     ["156", "144"], "exact_int_match"),
    ("vm_002", "What is 17 - 9?", "8", "9",  # off-by-one on subtraction
     ["8", "9"], "exact_int_match"),
    ("vm_003", "What is 25% of 60?", "15", "20",  # 60/3 not 60*0.25
     ["15", "20"], "exact_int_match"),
    ("vm_004", "Solve 2x + 7 = 15.", "4", "5",
     ["4", "5"], "exact_int_match"),
    ("vm_005", "What is 3/4 of 20?", "15", "16",  # 20-4 not 20*0.75
     ["15", "16"], "exact_int_match"),
    ("vm_006", "What is the square root of 144?", "12", "144",  # gave the input
     ["12", "144"], "exact_int_match"),
    ("vm_007", "Sum of 1 through 8?", "36", "32",  # n(n+1)/2 vs n^2/2 mistake
     ["36", "32"], "exact_int_match"),
    ("vm_008", "What is 7! divided by 5!?", "42", "7",  # ratio confusion
     ["42", "7"], "exact_int_match"),
    ("vm_009", "If price increases from 80 to 100, percent increase?", "25", "20",  # base confusion
     ["25", "20"], "exact_int_match"),
    ("vm_010", "What is (-3) squared?", "9", "-9",  # sign error
     ["9", "-9"], "exact_int_match"),
    ("vm_011", "How many primes are there from 1 through 10?", "4", "5",  # includes 1
     ["4", "5"], "exact_int_match"),
    ("vm_012", "What is the LCM of 6 and 8?", "24", "48",  # multiplied
     ["24", "48"], "exact_int_match"),
    ("vm_013", "What is the GCD of 12 and 18?", "6", "2",  # missed 3 factor
     ["6", "2"], "exact_int_match"),
    ("vm_014", "How many edges does a cube have?", "12", "8",  # confused with vertices
     ["12", "8"], "exact_int_match"),
    ("vm_015", "How many faces does a tetrahedron have?", "4", "3",
     ["4", "3"], "exact_int_match"),
    ("vm_016", "What is 10 choose 2?", "45", "20",  # nP2 vs nC2
     ["45", "20"], "exact_int_match"),
    ("vm_017", "Count distinct anagrams of word AB?", "2", "4",
     ["2", "4"], "exact_int_match"),
    ("vm_018", "What is the average of 4, 8, 12, 16?", "10", "12",  # mistaken median
     ["10", "12"], "exact_int_match"),
    ("vm_019", "If a discount of 20% is applied to $50, the final price?", "40", "30",  # subtracted 20 not 20%
     ["40", "30"], "exact_int_match"),
    ("vm_020", "What is 9 * 8 mod 7?", "2", "5",  # arithmetic error
     ["2", "5"], "exact_int_match"),
]


_CODE_TRAPS = [
    # (id, entry_point, prompt, weak_tests (subset that wrong impl passes),
    #  audit_tests (full suite incl. edge cases), common_wrong_impl_label)
    ("vc_001", "is_prime", "Write is_prime(n) returning True iff n is prime greater than 1.",
     ["assert is_prime(7) is True"],
     ["assert is_prime(7) is True", "assert is_prime(1) is False",
      "assert is_prime(0) is False", "assert is_prime(-3) is False",
      "assert is_prime(2) is True"],
     "no_n_le_1_check"),
    ("vc_002", "factorial", "Write factorial(n) for nonnegative integer n.",
     ["assert factorial(5) == 120"],
     ["assert factorial(5) == 120", "assert factorial(0) == 1",
      "assert factorial(1) == 1"],
     "missing_base_case"),
    ("vc_003", "max_of_list", "Write max_of_list(xs) returning the maximum element.",
     ["assert max_of_list([3,1,2]) == 3"],
     ["assert max_of_list([3,1,2]) == 3", "assert max_of_list([-3,-1,-2]) == -1",
      "assert max_of_list([5]) == 5"],
     "negative_list_bug"),
    ("vc_004", "is_even", "Write is_even(n).",
     ["assert is_even(2) is True"],
     ["assert is_even(2) is True", "assert is_even(0) is True",
      "assert is_even(-4) is True", "assert is_even(3) is False"],
     "no_zero_or_neg"),
    ("vc_005", "fib", "Write fib(n) returning nth Fibonacci with fib(0)=0, fib(1)=1.",
     ["assert fib(7) == 13"],
     ["assert fib(0) == 0", "assert fib(1) == 1",
      "assert fib(7) == 13", "assert fib(10) == 55"],
     "wrong_base_case_indexing"),
    ("vc_006", "binary_search", "Write binary_search(a, target) returning index or -1.",
     ["assert binary_search([1,3,5,7,9], 5) == 2"],
     ["assert binary_search([1,3,5,7,9], 5) == 2",
      "assert binary_search([1,3,5,7,9], 6) == -1",
      "assert binary_search([], 1) == -1",
      "assert binary_search([2], 2) == 0"],
     "no_empty_handling"),
    ("vc_007", "strip_whitespace", "Write strip_whitespace(s) returning s with leading/trailing whitespace removed.",
     ["assert strip_whitespace('  hi  ') == 'hi'"],
     ["assert strip_whitespace('  hi  ') == 'hi'",
      "assert strip_whitespace('') == ''",
      "assert strip_whitespace('   ') == ''",
      "assert strip_whitespace('\\t hi \\n') == 'hi'"],
     "no_empty_or_tab"),
    ("vc_008", "count_words", "Write count_words(s) counting words separated by whitespace.",
     ["assert count_words('one two three') == 3"],
     ["assert count_words('one two three') == 3",
      "assert count_words('') == 0",
      "assert count_words('   ') == 0",
      "assert count_words('  one  ') == 1"],
     "no_empty_handling"),
    ("vc_009", "reverse_list", "Write reverse_list(xs) returning a reversed copy.",
     ["assert reverse_list([1,2,3]) == [3,2,1]"],
     ["assert reverse_list([1,2,3]) == [3,2,1]",
      "assert reverse_list([]) == []",
      "assert reverse_list([1]) == [1]"],
     "no_empty_or_singleton"),
    ("vc_010", "find_first", "Write find_first(xs, target) returning the first index of target or -1.",
     ["assert find_first([1,2,3,2], 2) == 1"],
     ["assert find_first([1,2,3,2], 2) == 1",
      "assert find_first([], 1) == -1",
      "assert find_first([1], 1) == 0",
      "assert find_first([1,2,3], 9) == -1"],
     "no_empty_handling"),
    ("vc_011", "is_palindrome", "Write is_palindrome(s).",
     ["assert is_palindrome('racecar') is True"],
     ["assert is_palindrome('racecar') is True",
      "assert is_palindrome('') is True",
      "assert is_palindrome('a') is True",
      "assert is_palindrome('ab') is False"],
     "no_empty_or_singleton"),
    ("vc_012", "sum_positive", "Write sum_positive(xs) returning sum of strictly positive elements.",
     ["assert sum_positive([1,2,3]) == 6"],
     ["assert sum_positive([1,2,3]) == 6",
      "assert sum_positive([]) == 0",
      "assert sum_positive([-1,-2,3]) == 3",
      "assert sum_positive([0,0,0]) == 0"],
     "zero_or_negative_bug"),
    ("vc_013", "uniq_count", "Write uniq_count(xs) returning the number of distinct elements.",
     ["assert uniq_count([1,2,2,3]) == 3"],
     ["assert uniq_count([1,2,2,3]) == 3",
      "assert uniq_count([]) == 0",
      "assert uniq_count([1,1,1]) == 1"],
     "no_empty_handling"),
    ("vc_014", "abs_diff", "Write abs_diff(a, b) returning |a - b|.",
     ["assert abs_diff(5, 3) == 2"],
     ["assert abs_diff(5, 3) == 2",
      "assert abs_diff(3, 5) == 2",
      "assert abs_diff(-5, 5) == 10",
      "assert abs_diff(0, 0) == 0"],
     "no_negative_handling"),
    ("vc_015", "matrix_diag", "Write matrix_diag(m) returning the main diagonal of an n x n matrix as a list.",
     ["assert matrix_diag([[1,2],[3,4]]) == [1,4]"],
     ["assert matrix_diag([[1,2],[3,4]]) == [1,4]",
      "assert matrix_diag([[5]]) == [5]",
      "assert matrix_diag([[1,2,3],[4,5,6],[7,8,9]]) == [1,5,9]"],
     "no_singleton_or_3x3"),
]


_EVIDENCE_TRAPS = [
    # (id, question, gold_answer, authorized_doc, distractor_docs_containing_answer_phrase)
    # Distractor docs DELIBERATELY contain the gold-answer phrase verbatim
    # but discuss a different entity, so a weak verifier (gold phrase + any
    # citation) will FALSE-accept a citation to a distractor.
    ("ve_001", "Who discovered radium?", "Marie Curie",
     "doc_curie_marie", ["doc_curie_decoy"]),
    ("ve_002", "Who created Python?", "Guido van Rossum",
     "doc_python", ["doc_python_decoy"]),
    ("ve_003", "Who first walked on the Moon?", "Neil Armstrong",
     "doc_apollo", ["doc_armstrong_decoy"]),
    ("ve_004", "Who developed the theory of general relativity?", "Albert Einstein",
     "doc_einstein", ["doc_einstein_decoy"]),
    ("ve_005", "Who patented the telephone?", "Alexander Graham Bell",
     "doc_telephone", ["doc_telephone_decoy"]),
    ("ve_006", "Who wrote On the Origin of Species?", "Charles Darwin",
     "doc_darwin", ["doc_darwin_decoy"]),
    ("ve_007", "Who first summited Mount Everest?", "Edmund Hillary",
     "doc_everest", ["doc_everest_decoy"]),
    ("ve_008", "Who designed the Analytical Engine?", "Charles Babbage",
     "doc_babbage", ["doc_babbage_decoy"]),
    ("ve_009", "Who originated quantum theory?", "Max Planck",
     "doc_planck", ["doc_planck_decoy"]),
    ("ve_010", "Who patented the typewriter?", "Christopher Latham Sholes",
     "doc_typewriter", ["doc_typewriter_decoy"]),
    ("ve_011", "Who invented the lightbulb?", "Thomas Edison",
     "doc_edison", ["doc_edison_decoy"]),
    ("ve_012", "Who painted the Mona Lisa?", "Leonardo da Vinci",
     "doc_mona_lisa", ["doc_mona_lisa_decoy"]),
    ("ve_013", "Who composed the Ninth Symphony?", "Ludwig van Beethoven",
     "doc_beethoven", ["doc_beethoven_decoy"]),
    ("ve_014", "Who wrote Hamlet?", "William Shakespeare",
     "doc_shakespeare", ["doc_shakespeare_decoy"]),
    ("ve_015", "Who first proved the Pythagorean theorem in writing?", "Pythagoras",
     "doc_pythagoras", ["doc_pythagoras_decoy"]),
]


_DA_TRAPS = [
    # (id, question, gold, common_wrong, weak_accepts)
    # These are tasks where a typical small-model coding error produces a
    # specific wrong number that a weak verifier (looks for any integer) would accept.
    ("vd_001", "Given xs = [1, 2, 3, 4, 5], compute mean.", "3", "15",  # printed sum instead
     ["3", "15"]),
    ("vd_002", "Given xs = [1, 2, 3, 4, 5], compute median.", "3", "5",  # printed max
     ["3", "5"]),
    ("vd_003", "Given xs = [10, 20, 30], compute the index of the maximum value.", "2", "30",  # printed value
     ["2", "30"]),
    ("vd_004", "Given xs = [1, 2, 3, 4, 5], compute the variance (population).", "2", "10",
     ["2", "10"]),
    ("vd_005", "Given the list ['a', 'b', 'a', 'c'], how many distinct values?", "3", "4",  # used len not set
     ["3", "4"]),
    ("vd_006", "Given xs = [3, 1, 4, 1, 5, 9, 2, 6], how many ≥ 4?", "4", "8",  # printed total length
     ["4", "8"]),
    ("vd_007", "Given (k, v) = [('a',1),('b',2),('a',3)], compute total v for k='a'.", "4", "1",  # only first
     ["4", "1"]),
    ("vd_008", "Given xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], compute mode (return any single mode).", "1", "55",  # sum instead
     ["1", "55"]),
    ("vd_009", "Given the string 'banana', how many distinct characters?", "3", "6",  # len(s) not len(set(s))
     ["3", "6"]),
    ("vd_010", "Given xs = [10, 20, 30, 40], compute geometric mean rounded down to integer.", "22", "25",  # arithmetic instead
     ["22", "25"]),
]


# Decoy / distractor evidence corpus for evidence_traps. Each "decoy" doc
# contains the gold-answer phrase but discusses a different entity / context.
_EV_CORPUS = {
    "doc_curie_marie": "Marie Curie discovered radium and polonium in 1898 with her husband Pierre.",
    "doc_curie_decoy": "Marie Curie was the name of a fictional character in the novel 'Atomic Light' (2017) who was unrelated to actual radioactivity research.",
    "doc_python": "Python was created by Guido van Rossum and first released in 1991.",
    "doc_python_decoy": "Guido van Rossum is also the name of a Dutch chess instructor who has no relation to programming languages.",
    "doc_apollo": "The Apollo 11 mission, commanded by Neil Armstrong, landed humans on the Moon in July 1969.",
    "doc_armstrong_decoy": "Neil Armstrong is also a UK musician who released an album titled 'Lunar Echoes' in 2003 with no connection to space exploration.",
    "doc_einstein": "Albert Einstein developed the theory of general relativity, published in 1915.",
    "doc_einstein_decoy": "Albert Einstein is the name of a fictional cat in a 2018 children's book unrelated to physics.",
    "doc_telephone": "Alexander Graham Bell patented the telephone in 1876.",
    "doc_telephone_decoy": "Alexander Graham Bell is the name of a Scottish whisky distillery, not related to communication technology.",
    "doc_darwin": "Charles Darwin published On the Origin of Species in 1859.",
    "doc_darwin_decoy": "Charles Darwin is also the name of a 1990s television drama character with no relation to biology.",
    "doc_everest": "Mount Everest was first summited in 1953 by Edmund Hillary and Tenzing Norgay.",
    "doc_everest_decoy": "Edmund Hillary is the name of a Welsh poet active in the 1700s with no mountaineering activity.",
    "doc_babbage": "Charles Babbage designed the Analytical Engine in the 1830s.",
    "doc_babbage_decoy": "Charles Babbage is also the name of a 1980s comic-book artist with no engineering background.",
    "doc_planck": "Max Planck originated quantum theory in 1900.",
    "doc_planck_decoy": "Max Planck is also a German football coach with no physics involvement.",
    "doc_typewriter": "Christopher Latham Sholes patented the typewriter in 1868.",
    "doc_typewriter_decoy": "Christopher Latham Sholes was also a 20th-century jazz musician unrelated to office machinery.",
    "doc_edison": "Thomas Edison patented the incandescent lightbulb in 1879.",
    "doc_edison_decoy": "Thomas Edison is also the name of a Brazilian footballer with no inventions to his name.",
    "doc_mona_lisa": "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
    "doc_mona_lisa_decoy": "Leonardo da Vinci is also the name of an Italian fashion brand with no painting history.",
    "doc_beethoven": "Ludwig van Beethoven composed his Ninth Symphony, completed in 1824.",
    "doc_beethoven_decoy": "Ludwig van Beethoven was also the name of a Saint Bernard dog in a 1992 family comedy film.",
    "doc_shakespeare": "William Shakespeare wrote Hamlet around 1600.",
    "doc_shakespeare_decoy": "William Shakespeare is also a fictional barber in a children's musical from 2010.",
    "doc_pythagoras": "Pythagoras of Samos is traditionally credited with the first proof of the Pythagorean theorem.",
    "doc_pythagoras_decoy": "Pythagoras is also the name of a London restaurant specializing in Greek cuisine, with no mathematical association.",
}


def get_pool() -> dict:
    pool = {}
    for tid, q, gold, common_wrong, weak_accepts, audit in _MATH_TRAPS:
        pool[tid] = {"family": "math_traps", "id": tid, "question": q,
                     "gold_answer": gold, "common_wrong_answer": common_wrong,
                     "weak_verifier_accepts": weak_accepts,
                     "exact_verifier_audit": audit,
                     "difficulty": "trap", "provenance": "synthetic-local-apollo-trap"}
    for tid, ep, prompt, weak_tests, audit_tests, label in _CODE_TRAPS:
        pool[tid] = {"family": "code_traps", "id": tid, "entry_point": ep,
                     "prompt": prompt, "weak_tests": weak_tests,
                     "audit_tests": audit_tests, "common_wrong_label": label,
                     "difficulty": "trap", "provenance": "synthetic-local-apollo-trap"}
    for tid, q, gold, auth, decoys in _EVIDENCE_TRAPS:
        evidence = {auth: _EV_CORPUS.get(auth, ""),
                    **{d: _EV_CORPUS.get(d, "") for d in decoys}}
        pool[tid] = {"family": "evidence_traps", "id": tid, "question": q,
                     "gold_answer": gold, "authorized_citations": [auth],
                     "distractor_citations": list(decoys),
                     "evidence": evidence,
                     "difficulty": "trap", "provenance": "synthetic-local-apollo-trap"}
    for tid, q, gold, common_wrong, weak_accepts in _DA_TRAPS:
        pool[tid] = {"family": "data_analysis_traps", "id": tid, "question": q,
                     "gold_answer": gold, "common_wrong_answer": common_wrong,
                     "weak_verifier_accepts": weak_accepts,
                     "difficulty": "trap", "provenance": "synthetic-local-apollo-trap"}
    return pool


def main() -> int:
    pool = get_pool()
    fams = Counter(t["family"] for t in pool.values())
    out = {
        "n_total": len(pool),
        "family_counts": dict(fams),
        "tasks": [{"task_id": t["id"], "family": t["family"],
                   "difficulty": t["difficulty"]} for t in pool.values()],
        "honesty": (
            "Adversarial verifier-risk pool: each task is engineered so a *weak* "
            "verifier (loose pattern match: any-integer for numeric, gold-phrase + "
            "any-citation for evidence, weak-test-suite-only for code) can FALSE-"
            "accept a plausible wrong answer. An *exact* verifier (full audit suite) "
            "catches it. Tasks are auditable in this script. NOT a benchmark download."
        ),
    }
    out_path = APOLLO / "verifier_risk" / "experiments" / "verifier_risk_pool_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# VERIFIER_RISK_POOL\n",
          f"\n- total tasks: **{len(pool)}**\n",
          f"- families: {dict(fams)}\n",
          "\n## Honesty\n\n" + out["honesty"] + "\n"]
    (APOLLO / "verifier_risk" / "reports" / "VERIFIER_RISK_POOL.md").write_text(
        "".join(md), encoding="utf-8")
    print(f"families: {dict(fams)}; total {len(pool)}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
