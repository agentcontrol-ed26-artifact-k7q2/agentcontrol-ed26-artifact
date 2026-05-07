"""Interactive task pool for the GPU Main Rescue Fork.

Each task supports observation actions (run_tests, run_code, retrieve, etc.)
that produce new state information. Verifiers are deterministic.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _stable(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# code_debug_interactive (n=40) — Python function tasks where test failure
# traces unblock cheap repair. Each task carries (entry_point, prompt, tests,
# common_first_attempt_bug) so the harness can simulate a realistic cheap
# attempt and a realistic test trace.
# ---------------------------------------------------------------------------
_CODE_DEBUG = [
    ("ic_001", "fib", "Write fib(n) returning the nth Fibonacci with fib(0)=0 fib(1)=1.",
     ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(7) == 13", "assert fib(15) == 610"]),
    ("ic_002", "is_prime", "Write is_prime(n) returning True iff n is a prime > 1.",
     ["assert is_prime(2) is True", "assert is_prime(1) is False", "assert is_prime(97) is True", "assert is_prime(4) is False"]),
    ("ic_003", "gcd", "Write gcd(a, b) returning the gcd of two positive integers.",
     ["assert gcd(12, 18) == 6", "assert gcd(7, 13) == 1", "assert gcd(100, 75) == 25"]),
    ("ic_004", "edit_distance", "Write edit_distance(a, b) returning Levenshtein distance.",
     ["assert edit_distance('kitten', 'sitting') == 3", "assert edit_distance('', 'abc') == 3", "assert edit_distance('abc', 'abc') == 0"]),
    ("ic_005", "longest_common_subseq_len", "Write longest_common_subseq_len(a, b) returning LCS length.",
     ["assert longest_common_subseq_len('abcde', 'ace') == 3", "assert longest_common_subseq_len('abc', 'def') == 0", "assert longest_common_subseq_len('AGCAT', 'GAC') == 2"]),
    ("ic_006", "longest_increasing_subseq", "Write longest_increasing_subseq(xs) returning length of LIS.",
     ["assert longest_increasing_subseq([10,9,2,5,3,7,101,18]) == 4", "assert longest_increasing_subseq([3,3,3]) == 1", "assert longest_increasing_subseq([]) == 0"]),
    ("ic_007", "binary_search", "Write binary_search(a, target) returning index in sorted list a or -1.",
     ["assert binary_search([1,3,5,7,9], 5) == 2", "assert binary_search([1,3,5,7,9], 6) == -1", "assert binary_search([], 1) == -1"]),
    ("ic_008", "merge_sorted", "Write merge_sorted(a, b) merging two sorted lists.",
     ["assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]", "assert merge_sorted([], [1]) == [1]", "assert merge_sorted([1,2,3], []) == [1,2,3]"]),
    ("ic_009", "rle_encode", "Write rle_encode(s) returning [(char, count)] run-length encoding.",
     ["assert rle_encode('aaabbc') == [('a',3),('b',2),('c',1)]", "assert rle_encode('') == []", "assert rle_encode('a') == [('a',1)]"]),
    ("ic_010", "matrix_transpose", "Write matrix_transpose(m) returning the transpose of a 2D list.",
     ["assert matrix_transpose([[1,2],[3,4]]) == [[1,3],[2,4]]", "assert matrix_transpose([[1,2,3]]) == [[1],[2],[3]]"]),
    ("ic_011", "balanced_parens", "Write balanced_parens(s) iff s has balanced parentheses among ()[]{}.",
     ["assert balanced_parens('()[]{}') is True", "assert balanced_parens('(]') is False", "assert balanced_parens('(([]){})') is True", "assert balanced_parens('') is True"]),
    ("ic_012", "count_set_bits", "Write count_set_bits(n) returning popcount of nonnegative n.",
     ["assert count_set_bits(0) == 0", "assert count_set_bits(7) == 3", "assert count_set_bits(255) == 8", "assert count_set_bits(1024) == 1"]),
    ("ic_013", "first_missing_positive", "Write first_missing_positive(xs) returning the smallest positive integer not in xs.",
     ["assert first_missing_positive([1,2,0]) == 3", "assert first_missing_positive([3,4,-1,1]) == 2", "assert first_missing_positive([]) == 1", "assert first_missing_positive([1,2,3]) == 4"]),
    ("ic_014", "single_number", "Write single_number(xs) returning the int that appears exactly once when others appear twice.",
     ["assert single_number([2,2,1]) == 1", "assert single_number([4,1,2,1,2]) == 4"]),
    ("ic_015", "is_anagram", "Write is_anagram(a, b) iff a and b are anagrams.",
     ["assert is_anagram('listen', 'silent') is True", "assert is_anagram('hello', 'world') is False", "assert is_anagram('', '') is True"]),
    ("ic_016", "rotate_matrix", "Write rotate_matrix(m) rotating an n x n matrix 90 degrees clockwise in place; mutate m and return None.",
     ["m=[[1,2],[3,4]]; rotate_matrix(m); assert m == [[3,1],[4,2]]", "m=[[1,2,3],[4,5,6],[7,8,9]]; rotate_matrix(m); assert m == [[7,4,1],[8,5,2],[9,6,3]]"]),
    ("ic_017", "knapsack_01", "Write knapsack_01(weights, values, capacity) returning max total value with each item used at most once.",
     ["assert knapsack_01([1,3,4,5], [1,4,5,7], 7) == 9", "assert knapsack_01([2], [3], 1) == 0"]),
    ("ic_018", "topological_sort", "Write topological_sort(n, edges) returning a valid topological ordering of nodes 0..n-1, or [] if cyclic.",
     ["assert topological_sort(2, [(0,1)]) == [0,1]", "result = topological_sort(4, [(0,1),(0,2),(1,3),(2,3)]); assert len(result) == 4 and result.index(0) < result.index(3)"]),
    ("ic_019", "subset_sum", "Write subset_sum(xs, target) returning True iff some subset of nonnegative integers xs sums to target.",
     ["assert subset_sum([1,2,3,7], 6) is True", "assert subset_sum([1,2,3,7], 14) is False", "assert subset_sum([], 0) is True"]),
    ("ic_020", "longest_common_substring_len", "Write longest_common_substring_len(a, b) returning length of longest common contiguous substring.",
     ["assert longest_common_substring_len('abcde', 'cdeab') == 3", "assert longest_common_substring_len('abc', 'def') == 0"]),
    ("ic_021", "lru_cache_class", "Write a class LRUCache(capacity) with get(key) -> value or -1, and put(key, value) evicting LRU.",
     ["c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4"]),
    ("ic_022", "find_duplicates", "Write find_duplicates(xs) returning sorted list of duplicates (each once).",
     ["assert find_duplicates([1,2,2,3,3,3,4]) == [2,3]", "assert find_duplicates([1,2,3]) == []"]),
    ("ic_023", "count_islands", "Write count_islands(grid) counting connected components of 1s in a 2D grid (4-connected).",
     ["assert count_islands([[1,1,0],[0,1,0],[0,0,1]]) == 2", "assert count_islands([[0]]) == 0"]),
    ("ic_024", "decode_string", "Write decode_string(s) decoding s like '3[a]2[bc]' -> 'aaabcbc'.",
     ["assert decode_string('3[a]2[bc]') == 'aaabcbc'", "assert decode_string('2[ab3[c]]') == 'abcccabccc'"]),
    ("ic_025", "max_subarray_sum", "Write max_subarray_sum(xs) returning max contiguous subarray sum (handle negatives).",
     ["assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6", "assert max_subarray_sum([-1]) == -1"]),
    ("ic_026", "all_permutations", "Write all_permutations(xs) returning sorted list of all permutations of xs (each as tuple).",
     ["assert all_permutations([1,2]) == [(1,2),(2,1)]", "assert all_permutations([]) == [()]"]),
    ("ic_027", "valid_sudoku_3x3", "Write valid_sudoku_3x3(grid) iff a 3x3 grid contains digits 1..9 with no repeats.",
     ["assert valid_sudoku_3x3([[1,2,3],[4,5,6],[7,8,9]]) is True", "assert valid_sudoku_3x3([[1,2,3],[4,5,6],[7,8,8]]) is False"]),
    ("ic_028", "dijkstra_distance", "Write dijkstra_distance(n, edges, src, dst) for non-negative weighted edges; return distance or -1 if unreachable.",
     ["assert dijkstra_distance(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0, 3) == 4", "assert dijkstra_distance(2, [], 0, 1) == -1"]),
    ("ic_029", "json_path_lookup", "Write json_path_lookup(d, path) returning value at dotted path or None. Path is a string like 'a.b.c'.",
     ["assert json_path_lookup({'a':{'b':{'c':1}}}, 'a.b.c') == 1", "assert json_path_lookup({'a':1}, 'b') is None"]),
    ("ic_030", "interval_merge", "Write interval_merge(intervals) merging overlapping [start, end] tuples into a sorted list of merged tuples.",
     ["assert interval_merge([(1,3),(2,6),(8,10),(15,18)]) == [(1,6),(8,10),(15,18)]", "assert interval_merge([]) == []"]),
    ("ic_031", "string_compress", "Write string_compress(s) compressing run-lengths like 'aaabb' -> 'a3b2'; if not shorter than s, return s.",
     ["assert string_compress('aaabb') == 'a3b2'", "assert string_compress('ab') == 'ab'"]),
    ("ic_032", "n_queens_count", "Write n_queens_count(n) returning the number of solutions to the N-queens problem for n in [1..8].",
     ["assert n_queens_count(1) == 1", "assert n_queens_count(4) == 2", "assert n_queens_count(8) == 92"]),
    ("ic_033", "spiral_order", "Write spiral_order(m) traversing a 2D matrix in spiral (clockwise) order, returning a flat list.",
     ["assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]", "assert spiral_order([[1]]) == [1]"]),
    ("ic_034", "two_sum_sorted", "Write two_sum_sorted(xs, target) returning a 0-indexed (i, j) tuple with i<j and xs[i]+xs[j]==target, or None.",
     ["assert two_sum_sorted([1,2,4,7,11], 9) == (1,3)", "assert two_sum_sorted([1,2,3], 100) is None"]),
    ("ic_035", "trap_rain_water", "Write trap_rain_water(heights) returning total trapped rainwater above bars (classic problem).",
     ["assert trap_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6", "assert trap_rain_water([4,2,0,3,2,5]) == 9"]),
    ("ic_036", "word_break", "Write word_break(s, words) iff s can be segmented into a sequence from words (each word usable many times).",
     ["assert word_break('leetcode', {'leet','code'}) is True", "assert word_break('applepenapple', {'apple','pen'}) is True", "assert word_break('catsandog', {'cats','dog','sand','and','cat'}) is False"]),
    ("ic_037", "find_anagrams", "Write find_anagrams(s, p) returning sorted list of start indices of anagrams of p in s.",
     ["assert find_anagrams('cbaebabacd', 'abc') == [0,6]", "assert find_anagrams('abab', 'ab') == [0,1,2]"]),
    ("ic_038", "course_schedule_possible", "Write course_schedule_possible(n, prereqs) iff all n courses can be completed given (a, b) edges meaning a needs b first.",
     ["assert course_schedule_possible(2, [(1,0)]) is True", "assert course_schedule_possible(2, [(1,0),(0,1)]) is False"]),
    ("ic_039", "next_permutation", "Write next_permutation(xs) mutating xs to the next lexicographic permutation in place; if at last, set to first. Return None.",
     ["x=[1,2,3]; next_permutation(x); assert x == [1,3,2]", "x=[3,2,1]; next_permutation(x); assert x == [1,2,3]", "x=[1,1,5]; next_permutation(x); assert x == [1,5,1]"]),
    ("ic_040", "word_ladder_len", "Write word_ladder_len(begin, end, words) returning the shortest transformation length (changing one letter at a time within words), or 0 if impossible.",
     ["assert word_ladder_len('hit', 'cog', ['hot','dot','dog','lot','log','cog']) == 5", "assert word_ladder_len('hit', 'cog', ['hot','dot','dog','lot','log']) == 0"]),
]


# data_analysis_code (n=20) — small inline tabular tasks where running code reveals data.
_DATA_ANALYSIS = [
    ("id_001", "Given the list xs = [4, 8, 15, 16, 23, 42], compute the sum.", "108"),
    ("id_002", "Given xs = [4, 8, 15, 16, 23, 42], compute the mean to nearest integer.", "18"),
    ("id_003", "Given xs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], how many distinct values?", "6"),
    ("id_004", "Given a list of (name, score): [('A',80),('B',95),('C',72),('D',95)], how many have score >= 90?", "2"),
    ("id_005", "Given xs = [1,2,3,4,5,6,7,8,9,10], compute the median.", "5.5"),
    ("id_006", "Given xs = [10, 5, 0, -5, 10, 20], compute the range (max - min).", "25"),
    ("id_007", "Given (a,b) pairs [(1,2),(3,4),(5,6),(7,8)], compute sum of a*b across pairs.", "100"),
    ("id_008", "Given xs = [1,2,2,3,3,3,4,4,4,4], what is the mode?", "4"),
    ("id_009", "Given xs = [10, 20, 30, 40, 50], compute the variance (population).", "200"),
    ("id_010", "Given strings ['apple','banana','cherry','apple','banana','apple'], count unique strings.", "3"),
    ("id_011", "Given xs = list(range(1, 21)), compute sum of even values.", "110"),
    ("id_012", "Given xs = [3, 7, 11, 13, 17, 19, 23], how many primes? (all of them)", "7"),
    ("id_013", "Given (price, qty) = [(10, 3), (5, 4), (20, 1)], compute total revenue.", "70"),
    ("id_014", "Given a 2D list [[1,2],[3,4],[5,6]], compute the column-wise sums.", "[9, 12]"),
    ("id_015", "Given xs = [4, 8, 15, 16, 23, 42], compute the standard deviation (population, to 2 decimals).", "13.16"),
    ("id_016", "Given xs = [1,1,2,3,5,8,13,21,34], how many odd values?", "6"),
    ("id_017", "Given a list of dicts [{'k':1},{'k':2},{'k':3}], compute sum of 'k' values.", "6"),
    ("id_018", "Given a 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]], compute the trace (diagonal sum).", "15"),
    ("id_019", "Given the string 'mississippi', count occurrences of letter 's'.", "4"),
    ("id_020", "Given xs = [-3, -1, 0, 1, 3, 5, 7], compute the count of values >= 1.", "4"),
]


# evidence_multihop_local (n=30) — frozen 20-doc corpus + multi-hop QA + distractors.
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
    ("ie_001", "Who discovered radium?", "Marie Curie", ["doc_curie"], False),
    ("ie_002", "What did Marie Curie discover besides radium?", "polonium", ["doc_curie"], False),
    ("ie_003", "Who created Python?", "Guido van Rossum", ["doc_python"], False),
    ("ie_004", "In what year was Python first released?", "1991", ["doc_python"], False),
    ("ie_005", "Where was the Python creator born?", "Netherlands", ["doc_python", "doc_python2"], True),
    ("ie_006", "Which mission first landed humans on the Moon?", "Apollo 11", ["doc_apollo"], False),
    ("ie_007", "Who commanded the first lunar landing mission?", "Neil Armstrong", ["doc_apollo", "doc_armstrong"], True),
    ("ie_008", "In what state was the first person to walk on the Moon born?", "Ohio", ["doc_apollo", "doc_armstrong"], True),
    ("ie_009", "What did Albert Einstein win the Nobel Prize for?", "photoelectric effect", ["doc_einstein"], False),
    ("ie_010", "In what year did Einstein publish general relativity?", "1915", ["doc_einstein"], False),
    ("ie_011", "Who originated quantum theory in 1900?", "Max Planck", ["doc_planck"], False),
    ("ie_012", "Whose X-ray diffraction images contributed to the DNA double helix discovery?", "Rosalind Franklin", ["doc_dna", "doc_franklin"], True),
    ("ie_013", "In what year was the DNA double helix elucidated?", "1953", ["doc_dna"], False),
    ("ie_014", "When did World War II end?", "1945", ["doc_war"], False),
    ("ie_015", "Who patented the telephone in 1876?", "Alexander Graham Bell", ["doc_telephone"], False),
    ("ie_016", "Who wrote On the Origin of Species?", "Charles Darwin", ["doc_evolution"], False),
    ("ie_017", "On what ship did Darwin travel from 1831 to 1836?", "HMS Beagle", ["doc_evolution", "doc_darwin"], True),
    ("ie_018", "What is the second longest river in the world?", "Amazon", ["doc_amazon"], False),
    ("ie_019", "Who first summited Mount Everest in 1953?", "Edmund Hillary", ["doc_everest"], False),
    ("ie_020", "Who is described as the first computer programmer for the Analytical Engine?", "Ada Lovelace", ["doc_lovelace"], False),
    ("ie_021", "Who designed the Analytical Engine?", "Charles Babbage", ["doc_babbage"], False),
    ("ie_022", "Which Nobel laureate is associated with the Analytical Engine programmer's machine?", "Charles Babbage", ["doc_lovelace", "doc_babbage"], True),
    ("ie_023", "Who shared the 1903 Nobel in Physics with Becquerel and discovered radium?", "Marie Curie", ["doc_curie", "doc_curie2"], True),
    ("ie_024", "Where did the Wright brothers achieve their first sustained flight?", "Kitty Hawk", ["doc_wright"], False),
    ("ie_025", "In what year did the Wright brothers achieve their first sustained flight?", "1903", ["doc_wright"], False),
    ("ie_026", "What river flows through Egypt and Sudan and is considered the longest?", "Nile", ["doc_nile"], False),
    ("ie_027", "Who climbed Everest in 1953 alongside Tenzing Norgay?", "Edmund Hillary", ["doc_everest"], False),
    ("ie_028", "Who led the Nazi regime during World War II?", "Adolf Hitler", ["doc_war"], False),
    ("ie_029", "When did Charles Darwin publish On the Origin of Species?", "1859", ["doc_evolution"], False),
    ("ie_030", "Where was Charles Darwin born?", "Shrewsbury", ["doc_darwin"], False),
]


# tool_planning_deterministic (n=15) — small grid/state environments.
_TOOL = [
    ("it_001", "A robot at (0,0) on a 3x3 grid. Goal at (2,2). Each move costs 1; obstacles at (1,1). Minimum total cost path length?", "4"),
    ("it_002", "Same grid, obstacles at (0,1) and (1,0). Min cost from (0,0) to (2,2)?", "8"),
    ("it_003", "Maze (5x5) start (0,0) goal (4,4). Walls form a U: (1,1),(1,2),(1,3),(2,3),(3,3). Shortest distance?", "8"),
    ("it_004", "Tower of Hanoi: move 4 disks from peg A to peg C. Minimum number of moves?", "15"),
    ("it_005", "8-puzzle: starting [[1,2,3],[4,5,6],[7,0,8]], goal [[1,2,3],[4,5,6],[7,8,0]]. Min moves?", "1"),
    ("it_006", "Robot must visit 4 named cities A=(0,0) B=(1,0) C=(1,1) D=(0,1) and return to A in any order, taking Manhattan distance. Min total tour length?", "4"),
    ("it_007", "Bridge crossing: 4 people times {1,2,5,10} minutes; one torch; max 2 cross at a time, slower person sets pace. Minimum total minutes?", "17"),
    ("it_008", "Coin change: target 11 cents, denominations [1, 5, 10]. Minimum number of coins?", "2"),
    ("it_009", "Knapsack: capacity 7, items (weight, value) = [(3,4),(4,5),(2,3),(5,6)]. Maximum value (each item once)?", "9"),
    ("it_010", "Job scheduling: jobs (deadline, profit) = [(2,100),(1,19),(2,27),(1,25),(3,15)]; one job per slot, integer slots starting at 1, choose at most one job per slot, profit only if completed by deadline. Maximum total profit?", "142"),
    ("it_011", "8-queens: number of distinct solutions on a standard 8x8 board?", "92"),
    ("it_012", "Water jugs: jug A 5L, jug B 3L. Goal: measure exactly 4L using fill, empty, pour. Min number of operations?", "6"),
    ("it_013", "Light bulbs: 100 bulbs initially off; on round k (k from 1 to 100) toggle every kth bulb. After all rounds, how many bulbs are on?", "10"),
    ("it_014", "Burning rope: 2 ropes each take 60 min to burn unevenly; using only matches and ropes, can you measure exactly 45 min, and if so, what is min ropes used? Answer with the minimum count.", "2"),
    ("it_015", "Missionaries and cannibals classic puzzle: 3 of each on left bank, boat carries 1-2, never outnumber missionaries. Min crossings to move all to right bank?", "11"),
]


# math_checkpoint (n=15) — multi-step problems with intermediate-checkpoint verifier.
_MATH_CHECK = [
    ("im_001", "Compute (3+4)*(5-2) where the intermediate step (3+4) should equal 7.", "21", "7"),
    ("im_002", "Compute the area of a triangle with base 8 and height 5; intermediate is base*height = 40.", "20", "40"),
    ("im_003", "Compute mean of [10, 20, 30, 40]; intermediate sum is 100.", "25", "100"),
    ("im_004", "Compute the hypotenuse of right triangle with legs 3 and 4; intermediate is 9+16=25.", "5", "25"),
    ("im_005", "Compute discriminant of x^2-5x+6; intermediate is b^2 = 25 and 4ac = 24.", "1", "1"),
    ("im_006", "If 3x+5=20, x=?; intermediate is 3x = 15.", "5", "15"),
    ("im_007", "Compute volume of a cube with side 4; intermediate is 4^2 = 16.", "64", "16"),
    ("im_008", "Sum of first 20 positive integers; intermediate uses formula n(n+1)/2 with n=20 giving 20*21=420.", "210", "420"),
    ("im_009", "Compute 12 choose 2; intermediate is 12*11 = 132.", "66", "132"),
    ("im_010", "Compute sum of digits of 999; intermediate observation: each digit is 9.", "27", "9"),
    ("im_011", "Compute LCM(8, 12); intermediate is GCD(8,12) = 4.", "24", "4"),
    ("im_012", "Compute total cents in 3 quarters and 4 dimes and 5 nickels; intermediate values are 75, 40, 25.", "140", "75"),
    ("im_013", "Compute 2^10; intermediate is 2^5 = 32.", "1024", "32"),
    ("im_014", "Solve x/3 + 4 = 10; intermediate is x/3 = 6.", "18", "6"),
    ("im_015", "If price drops from 80 to 60, percent decrease? Intermediate is the drop = 20.", "25", "20"),
]


def get_pool() -> dict[str, dict[str, Any]]:
    pool = {}
    for tid, ep, prompt, tests in _CODE_DEBUG:
        pool[tid] = {"family": "code_debug_interactive", "id": tid,
                     "entry_point": ep, "prompt": prompt, "tests": tests,
                     "interactive": True,
                     "available_observations": ["run_tests"],
                     "difficulty": "medium" if len(tests) <= 3 else "hard"}
    for tid, q, a in _DATA_ANALYSIS:
        pool[tid] = {"family": "data_analysis_code", "id": tid,
                     "question": q, "answer": a,
                     "interactive": True,
                     "available_observations": ["run_code"],
                     "difficulty": "medium"}
    for tid, q, a, citations, multihop in _EVIDENCE:
        ev_subset = {c: _EV_CORPUS[c] for c in citations}
        # Add 2 distractors deterministically for multihop tasks; 1 otherwise.
        all_docs = sorted(set(_EV_CORPUS.keys()) - set(citations))
        n_dist = 3 if multihop else 1
        h = _stable(tid)
        distractors = [all_docs[(h + i) % len(all_docs)] for i in range(n_dist)]
        for d in distractors:
            ev_subset[d] = _EV_CORPUS[d]
        pool[tid] = {"family": "evidence_multihop_local", "id": tid,
                     "question": q, "answer": a, "citations": citations,
                     "evidence": ev_subset, "distractors": distractors,
                     "multihop": multihop, "interactive": True,
                     "available_observations": ["retrieve", "citation_check"],
                     "difficulty": "hard" if multihop else "medium"}
    for tid, q, a in _TOOL:
        pool[tid] = {"family": "tool_planning_deterministic", "id": tid,
                     "question": q, "answer": a, "interactive": True,
                     "available_observations": ["tool_observation"],
                     "difficulty": "hard"}
    for tid, q, a, intermediate in _MATH_CHECK:
        pool[tid] = {"family": "math_checkpoint", "id": tid,
                     "question": q, "answer": a, "intermediate": intermediate,
                     "interactive": True,
                     "available_observations": ["checkpoint_check"],
                     "difficulty": "medium"}
    return pool


# ---------------------------------------------------------------------------
# Verifiers (deterministic).
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.strip().lower()


def _last_int(s: str) -> str | None:
    nums = re.findall(r"-?\d+", (s or "").replace(",", ""))
    return nums[-1] if nums else None


def _extract_python(output: str) -> str:
    if not output:
        return ""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return output.strip()


def run_tests_observation(task: dict, model_output: str) -> tuple[bool, str]:
    """Execute the model's emitted code and run the gold tests.

    Returns (all_passed, observation_text). The observation_text is a short
    deterministic summary suitable for inclusion in a follow-up prompt.
    """
    code = _extract_python(model_output)
    if not code:
        return False, "ERROR: no code block extracted from model output."
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<obs>", "exec"), ns, ns)  # noqa: S102
    except Exception as e:
        return False, f"ERROR during definition: {type(e).__name__}: {str(e)[:200]}"
    if task["family"] == "code_debug_interactive" and task["entry_point"] not in ns:
        return False, f"ERROR: function `{task['entry_point']}` not defined."
    failures = []
    for t in task["tests"]:
        try:
            exec(compile(t, "<obs_test>", "exec"), ns, ns)  # noqa: S102
        except AssertionError:
            failures.append(t)
        except Exception as e:
            failures.append(f"{t}  --> {type(e).__name__}: {str(e)[:80]}")
        if len(failures) >= 2:
            break
    if not failures:
        return True, "All tests passed."
    return False, "Test failures:\n" + "\n".join(f"  - {f}" for f in failures)


def run_code_observation(task: dict, model_output: str) -> tuple[bool, str]:
    """For data_analysis_code: extract last printed value or eval result.

    Many model attempts will write code that prints the answer; we run it and
    capture stdout. If no stdout, fall back to scanning the model's prose.
    """
    import io
    import contextlib

    code = _extract_python(model_output)
    if not code:
        return False, "ERROR: no code block extracted."
    buf = io.StringIO()
    ns: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<obs_code>", "exec"), ns, ns)  # noqa: S102
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {str(e)[:200]}"
    out = buf.getvalue().strip()
    if not out:
        return False, "ERROR: code did not print a result."
    return True, f"Output: {out[-500:]}"


def retrieve_observation(task: dict, query_terms: list[str] | None = None) -> str:
    """Return top-2 docs by simple term overlap with the question."""
    q = (task["question"] + " " + " ".join(query_terms or [])).lower()
    scored = []
    for k, v in task["evidence"].items():
        score = sum(1 for tok in re.split(r"\W+", q) if tok and tok in v.lower())
        scored.append((score, k, v))
    scored.sort(reverse=True)
    top = [(k, v) for _, k, v in scored[:2]]
    return "Retrieved docs:\n" + "\n".join(f"  {k}: {v}" for k, v in top)


def citation_check_observation(task: dict, model_output: str) -> tuple[bool, str]:
    """Citation discipline observation: per-cited-doc accept/reject WITHOUT
    revealing which docs are authorized. The model is told that some of its
    citations are not in the supporting evidence; it must reread and pick a
    citation that strictly supports the answer.
    """
    cited = re.findall(r"\[(doc_[a-zA-Z0-9_]+)\]", model_output or "")
    if not cited:
        return False, "No citation found in your answer. Provide exactly one [doc_xxx] that supports the answer."
    valid = [c for c in cited if c in task["citations"]]
    invalid = [c for c in cited if c not in task["citations"]]
    if invalid and not valid:
        return False, "All citations cite documents that do not support the answer. Reread the evidence and cite a document whose text contains the answer."
    if invalid:
        return False, "Some citations are not in the supporting evidence. Provide exactly one [doc_xxx] citation that supports the answer."
    return True, "Citation discipline: accepted."


def checkpoint_check_observation(task: dict, model_output: str) -> tuple[bool, str]:
    """Checkpoint observation: BINARY accept/reject of model's intermediate
    value without revealing the gold intermediate.

    This is oracle-binary feedback. The paper describes it as such; it is
    NOT a production checker that returns the correct value.
    """
    expected = task["intermediate"]
    last = _last_int(model_output)
    if last == expected:
        return True, "Checkpoint: intermediate value accepted."
    return False, "Checkpoint: intermediate value rejected. Recompute the intermediate step before producing the final answer."


def tool_observation_observation(task: dict, model_output: str) -> tuple[bool, str]:
    """Tool observation: BINARY accept/reject without revealing the gold answer.

    This is a deterministic verifier-style oracle (yes/no), not a production
    tool. It tells the model whether its current candidate is accepted; it
    does NOT reveal the correct value. The paper must describe this honestly
    as oracle-binary feedback, not a real-world tool observation.
    """
    answer = task["answer"]
    last = _last_int(model_output)
    if last == answer:
        return True, "Verifier: current candidate accepted."
    return False, "Verifier: current candidate rejected. Re-derive the answer step by step using a different approach."


# ---------------------------------------------------------------------------
# Final verifiers (terminal success check).
# ---------------------------------------------------------------------------

def verify_code(task: dict, output: str) -> bool:
    return run_tests_observation(task, output)[0]


def verify_data(task: dict, output: str) -> bool:
    """Accept if model's output contains the gold answer string (numeric or list)."""
    if not output:
        return False
    answer = task["answer"]
    out = output.strip()
    if answer.startswith("[") or "." in answer:
        return answer in out  # exact substring for lists / floats
    # numeric
    last = _last_int(out)
    if last is not None and last == answer:
        return True
    return answer in out


def verify_evidence(task: dict, output: str) -> tuple[bool, float]:
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


def verify_tool(task: dict, output: str) -> bool:
    answer = task["answer"]
    last = _last_int(output or "")
    return last is not None and last == answer


def verify_math_checkpoint(task: dict, output: str) -> bool:
    answer = task["answer"]
    last = _last_int(output or "")
    return last is not None and last == answer


def verify(task: dict, output: str) -> tuple[bool, float]:
    fam = task["family"]
    if fam == "code_debug_interactive":
        return verify_code(task, output), 0.0
    if fam == "data_analysis_code":
        return verify_data(task, output), 0.0
    if fam == "evidence_multihop_local":
        return verify_evidence(task, output)
    if fam == "tool_planning_deterministic":
        return verify_tool(task, output), 0.0
    if fam == "math_checkpoint":
        return verify_math_checkpoint(task, output), 0.0
    raise ValueError(f"unknown family {fam}")
