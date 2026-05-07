"""F2 extension: data_analysis_code n=20 → n=50.

Inherits the original 20 tasks from main_rescue_gpu/interactive_tasks.py
(id_001..id_020) and adds 30 new locally-constructed tasks (id_021..id_050)
spanning harder regimes:
  - multi-step aggregation
  - conditional filtering
  - group-by counts and sums
  - weighted statistics
  - regex / text manipulation
  - time / percentage arithmetic
  - simple SQL-ish questions over inline tables

All tasks have inline data and deterministic verifiers. `run_code` returns
the model's emitted stdout (real, not gold-leaky).

Each task carries: task_id, family, question, answer, difficulty, provenance.
"""
from __future__ import annotations

import re
import sys
from typing import Any
from pathlib import Path


# Inherit the original n=20 from main_rescue_gpu's interactive task pool.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "main_rescue_gpu" / "src"))
from agentcontrol_main_rescue.interactive_tasks import get_pool as _legacy_pool  # noqa: E402


# 30 new tasks (id_021..id_050).
_NEW_DATA_ANALYSIS = [
    # easy_saturation continuation (5 new easy)
    ("id_021", "Given xs = [2, 4, 6, 8, 10, 12], compute the product of all elements.", "46080", "easy"),
    ("id_022", "Given xs = ['cat', 'dog', 'bird', 'fish', 'cat', 'dog', 'cat'], how many distinct strings?", "4", "easy"),
    ("id_023", "Given temperatures = [-5, 0, 12, 25, 18, -3, 30, 4], how many readings are above freezing (>0)?", "5", "easy"),
    ("id_024", "Given xs = list(range(1, 31)), compute the sum of elements that are multiples of 3.", "165", "easy"),
    ("id_025", "Given grades = [85, 72, 91, 68, 95, 79, 88, 76], what is the average rounded to nearest integer?", "82", "easy"),

    # medium (15 new medium)
    ("id_026", "Given (name, score) = [('A',82),('B',91),('C',75),('D',88),('E',93),('F',79)], how many scored at least 85?", "3", "medium"),
    ("id_027", "Given xs = [10, 20, 15, 30, 25, 40, 35], compute the difference between the maximum and the minimum.", "30", "medium"),
    ("id_028", "Given (item, price, qty) = [('A',5,3),('B',10,2),('C',8,5),('D',12,1)], compute total revenue.", "87", "medium"),
    ("id_029", "Given xs = [4, 1, 7, 3, 9, 2, 8, 6, 5], how many strictly greater than 5?", "4", "medium"),
    ("id_030", "Given the string 'banana banana cherry banana cherry apple', how many times does 'banana' occur as a word?", "3", "medium"),
    ("id_031", "Given xs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9], compute the median (middle element of sorted xs).", "5", "medium"),
    ("id_032", "Given pairs = [(2,12),(4,24),(6,36)], compute the value of y when x = 10, assuming y = k*x for some integer k.", "60", "medium"),
    ("id_033", "Given xs = [12, 7, 9, 14, 5, 11, 8, 15, 10, 6], how many elements are within 3 of the mean?", "6", "medium"),
    ("id_034", "Given (department, salary) = [('eng',100),('eng',120),('hr',60),('hr',70),('eng',110),('hr',65)], compute the average salary in eng.", "110", "medium"),
    ("id_035", "Given a 2D list [[1,2,3,4],[5,6,7,8],[9,10,11,12]], compute the sum of the second column (index 1).", "18", "medium"),
    ("id_036", "Given xs = [1, 4, 9, 16, 25, 36, 49, 64], how many are perfect squares of even numbers?", "4", "medium"),
    ("id_037", "Given the string 'The quick brown fox jumps over the lazy dog', how many words have length >= 4?", "5", "medium"),
    ("id_038", "Given (year, sales) = [(2018, 100),(2019, 120),(2020, 80),(2021, 150),(2022, 200)], compute the largest year-over-year change as a positive integer.", "70", "medium"),
    ("id_039", "Given xs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], compute the sum of unique values.", "30", "medium"),
    ("id_040", "Given (gender, age) = [('M',25),('F',30),('M',45),('F',22),('M',38),('F',38),('F',30)], compute the average age of all rows where gender == 'F'.", "30", "medium"),

    # hard (10 new hard) — multi-step / weighted / group-by-aggregate
    ("id_041", "Given (price, qty) = [(10,3),(20,2),(15,5),(8,7),(25,1)], compute the weighted average price (weighted by qty), rounded down to nearest integer.", "12", "hard"),
    ("id_042", "Given xs = [3, 5, 2, 8, 1, 4, 9, 7, 8], how many strictly increasing adjacent pairs (xs[i] < xs[i+1]) are there?", "5", "hard"),
    ("id_043", "Given the string 'aA1bB2cC3dD4eE5', count the number of lowercase letter characters.", "5", "hard"),
    ("id_044", "Given table = [(1,'A','x',10),(2,'B','y',20),(3,'A','y',30),(4,'B','x',15),(5,'A','x',25)] with columns id, group, kind, value, compute the total value where group=='A' and kind=='x'.", "35", "hard"),
    ("id_045", "Given (timestamp_min, value) = [(0,10),(15,20),(30,15),(45,25),(60,30),(75,20)], compute the simple integer sum of value for timestamps in the last 30 minutes (>=45).", "75", "hard"),
    ("id_046", "Given xs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], compute the count of elements in the upper quartile (>= 75th percentile rounded to integer 75).", "3", "hard"),
    ("id_047", "Given the integer matrix [[1,3,5,7],[2,4,6,8],[9,11,13,15],[10,12,14,16]], compute the sum of elements on the main anti-diagonal (top-right to bottom-left).", "34", "hard"),
    ("id_048", "Given (sku, returns) = [('A1',2),('B2',5),('A1',1),('C3',3),('B2',2),('A1',4),('D4',1)], compute the total returns for sku 'A1'.", "7", "hard"),
    ("id_049", "Given the string 'one TWO three FOUR five SIX seven EIGHT', count words written entirely in uppercase letters.", "4", "hard"),
    ("id_050", "Given xs = [4, -3, 7, -1, 2, 0, -5, 6, -8, 1, 3], compute the sum of strictly positive values minus the sum of strictly negative values (i.e. abs sum without zeros).", "40", "hard"),
]


# Lift legacy 20 → standardised dict layout that mirrors `_NEW_DATA_ANALYSIS`.
def _legacy_data_analysis() -> list[tuple[str, str, str, str]]:
    legacy = _legacy_pool()
    out = []
    for tid, t in legacy.items():
        if t["family"] != "data_analysis_code":
            continue
        # Heuristic difficulty by id position.
        n = int(tid.split("_")[1])
        diff = "easy" if n <= 6 else "medium" if n <= 14 else "hard"
        out.append((tid, t["question"], t["answer"], diff))
    return out


def get_pool() -> dict[str, dict[str, Any]]:
    """All 50 data_analysis_code tasks combining legacy n=20 + new n=30."""
    pool: dict[str, dict[str, Any]] = {}
    for tid, q, a, d in _legacy_data_analysis():
        pool[tid] = {
            "family": "data_analysis_code", "id": tid,
            "question": q, "answer": a, "difficulty": d,
            "provenance": "synthetic-local-legacy-from-main-rescue-gpu",
            "interactive": True, "available_observations": ["run_code"],
        }
    for tid, q, a, d in _NEW_DATA_ANALYSIS:
        pool[tid] = {
            "family": "data_analysis_code", "id": tid,
            "question": q, "answer": a, "difficulty": d,
            "provenance": "synthetic-local-extension-f2",
            "interactive": True, "available_observations": ["run_code"],
        }
    return pool


# ---------------------------------------------------------------------------
# Verifiers + observation reused from main_rescue_gpu / interactive_tasks.
# Imported locally to avoid duplication.
# ---------------------------------------------------------------------------
from agentcontrol_main_rescue.interactive_tasks import (  # noqa: E402,F401
    run_code_observation, verify as _legacy_verify,
)


def verify(task: dict, output: str) -> tuple[bool, float]:
    """Family-dispatched verifier. Returns (success, unsupported_risk)."""
    if task["family"] != "data_analysis_code":
        raise ValueError(f"unexpected family {task['family']}")
    return _legacy_verify(task, output)
