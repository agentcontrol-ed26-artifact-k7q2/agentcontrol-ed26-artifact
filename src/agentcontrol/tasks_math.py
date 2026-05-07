from __future__ import annotations
from typing import Any

MATH_TASKS: list[dict[str, Any]] = [
    {'id':'m001','question':'What is 2 + 3?','answer':'5','cheap':'5','repair':'5','difficulty':'easy'},
    {'id':'m002','question':'What is 7 * 6?','answer':'42','cheap':'42','repair':'42','difficulty':'easy'},
    {'id':'m003','question':'If x + 4 = 9, what is x?','answer':'5','cheap':'5','repair':'5','difficulty':'easy'},
    {'id':'m004','question':'What is 15 - 8?','answer':'7','cheap':'7','repair':'7','difficulty':'easy'},
    {'id':'m005','question':'What is 12 / 3?','answer':'4','cheap':'4','repair':'4','difficulty':'easy'},
    {'id':'m006','question':'A box has 3 red and 5 blue balls. How many balls total?','answer':'8','cheap':'8','repair':'8','difficulty':'easy'},
    {'id':'m007','question':'If 3 notebooks cost $12, what is the cost of 5 notebooks?','answer':'20','cheap':'15','repair':'20','difficulty':'medium'},
    {'id':'m008','question':'A train travels 60 miles in 2 hours. How far in 5 hours?','answer':'150','cheap':'120','repair':'150','difficulty':'medium'},
    {'id':'m009','question':'What is the next number: 2, 4, 8, 16, ?','answer':'32','cheap':'24','repair':'32','difficulty':'medium'},
    {'id':'m010','question':'A rectangle has area 45 and width 5. What is its length?','answer':'9','cheap':'40','repair':'9','difficulty':'medium'},
    {'id':'m011','question':'What is 25% of 80?','answer':'20','cheap':'25','repair':'20','difficulty':'medium'},
    {'id':'m012','question':'Solve 2x - 3 = 11.','answer':'7','cheap':'8','repair':'7','difficulty':'medium'},
    {'id':'m013','question':'What is the sum of integers from 1 to 10?','answer':'55','cheap':'50','repair':'55','difficulty':'medium'},
    {'id':'m014','question':'If a fair coin is flipped twice, how many possible outcomes?','answer':'4','cheap':'2','repair':'4','difficulty':'medium'},
    {'id':'m015','question':'A number is doubled and then increased by 6 to get 22. What is the number?','answer':'8','cheap':'11','repair':'8','difficulty':'medium'},
    {'id':'m016','question':'What is 9 squared minus 5 squared?','answer':'56','cheap':'36','repair':'56','difficulty':'hard'},
    {'id':'m017','question':'If 4 workers build 4 tables in 4 hours, how many tables do 8 workers build in 8 hours?','answer':'16','cheap':'8','repair':'16','difficulty':'hard'},
    {'id':'m018','question':'What is the least common multiple of 6 and 8?','answer':'24','cheap':'48','repair':'24','difficulty':'hard'},
    {'id':'m019','question':'A price increases from 50 to 60. What is the percent increase?','answer':'20','cheap':'10','repair':'20','difficulty':'hard'},
    {'id':'m020','question':'If x/3 + 2 = 6, what is x?','answer':'12','cheap':'8','repair':'12','difficulty':'hard'},
]


def load_math_tasks(n: int | None = None) -> list[dict[str, Any]]:
    return MATH_TASKS[:n] if n else list(MATH_TASKS)


def format_dummy_math_prompt(task: dict[str, Any], mode: str = 'answer') -> str:
    return '\n'.join([
        f"MATH_TASK:{task['id']}",
        f"Question: {task['question']}",
        f"DUMMY_CHEAP: {task['cheap']}",
        f"DUMMY_REPAIR: {task['repair']}",
        f"DUMMY_STRONG: {task['answer']}",
        f"DUMMY_HINT: Check arithmetic carefully for {task['id']}.",
        f"Mode: {mode}",
    ])
