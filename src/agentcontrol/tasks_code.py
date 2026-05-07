from __future__ import annotations
from typing import Any

CODE_TASKS: list[dict[str, Any]] = [
    {'id':'code_add','entry_point':'add','prompt':'Write add(a, b) returning the sum.','tests':['assert add(2, 3) == 5','assert add(-1, 1) == 0']},
    {'id':'code_is_even','entry_point':'is_even','prompt':'Write is_even(n) returning True iff n is even.','tests':['assert is_even(2) is True','assert is_even(3) is False']},
    {'id':'code_factorial','entry_point':'factorial','prompt':'Write factorial(n) for nonnegative integer n.','tests':['assert factorial(0) == 1','assert factorial(5) == 120']},
    {'id':'code_reverse_words','entry_point':'reverse_words','prompt':'Write reverse_words(s) reversing word order, not characters.','tests':["assert reverse_words('one two three') == 'three two one'"]},
]


def load_code_tasks(n: int | None = None) -> list[dict[str, Any]]:
    return CODE_TASKS[:n] if n else list(CODE_TASKS)


def format_dummy_code_prompt(task: dict[str, Any], mode: str = 'answer') -> str:
    return '\n'.join([f"CODE_TASK:{task['id']}", f"Entry point: {task['entry_point']}", f"Prompt: {task['prompt']}", f"Mode: {mode}"])
