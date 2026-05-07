from __future__ import annotations
from .deliberation_graph import Action


def always_cheapest_plan() -> list[str]: return [Action.CHEAP_ANSWER.value]
def always_strongest_plan() -> list[str]: return [Action.STRONG_ANSWER.value]
def frugalgpt_cascade_plan() -> list[str]: return [Action.CHEAP_ANSWER.value, f'{Action.STRONG_ANSWER.value}_if_needed']
def automix_self_verification_cascade_plan() -> list[str]: return [Action.CHEAP_ANSWER.value, f'{Action.CHEAP_REPAIR.value}_if_needed', f'{Action.STRONG_ANSWER.value}_if_needed']
def shepherding_hint_plan() -> list[str]: return [Action.CHEAP_ANSWER.value, f'{Action.STRONG_HINT.value}_if_needed', f'{Action.CHEAP_REPAIR_AFTER_HINT.value}_if_needed']
def heuristic_bdelg_plan() -> list[str]: return [Action.CHEAP_ANSWER.value, f'{Action.CHEAP_REPAIR.value}_if_needed', f'{Action.STRONG_HINT.value}_if_needed', f'{Action.CHEAP_REPAIR_AFTER_HINT.value}_if_needed', f'{Action.STRONG_ANSWER.value}_if_needed']

BASELINE_PLANS = {
    'always_cheapest': always_cheapest_plan(),
    'always_strongest': always_strongest_plan(),
    'frugalgpt_cascade': frugalgpt_cascade_plan(),
    'automix_self_verification_cascade': automix_self_verification_cascade_plan(),
    'shepherding_hint': shepherding_hint_plan(),
}
