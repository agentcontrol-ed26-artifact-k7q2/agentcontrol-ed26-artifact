from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class Action(str, Enum):
    CHEAP_ANSWER = 'cheap_answer'
    STRONG_ANSWER = 'strong_answer'
    STRONG_HINT = 'strong_hint'
    STRONG_CRITIQUE = 'strong_critique'
    STRONG_CHECKLIST = 'strong_checklist'
    RETRIEVE = 'retrieve'
    READ_DOCUMENT = 'read_document'
    RUN_CODE = 'run_code'
    CALL_VERIFIER = 'call_verifier'
    ASK_CRITIC = 'ask_critic'
    DECOMPOSE = 'decompose'
    CHEAP_REPAIR = 'cheap_repair'
    CHEAP_REPAIR_AFTER_HINT = 'cheap_repair_after_hint'
    STOP = 'stop'
    ABSTAIN = 'abstain'


DEFAULT_ACTION_COSTS = {
    Action.CHEAP_ANSWER: 1.0, Action.STRONG_ANSWER: 10.0, Action.STRONG_HINT: 2.0,
    Action.STRONG_CRITIQUE: 2.5, Action.STRONG_CHECKLIST: 2.0, Action.RETRIEVE: 0.5,
    Action.READ_DOCUMENT: 0.5, Action.RUN_CODE: 0.2, Action.CALL_VERIFIER: 0.1,
    Action.ASK_CRITIC: 2.0, Action.DECOMPOSE: 1.0, Action.CHEAP_REPAIR: 1.0,
    Action.CHEAP_REPAIR_AFTER_HINT: 1.0, Action.STOP: 0.0, Action.ABSTAIN: 0.0,
}


@dataclass
class State:
    task_id: str
    family: str
    candidates: list[str] = field(default_factory=list)
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[str] = field(default_factory=list)
    uncertainty: float = 1.0
    remaining_budget: float = 10.0
    latency_slo_ms: int = 10000
    risk_level: float = 0.0
    context_tokens: int = 0
    done: bool = False


def action_cost(action: Action | str) -> float:
    return DEFAULT_ACTION_COSTS[Action(action)]


def feasible_actions(state: State) -> list[Action]:
    if state.done:
        return []
    actions = []
    for action, cost in DEFAULT_ACTION_COSTS.items():
        if cost <= state.remaining_budget:
            if action == Action.READ_DOCUMENT and not state.retrieved_evidence:
                continue
            actions.append(action)
    return actions


def apply_budget(state: State, action: Action | str, cost: float | None = None, latency_ms: int = 0) -> State:
    a = Action(action)
    c = action_cost(a) if cost is None else cost
    if c > state.remaining_budget:
        raise ValueError(f'Budget exceeded for action {a}: need {c}, have {state.remaining_budget}')
    return replace(state, remaining_budget=state.remaining_budget - c, latency_slo_ms=state.latency_slo_ms - latency_ms, done=state.done or a in {Action.STOP, Action.ABSTAIN})
