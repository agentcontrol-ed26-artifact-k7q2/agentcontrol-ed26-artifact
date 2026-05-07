from __future__ import annotations
from typing import Any

EVIDENCE_CORPUS: dict[str, str] = {
    'doc_curie': 'Marie Curie discovered polonium and radium and won Nobel Prizes in physics and chemistry.',
    'doc_lovelace': 'Ada Lovelace wrote notes on the Analytical Engine and is often described as an early computer programmer.',
    'doc_moon': 'The Apollo 11 mission landed humans on the Moon in 1969.',
    'doc_python': 'Python was created by Guido van Rossum and first released in 1991.',
    'doc_everest': 'Mount Everest is the highest mountain above sea level.',
}

EVIDENCE_QA: list[dict[str, Any]] = [
    {'id':'e001','question':'Who discovered radium?','answer':'Marie Curie','citations':['doc_curie'],'cheap':'Pierre Curie [doc_curie]','strong':'Marie Curie [doc_curie]'},
    {'id':'e002','question':'Which mission landed humans on the Moon in 1969?','answer':'Apollo 11','citations':['doc_moon'],'cheap':'Apollo 13 [doc_moon]','strong':'Apollo 11 [doc_moon]'},
    {'id':'e003','question':'Who created Python?','answer':'Guido van Rossum','citations':['doc_python'],'cheap':'Dennis Ritchie [doc_python]','strong':'Guido van Rossum [doc_python]'},
    {'id':'e004','question':'What is the highest mountain above sea level?','answer':'Mount Everest','citations':['doc_everest'],'cheap':'K2 [doc_everest]','strong':'Mount Everest [doc_everest]'},
]


def load_evidence_tasks(n: int | None = None) -> list[dict[str, Any]]:
    return EVIDENCE_QA[:n] if n else list(EVIDENCE_QA)


def retrieve(query: str, k: int = 2) -> list[tuple[str, str]]:
    q = query.lower()
    scored = []
    for doc_id, text in EVIDENCE_CORPUS.items():
        score = sum(1 for token in q.split() if token.strip('?.!,').lower() in text.lower())
        scored.append((score, doc_id, text))
    scored.sort(reverse=True)
    return [(doc_id, text) for score, doc_id, text in scored[:k] if score > 0] or list(EVIDENCE_CORPUS.items())[:k]


def format_dummy_evidence_prompt(task: dict[str, Any], mode: str = 'answer') -> str:
    docs = '\n'.join(f"{doc_id}: {EVIDENCE_CORPUS[doc_id]}" for doc_id in task['citations'])
    return '\n'.join([
        f"EVIDENCE_QA:{task['id']}",
        f"Question: {task['question']}",
        f"Evidence:\n{docs}",
        f"DUMMY_CHEAP: {task['cheap']}",
        f"DUMMY_STRONG: {task['strong']}",
        f"DUMMY_ANSWER: {task['answer']} [{task['citations'][0]}]",
        f"Mode: {mode}",
    ])
