from __future__ import annotations
import math, re
from typing import Any

STOPWORDS = {'the','a','an','and','or','of','in','on','to','is','was','were','by','for','with'}


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\[[^\]]+\]', ' ', text)
    text = re.sub(r'[^a-z0-9.\- ]+', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def math_exact_match(prediction: str, gold: str) -> bool:
    pnum, gnum = extract_last_number(prediction), extract_last_number(gold)
    if pnum is not None and gnum is not None:
        try:
            return math.isclose(float(pnum), float(gnum), rel_tol=1e-9, abs_tol=1e-9)
        except ValueError:
            pass
    return normalize_answer(prediction) == normalize_answer(gold)


def verify_math_answer(prediction: str, gold: str) -> dict[str, Any]:
    ok = math_exact_match(prediction, gold)
    return {'pass': ok, 'score': 1.0 if ok else 0.0, 'type': 'math_exact'}


FORBIDDEN_CODE_PATTERNS = ['import os','import subprocess','import socket','__import__','open(','eval(','exec(','compile(','globals(','locals(']
SAFE_BUILTINS = {'abs':abs,'all':all,'any':any,'bool':bool,'dict':dict,'enumerate':enumerate,'float':float,'int':int,'len':len,'list':list,'max':max,'min':min,'range':range,'reversed':reversed,'round':round,'set':set,'str':str,'sum':sum,'zip':zip}


def run_python_unit_tests(code: str, tests: list[str]) -> dict[str, Any]:
    lowered = code.lower()
    for pattern in FORBIDDEN_CODE_PATTERNS:
        if pattern in lowered:
            return {'pass': False, 'type': 'unit_tests', 'error': f'forbidden pattern: {pattern}'}
    namespace: dict[str, Any] = {}
    globals_dict = {'__builtins__': SAFE_BUILTINS}
    try:
        exec(code, globals_dict, namespace)
        for test in tests:
            exec(test, globals_dict, namespace)
    except Exception as e:
        return {'pass': False, 'type': 'unit_tests', 'error': repr(e)}
    return {'pass': True, 'type': 'unit_tests', 'error': ''}


def parse_citations(answer: str) -> list[str]:
    return re.findall(r'\[([A-Za-z0-9_\-]+)\]', answer)


def citation_support_checker(answer: str, citations: list[str] | None, corpus: dict[str, str], expected_answer: str | None = None) -> dict[str, Any]:
    cited = citations or parse_citations(answer)
    if not cited:
        return {'pass': False, 'type': 'citation_support', 'unsupported_risk': 1.0, 'missing_citations': True}
    evidence = ' '.join(corpus.get(c, '') for c in cited)
    if not evidence.strip():
        return {'pass': False, 'type': 'citation_support', 'unsupported_risk': 1.0, 'missing_citations': False}
    target = expected_answer or re.sub(r'\[[^\]]+\]', '', answer)
    norm_target = normalize_answer(target)
    norm_evidence = normalize_answer(evidence)
    target_tokens = [t for t in norm_target.split() if t not in STOPWORDS]
    supported = bool(norm_target and norm_target in norm_evidence)
    if not supported and target_tokens:
        supported = all(t in norm_evidence for t in target_tokens)
    return {'pass': supported, 'type': 'citation_support', 'unsupported_risk': 0.0 if supported else 1.0, 'missing_citations': False}
