from agentcontrol.tasks_evidence import EVIDENCE_CORPUS
from agentcontrol.verifiers import citation_support_checker, parse_citations


def test_parse_citations():
    assert parse_citations('Marie Curie [doc_curie]') == ['doc_curie']


def test_citation_support_checker_sanity():
    ok = citation_support_checker('Marie Curie [doc_curie]', None, EVIDENCE_CORPUS, expected_answer='Marie Curie')
    bad = citation_support_checker('Einstein [doc_curie]', None, EVIDENCE_CORPUS, expected_answer='Einstein')
    assert ok['pass'] is True
    assert bad['pass'] is False
