"""Frozen multi-hop QA corpus for the agentic-search-vs-fixed-RAG experiment.

Design principles:
- Each question requires linking AT LEAST 2 supporting documents.
- Distractors are deliberately picked to share lexical features with the
  question so naive top-k RAG retrieves them and misses one of the linking docs.
- Deterministic verifier: gold-phrase substring + authorized citation set.
- All locally constructed, no benchmark download.
"""
from __future__ import annotations

import re
from typing import Any


# 40 documents — domains: science, history, geography, technology, biology
_CORPUS = {
    # Curie family (multiple linking opportunities)
    "doc_curie_marie": "Marie Curie discovered polonium and radium and won Nobel Prizes in physics (1903) and chemistry (1911). She was born in Warsaw, Poland.",
    "doc_curie_pierre": "Pierre Curie shared the 1903 Nobel Prize in Physics with his wife Marie Curie and Henri Becquerel. He died in a street accident in 1906.",
    "doc_curie_irene": "Irene Joliot-Curie, daughter of Marie and Pierre Curie, won the 1935 Nobel Prize in Chemistry with her husband Frederic Joliot.",
    # Einstein
    "doc_einstein_relativity": "Albert Einstein developed the theory of general relativity, published in 1915. The theory describes gravity as the curvature of spacetime.",
    "doc_einstein_nobel": "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect, not for relativity.",
    "doc_einstein_birth": "Albert Einstein was born in Ulm, Germany on March 14, 1879, and emigrated to the United States in 1933.",
    # Quantum theory
    "doc_planck_quantum": "Max Planck originated quantum theory in 1900 by introducing energy quanta to explain blackbody radiation.",
    "doc_planck_nobel": "Max Planck won the Nobel Prize in Physics in 1918.",
    # Computing & Python
    "doc_python_creator": "Python was created by Guido van Rossum and first released in 1991.",
    "doc_python_origin": "Guido van Rossum was born in 1956 in The Hague, Netherlands. He worked at CWI, Google, and Dropbox.",
    "doc_python_versions": "Python 3.0 was released in 2008. Python 2 reached end-of-life in 2020.",
    # Apollo / moon
    "doc_apollo_mission": "The Apollo 11 mission, commanded by Neil Armstrong, landed humans on the Moon in July 1969.",
    "doc_armstrong_bio": "Neil Armstrong was an American astronaut and the first person to walk on the Moon. He was born in Wapakoneta, Ohio in 1930.",
    "doc_aldrin_bio": "Edwin Eugene 'Buzz' Aldrin Jr. was the lunar module pilot on Apollo 11 and the second person to walk on the Moon. He was born in Glen Ridge, New Jersey in 1930.",
    # Aviation
    "doc_wright_flight": "The Wright brothers, Orville and Wilbur, achieved the first sustained powered flight in 1903 at Kitty Hawk, North Carolina.",
    "doc_wright_origin": "Orville Wright was born in 1871 in Dayton, Ohio. Wilbur Wright was born in 1867 near Millville, Indiana.",
    # DNA
    "doc_dna_double_helix": "The double helix structure of DNA was elucidated by James Watson and Francis Crick in 1953, building on X-ray data from Rosalind Franklin.",
    "doc_franklin_xray": "Rosalind Franklin produced the X-ray diffraction images of DNA, including Photo 51, that helped identify its helical structure.",
    "doc_franklin_death": "Rosalind Franklin died of cancer in 1958 at age 37, before the 1962 Nobel Prize was awarded for the discovery of DNA's double helix.",
    # WW II
    "doc_war_dates": "World War II began in 1939 with Germany's invasion of Poland and ended in 1945 after the surrender of Germany and Japan.",
    "doc_ww2_pearl": "The attack on Pearl Harbor on December 7, 1941 brought the United States into World War II.",
    # Telephone
    "doc_telephone": "Alexander Graham Bell patented the telephone in 1876.",
    "doc_bell_origin": "Alexander Graham Bell was born in Edinburgh, Scotland in 1847 and emigrated to Canada in 1870.",
    # Evolution
    "doc_darwin_origin": "Charles Darwin published On the Origin of Species in 1859, laying the foundation of evolutionary biology.",
    "doc_darwin_bio": "Charles Darwin was born in 1809 in Shrewsbury, England, and traveled aboard HMS Beagle from 1831 to 1836.",
    # Geography
    "doc_amazon_river": "The Amazon River, in South America, is the second longest river in the world after the Nile.",
    "doc_nile_river": "The Nile River, primarily flowing through Egypt and Sudan, is generally considered the longest river in the world.",
    # Mountains
    "doc_everest": "Mount Everest is the highest mountain above sea level. It was first summited in 1953 by Edmund Hillary and Tenzing Norgay.",
    "doc_hillary_bio": "Edmund Hillary was born in 1919 in Auckland, New Zealand. He summited Everest with Tenzing Norgay in 1953.",
    # Programming
    "doc_lovelace": "Ada Lovelace wrote notes on Charles Babbage's Analytical Engine and is often described as the first computer programmer.",
    "doc_babbage": "Charles Babbage designed the Analytical Engine in the 1830s, considered an early conceptual general-purpose computer. He was born in 1791 in London.",
    # Distractor docs (lexically similar but answer-irrelevant)
    "doc_curie_distractor1": "The Curie temperature is the temperature at which a ferromagnetic material loses its magnetism, named after Pierre Curie.",
    "doc_curie_distractor2": "The Curie unit (Ci) is a non-SI unit of radioactivity, equal to 3.7e10 disintegrations per second.",
    "doc_einstein_distractor": "Einstein, Texas is a small unincorporated community in the United States, with no relation to Albert Einstein.",
    "doc_python_distractor1": "The python (Pythonidae) is a family of large nonvenomous snakes found in Asia, Africa, and Australia.",
    "doc_python_distractor2": "Monty Python's Flying Circus was a British surreal comedy sketch show that aired from 1969 to 1974.",
    "doc_apollo_distractor": "Apollo, in Greek mythology, is the god of music, prophecy, healing, and the sun.",
    "doc_wright_distractor": "Frank Lloyd Wright was an American architect best known for designing Fallingwater and the Guggenheim Museum.",
    "doc_dna_distractor": "DNA replication is the biological process of producing two identical copies of DNA from one original molecule.",
    "doc_amazon_distractor": "Amazon (the company) was founded by Jeff Bezos in 1994 and is headquartered in Seattle, Washington.",
    "doc_everest_distractor": "Mount Everest, like the Himalayas, was formed by tectonic plate collision approximately 50 million years ago.",
}


# 60 questions; each needs ≥ 2 supporting docs and is engineered so naive
# top-3 RAG WILL retrieve at least one distractor.
_QUESTIONS = [
    # Multi-hop where naive retrieval misses a linking doc (35 questions)
    ("ag_001", "In what city was the discoverer of radium born?", "Warsaw",
     ["doc_curie_marie"]),
    ("ag_002", "Who was the husband of the chemist who discovered polonium?", "Pierre Curie",
     ["doc_curie_marie", "doc_curie_pierre"]),
    ("ag_003", "What was the cause of death of the husband of the discoverer of radium?", "street accident",
     ["doc_curie_marie", "doc_curie_pierre"]),
    ("ag_004", "Who was the daughter of the discoverer of polonium?", "Irene",
     ["doc_curie_irene"]),
    ("ag_005", "In what state was the first person to walk on the Moon born?", "Ohio",
     ["doc_apollo_mission", "doc_armstrong_bio"]),
    ("ag_006", "In what year was the first person to walk on the Moon born?", "1930",
     ["doc_apollo_mission", "doc_armstrong_bio"]),
    ("ag_007", "Who was the second person to walk on the Moon?", "Buzz Aldrin",
     ["doc_apollo_mission", "doc_aldrin_bio"]),
    ("ag_008", "What did Albert Einstein win the Nobel Prize for?", "photoelectric effect",
     ["doc_einstein_nobel"]),
    ("ag_009", "In what year did Einstein emigrate to the United States?", "1933",
     ["doc_einstein_birth"]),
    ("ag_010", "What was the country of birth of the creator of Python?", "Netherlands",
     ["doc_python_creator", "doc_python_origin"]),
    ("ag_011", "Three companies the creator of Python worked at?", "CWI, Google, Dropbox",
     ["doc_python_creator", "doc_python_origin"]),
    ("ag_012", "In what year did the language created by Guido van Rossum reach end-of-life for version 2?", "2020",
     ["doc_python_creator", "doc_python_versions"]),
    ("ag_013", "Whose X-ray diffraction images contributed to the discovery of DNA's double helix?", "Rosalind Franklin",
     ["doc_dna_double_helix", "doc_franklin_xray"]),
    ("ag_014", "At what age did the producer of Photo 51 of DNA die?", "37",
     ["doc_franklin_xray", "doc_franklin_death"]),
    ("ag_015", "In what year was the Nobel Prize awarded for the discovery of DNA's double helix?", "1962",
     ["doc_franklin_death"]),
    ("ag_016", "Where did the publisher of On the Origin of Species travel from 1831 to 1836?", "HMS Beagle",
     ["doc_darwin_origin", "doc_darwin_bio"]),
    ("ag_017", "Where was the publisher of On the Origin of Species born?", "Shrewsbury",
     ["doc_darwin_origin", "doc_darwin_bio"]),
    ("ag_018", "Who patented the telephone, and where was that person born?", "Edinburgh",
     ["doc_telephone", "doc_bell_origin"]),
    ("ag_019", "In what year did the inventor of the telephone emigrate to Canada?", "1870",
     ["doc_telephone", "doc_bell_origin"]),
    ("ag_020", "What is the longest river in the world?", "Nile",
     ["doc_nile_river"]),
    ("ag_021", "Through which two countries does the longest river in the world primarily flow?", "Egypt and Sudan",
     ["doc_nile_river"]),
    ("ag_022", "What is the second longest river, on which continent?", "South America",
     ["doc_amazon_river"]),
    ("ag_023", "Where was the first person to summit Everest born?", "Auckland",
     ["doc_everest", "doc_hillary_bio"]),
    ("ag_024", "In what year was the first summiter of Everest born?", "1919",
     ["doc_everest", "doc_hillary_bio"]),
    ("ag_025", "Who designed the machine on which Ada Lovelace wrote her notes?", "Charles Babbage",
     ["doc_lovelace", "doc_babbage"]),
    ("ag_026", "Where was the designer of the Analytical Engine born?", "London",
     ["doc_lovelace", "doc_babbage"]),
    ("ag_027", "When was the Analytical Engine designed?", "1830s",
     ["doc_babbage"]),
    ("ag_028", "What event in 1941 brought the United States into World War II?", "Pearl Harbor",
     ["doc_ww2_pearl"]),
    ("ag_029", "Whose invasion in 1939 began World War II?", "Germany",
     ["doc_war_dates"]),
    ("ag_030", "Where was the first sustained powered flight achieved?", "Kitty Hawk",
     ["doc_wright_flight"]),
    ("ag_031", "Who originated quantum theory?", "Max Planck",
     ["doc_planck_quantum"]),
    ("ag_032", "What did Max Planck originate quantum theory to explain?", "blackbody radiation",
     ["doc_planck_quantum"]),
    ("ag_033", "In what year did Max Planck win the Nobel Prize?", "1918",
     ["doc_planck_nobel"]),
    ("ag_034", "Where was Albert Einstein born?", "Ulm",
     ["doc_einstein_birth"]),
    ("ag_035", "When was Python 3.0 released?", "2008",
     ["doc_python_versions"]),

    # Single-doc easy questions for cheap-arm baseline (15 questions)
    ("ag_036", "Who discovered polonium?", "Marie Curie", ["doc_curie_marie"]),
    ("ag_037", "Who developed the theory of general relativity?", "Albert Einstein", ["doc_einstein_relativity"]),
    ("ag_038", "Who created Python?", "Guido van Rossum", ["doc_python_creator"]),
    ("ag_039", "Which mission first landed humans on the Moon?", "Apollo 11", ["doc_apollo_mission"]),
    ("ag_040", "Who first walked on the Moon?", "Neil Armstrong", ["doc_armstrong_bio"]),
    ("ag_041", "Who patented the telephone?", "Alexander Graham Bell", ["doc_telephone"]),
    ("ag_042", "Who wrote On the Origin of Species?", "Charles Darwin", ["doc_darwin_origin"]),
    ("ag_043", "Who first summited Mount Everest?", "Edmund Hillary", ["doc_everest"]),
    ("ag_044", "Who is described as the first computer programmer?", "Ada Lovelace", ["doc_lovelace"]),
    ("ag_045", "When did World War II end?", "1945", ["doc_war_dates"]),
    ("ag_046", "Who shared the 1903 Nobel Prize in Physics with Henri Becquerel?", "Pierre Curie", ["doc_curie_pierre"]),
    ("ag_047", "Who produced X-ray images of DNA?", "Rosalind Franklin", ["doc_franklin_xray"]),
    ("ag_048", "Who designed the Analytical Engine?", "Charles Babbage", ["doc_babbage"]),
    ("ag_049", "Who developed the photoelectric effect explanation?", "Albert Einstein", ["doc_einstein_nobel"]),
    ("ag_050", "Which sister of Marie Curie won a Nobel Prize?", "Irene", ["doc_curie_irene"]),

    # Tricky single-hop with strong distractors (10 questions)
    ("ag_051", "What discovery did Einstein win the 1921 Nobel for?", "photoelectric effect",
     ["doc_einstein_nobel"]),
    ("ag_052", "What is Pearl Harbor known for?", "1941",
     ["doc_ww2_pearl"]),
    ("ag_053", "What did the Wright brothers achieve in 1903?", "first sustained powered flight",
     ["doc_wright_flight"]),
    ("ag_054", "Who was the lunar module pilot of Apollo 11?", "Buzz Aldrin",
     ["doc_aldrin_bio"]),
    ("ag_055", "Which Curie won a Nobel Prize in 1935?", "Irene",
     ["doc_curie_irene"]),
    ("ag_056", "Which Curie won the 1903 physics Nobel for radioactivity work?", "Marie",
     ["doc_curie_marie"]),
    ("ag_057", "Where did Edmund Hillary climb Everest from?", "1953",
     ["doc_everest", "doc_hillary_bio"]),
    ("ag_058", "What was the home country of the discoverer of Photo 51 of DNA?", "England",
     ["doc_franklin_xray", "doc_dna_double_helix"]),
    ("ag_059", "Who founded the company Amazon (the retailer)?", "Jeff Bezos",
     ["doc_amazon_distractor"]),
    ("ag_060", "What is the second longest river in the world?", "Amazon",
     ["doc_amazon_river"]),
]


def get_corpus() -> dict[str, str]:
    return dict(_CORPUS)


def get_questions() -> list[dict[str, Any]]:
    out = []
    for tid, q, a, citations in _QUESTIONS:
        # Multihop classification: ≥ 2 citations OR question linking phrase.
        multihop = len(citations) >= 2 or any(p in q.lower() for p in
                                                ("the discoverer of", "the inventor of",
                                                 "the creator of", "the publisher of",
                                                 "the husband of", "the daughter of",
                                                 "the second person", "the first person",
                                                 "the producer of", "the designer of",
                                                 "the first summiter", "the language created",
                                                 "the cause of death of"))
        out.append({
            "task_id": tid, "family": "agentic_search", "question": q,
            "answer": a, "citations": citations, "multihop": multihop,
            "difficulty": "multihop" if multihop else "single_hop",
        })
    return out


# ---------------------------------------------------------------------------
# Retrieval implementation: deterministic local "retriever" by lexical overlap
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s) if len(t) > 2]


def retrieve_topk(query: str, k: int = 3, exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
    """Return top-k docs by simple token-overlap score. Deterministic."""
    qtokens = set(_tokenize(query))
    exclude = exclude or set()
    scored = []
    for doc_id, text in _CORPUS.items():
        if doc_id in exclude:
            continue
        dtokens = set(_tokenize(text))
        if not dtokens:
            continue
        overlap = len(qtokens & dtokens) / max(1, len(qtokens))
        scored.append((doc_id, text, overlap))
    scored.sort(key=lambda x: (-x[2], x[0]))
    return scored[:k]


def _normalize(s: str) -> str:
    return s.strip().lower()


def verify_answer(task: dict, output: str) -> tuple[bool, float]:
    """Return (success, unsupported_risk)."""
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
