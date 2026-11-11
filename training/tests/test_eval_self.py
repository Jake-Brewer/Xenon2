"""The self-knowledge exam stays answerable from .xenon/self.md, and scores as described."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_self as es

SELF_MD = Path(__file__).resolve().parents[2] / ".xenon" / "self.md"


def test_every_fact_is_in_self_md():
    # A fact missing from self.md is either a wrong question or a gap in the doc.
    doc = SELF_MD.read_text(encoding="utf-8")
    missing = [(q, f) for q, facts in es.QUESTIONS for f in facts if not re.search(f, doc, re.IGNORECASE)]
    assert missing == []


def test_score_is_the_share_of_facts_found():
    assert es.score("RWKV-7, 2.9B", [r"rwkv", r"2\.9 ?b"]) == (1.0, [])
    got, missing = es.score("It is RWKV.", [r"rwkv", r"2\.9 ?b"])
    assert got == 0.5 and missing == [r"2\.9 ?b"]


def test_questions_are_unique():
    questions = [q for q, _ in es.QUESTIONS]
    assert len(questions) == len(set(questions))
