"""Shared tokenization for lexical retrieval."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Small English + boilerplate-finance stoplist; trims 10-K filler without a dep.
_STOP = set(
    """a an the and or of to in for on with is are be as by at from that this its it
    we our us their they i you he she his her them these those which who whom whose
    will would can could should may might must shall not no nor only own same so than
    too very s t don will just company companies inc corp ltd llc may also such other
    any all each more most some no than then there here when where why how what
    business operations including based product products service services million
    billion fiscal year years period periods financial""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOP]
