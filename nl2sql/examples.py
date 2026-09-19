"""
Curated few-shot examples, retrieved by lexical overlap. Examples are the
cheapest, most controllable way to teach a model house conventions (date
handling, naming, which join to use); the bank is a JSON file a reviewer can
read and edit.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "what", "which", "who",
    "how", "many", "much", "list", "show", "give", "me", "all", "with", "by", "per", "each", "that",
    "there", "do", "does", "have", "has", "number", "total", "count", "find", "get", "from", "at",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


@dataclass(frozen=True)
class Example:
    question: str
    sql: str
    note: str = ""


class ExampleBank:
    def __init__(self, examples: Sequence[Example]):
        self.examples = list(examples)

    @classmethod
    def load(cls, path: Optional[str]) -> "ExampleBank":
        if not path:
            return cls([])
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Example(question=d["question"], sql=d["sql"], note=d.get("note", "")) for d in data])

    def __len__(self) -> int:
        return len(self.examples)

    def retrieve(self, question: str, k: int = 4) -> List[Example]:
        """Top-k examples by Jaccard overlap of content words; ties keep file order."""
        q = _tokens(question)
        if not q or not self.examples:
            return []
        scored = []
        for i, ex in enumerate(self.examples):
            e = _tokens(ex.question)
            score = len(q & e) / len(q | e) if (q | e) else 0.0
            if score > 0:
                scored.append((score, -i, ex))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [ex for _, _, ex in scored[:k]]
