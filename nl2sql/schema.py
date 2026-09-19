"""
Schema grounding: introspect a SQLite database into a typed model, render a
compact schema card for prompts, and resolve join paths over foreign keys so
the model is told how tables connect instead of guessing.
"""
from __future__ import annotations

import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    pk: bool = False
    notnull: bool = False


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str

    def as_join(self) -> str:
        return f"{self.table}.{self.column} = {self.ref_table}.{self.ref_column}"


@dataclass
class Table:
    name: str
    columns: List[Column] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    row_count: int = 0

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def has_column(self, name: str) -> bool:
        return name.lower() in {c.name.lower() for c in self.columns}


class Schema:
    """A database schema with case-insensitive lookups and a foreign-key join graph."""

    def __init__(self, tables: Sequence[Table]):
        self.tables: Dict[str, Table] = {t.name: t for t in tables}
        self._lower = {t.name.lower(): t.name for t in tables}

    # ------------------------------------------------------------------
    @classmethod
    def from_sqlite(cls, db_path: str) -> "Schema":
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            tables: List[Table] = []
            for name in names:
                cols = [Column(name=r[1], type=(r[2] or "").upper(), notnull=bool(r[3]), pk=bool(r[5]))
                        for r in conn.execute(f'PRAGMA table_info("{name}")')]
                fks = [ForeignKey(table=name, column=r[3], ref_table=r[2], ref_column=r[4])
                       for r in conn.execute(f'PRAGMA foreign_key_list("{name}")')]
                count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                tables.append(Table(name=name, columns=cols, foreign_keys=fks, row_count=count))
        finally:
            conn.close()
        return cls(tables)

    # ------------------------------------------------------------------
    def resolve_table(self, name: str) -> Optional[str]:
        """Canonical table name for a case-insensitive lookup, or None."""
        return self._lower.get(name.lower())

    def has_table(self, name: str) -> bool:
        return self.resolve_table(name) is not None

    def has_column(self, table: str, column: str) -> bool:
        canonical = self.resolve_table(table)
        return bool(canonical) and self.tables[canonical].has_column(column)

    def table_names(self) -> List[str]:
        return list(self.tables)

    # ------------------------------------------------------------------
    def card(self) -> str:
        """Compact, prompt-friendly description: one line per table with types, keys and FK targets."""
        lines = []
        for t in self.tables.values():
            fk_by_col = {fk.column: fk for fk in t.foreign_keys}
            parts = []
            for c in t.columns:
                label = f"{c.name} {c.type}".strip()
                if c.pk:
                    label += " PK"
                if c.name in fk_by_col:
                    fk = fk_by_col[c.name]
                    label += f" -> {fk.ref_table}.{fk.ref_column}"
                parts.append(label)
            lines.append(f"{t.name}({', '.join(parts)})  -- {t.row_count} rows")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def join_graph(self) -> Dict[str, List[Tuple[str, ForeignKey]]]:
        graph: Dict[str, List[Tuple[str, ForeignKey]]] = {name: [] for name in self.tables}
        for t in self.tables.values():
            for fk in t.foreign_keys:
                if fk.ref_table in graph:
                    graph[t.name].append((fk.ref_table, fk))
                    graph[fk.ref_table].append((t.name, fk))
        return graph

    def join_path(self, start: str, end: str) -> List[ForeignKey]:
        """Shortest chain of foreign keys connecting two tables (BFS), or [] if none / same table."""
        a, b = self.resolve_table(start), self.resolve_table(end)
        if not a or not b or a == b:
            return []
        graph = self.join_graph()
        prev: Dict[str, Tuple[str, ForeignKey]] = {}
        queue = deque([a])
        seen = {a}
        while queue:
            cur = queue.popleft()
            if cur == b:
                break
            for nxt, fk in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = (cur, fk)
                    queue.append(nxt)
        if b not in prev:
            return []
        path: List[ForeignKey] = []
        node = b
        while node != a:
            node, fk = prev[node]
            path.append(fk)
        path.reverse()
        return path

    def join_hints(self, tables: Iterable[str]) -> List[str]:
        """Join conditions covering every pair of the given tables, deduplicated, in order."""
        names = [self.resolve_table(t) for t in tables]
        names = [n for n in names if n]
        hints: List[str] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for fk in self.join_path(names[i], names[j]):
                    hint = fk.as_join()
                    if hint not in hints:
                        hints.append(hint)
        return hints

    # ------------------------------------------------------------------
    def tables_mentioned(self, question: str, synonyms: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
        """Tables whose name (or a configured synonym) appears in the question, singular or plural."""
        q = question.lower()
        words = set(re.findall(r"[a-z]+", q))
        found: List[str] = []
        for name in self.tables:
            candidates = {name.lower(), name.lower() + "s", name.lower() + "es"}
            candidates |= {s.lower() for s in (synonyms or {}).get(name, [])}
            # split CamelCase names so "InvoiceLine" also matches "invoice lines"
            spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()
            if any(c in words for c in candidates) or (spaced != name.lower() and spaced in q):
                found.append(name)
        return found
