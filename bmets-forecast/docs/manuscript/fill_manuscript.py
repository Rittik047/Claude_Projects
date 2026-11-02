"""Fill {{dotted.key|fmt}} placeholders in manuscript.md from reports/results.json.

Usage: python docs/manuscript/fill_manuscript.py reports/results.json docs/manuscript/manuscript.md > manuscript_filled.md
Missing keys render as [pending: key] so a partially-run pipeline still produces a readable draft.
"""
from __future__ import annotations

import json
import re
import sys


def _get(d, key):
    for part in key.split("."):
        if isinstance(d, list):
            d = d[int(part)]
        else:
            d = d[part]
    return d


def fill(text: str, results: dict) -> str:
    def rep(m):
        key, fmt = m.group(1), m.group(2)
        try:
            v = _get(results, key)
        except (KeyError, IndexError, ValueError, TypeError):
            return f"[pending: {key}]"
        return format(v, fmt) if fmt and v is not None else str(v)
    return re.sub(r"\{\{([A-Za-z0-9_.]+)(?:\|([^}]+))?\}\}", rep, text)


if __name__ == "__main__":
    res = json.loads(open(sys.argv[1]).read())
    sys.stdout.write(fill(open(sys.argv[2]).read(), res))
