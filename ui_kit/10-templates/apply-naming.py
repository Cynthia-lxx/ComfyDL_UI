#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Apply the neutral-naming map to the UI reference kit. Python 3.9+, stdlib only.
#
#   python 10-templates/apply-naming.py --dry-run
#   python 10-templates/apply-naming.py --apply
#
# The map is data (00-overview/naming-map.json), so the rename is auditable and
# repeatable. Re-running --apply is a no-op: old tokens no longer match and
# already-renamed paths are skipped.
#
# Legal/citation metadata is deliberately preserved: SPDX headers, LICENCE,
# licence-audit.json, and upstream .cs paths inside comments / `source` fields.

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "00-overview/naming-map.json"
TEXT_EXTENSIONS = {".js", ".css", ".html", ".py", ".txt", ".md", ".json"}

# Upstream source references must survive even if a token happens to overlap.
# `.cs` must not be followed by another letter, otherwise `.css` is mistaken
# for an upstream C# reference and every stylesheet path gets corrupted.
UPSTREAM_RE = re.compile(r"[A-Za-z0-9_./\\\-]*\.cs(?![A-Za-z0-9])(?::[\d,\-\s]+)?|ppy/osu")


def protect(text: str):
    saved: list[str] = []

    def _stash(match):
        saved.append(match.group(0))
        return f"@@UPSRC{len(saved) - 1}@@"

    return UPSTREAM_RE.sub(_stash, text), saved


def restore(text: str, saved: list[str]) -> str:
    for index, original in enumerate(saved):
        text = text.replace(f"@@UPSRC{index}@@", original)
    return text


def apply_tokens(text: str, tokens: list, hits: Counter) -> str:
    for old, new in tokens:
        if not old or old == new:
            continue
        count = text.count(old)
        if count:
            hits[old] += count
            text = text.replace(old, new)
    return text


def apply_key(key: str, key_tokens: list, hits: Counter) -> str:
    for old, new in key_tokens:
        if key == old and old != new:
            hits[f"KEY:{old}"] += 1
            return new
    return key


def transform_json(node, tokens, key_tokens, skip_keys, hits):
    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            if key in skip_keys:
                result[key] = value  # citation / legal metadata: untouched
                continue
            result[apply_key(key, key_tokens, hits)] = transform_json(
                value, tokens, key_tokens, skip_keys, hits)
        return result
    if isinstance(node, list):
        return [transform_json(item, tokens, key_tokens, skip_keys, hits) for item in node]
    if isinstance(node, str):
        protected, saved = protect(node)
        return restore(apply_tokens(protected, tokens, hits), saved)
    return node


def iter_source_files(excluded_dirs, excluded_files):
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in excluded_dirs for part in path.relative_to(ROOT).parts):
            continue
        if rel in excluded_files or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        yield path, rel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Apply neutral naming to the UI reference kit.")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--dry-run", action="store_true", help="only report planned changes")
    args = parser.parse_args(argv)
    do_apply = args.apply and not args.dry_run

    config = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    policy = config["policy"]
    tokens = [tuple(pair) for pair in config["tokens"]]
    key_tokens = [tuple(pair) for pair in config.get("keyTokens", [])]
    excluded_files = set(policy["excludedFiles"]) | {"10-templates/apply-naming.py"}
    excluded_dirs = set(policy["excludedDirs"])
    structured = set(policy["structuredFiles"])
    skip_keys = set(policy["skipKeys"])
    hits: Counter = Counter()

    changed_files = 0
    for path, rel in iter_source_files(excluded_dirs, excluded_files):
        raw = path.read_bytes().decode("utf-8")
        if rel in structured:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[naming] WARNING: cannot parse {rel}: {exc}", file=sys.stderr)
                continue
            updated = transform_json(data, tokens, key_tokens, skip_keys, hits)
            text = json.dumps(updated, ensure_ascii=False, indent=2) + "\n"
        else:
            protected, saved = protect(raw)
            text = restore(apply_tokens(protected, tokens, hits), saved)
        if text != raw:
            changed_files += 1
            if do_apply:
                path.write_bytes(text.encode("utf-8"))

    # Directory renames are ordered child-first in the map.
    dir_moves = 0
    for item in config.get("dirRenames", []):
        src, dst = ROOT / item["from"], ROOT / item["to"]
        if src.is_dir() and not dst.exists():
            dir_moves += 1
            if do_apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    file_moves = 0
    for item in config.get("fileRenames", []):
        src, dst = ROOT / item["from"], ROOT / item["to"]
        if src.is_file() and not dst.exists():
            file_moves += 1
            if do_apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    cache_removed = 0
    for cache in ROOT.rglob("__pycache__"):
        if cache.is_dir() and "dist" not in cache.parts:
            cache_removed += 1
            if do_apply:
                shutil.rmtree(cache, ignore_errors=True)

    mode = "APPLIED" if do_apply else "DRY RUN"
    print(f"[naming] {mode} · files changed={changed_files} dirs={dir_moves} files renamed={file_moves} caches removed={cache_removed}")
    used = [(old, hits[old]) for old, _ in tokens if hits.get(old)]
    print(f"[naming] tokens matched: {len(used)} / {len(tokens)}")
    for old, count in used[:25]:
        print(f"  {count:>4}  {old}")
    unused = [old for old, _ in tokens if not hits.get(old) and old]
    print(f"[naming] tokens with no match: {len(unused)}")
    for old in unused[:20]:
        print(f"  -     {old}")
    key_hits = [k for k in hits if k.startswith("KEY:")]
    print(f"[naming] JSON keys renamed: {len(key_hits)}")
    for key in key_hits[:20]:
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
