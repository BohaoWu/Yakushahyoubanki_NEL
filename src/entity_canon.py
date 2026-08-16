#!/usr/bin/env python3
"""Entity-catalog canonicalization for the yakusya NEL eval.

The Ritsumeikan-derived catalog contains many *duplicate person records*: the
same physical actor appears under more than one numeric_id. Two mechanisms:

  A. same source record  -- two numeric_ids share the same `idx`
     (e.g. ritsumei_PD:80652), i.e. the exact same source page copied twice.
  B. one person, many name-head entries -- the DB makes a separate "見出" entry
     for each stage name in an actor's career (九蔵 -> 団十郎 -> 海老蔵 ...),
     all carrying the *identical* 名乗期間 (name-succession) table and the same
     birth/death year. Different `idx`, same person.

Strict numeric_id matching unfairly scores a prediction wrong when it picks a
different record of the *same person*. build_canon_map() returns a mapping
numeric_id -> canonical numeric_id (the min id of each merge group) so the eval
can compare people, not record ids.

Only A is risk-free; B is gated on the *full* normalized 名乗期間 segment plus
birth/death year being identical, which only holds for genuine same-person
duplicates (a teacher/relative sharing one name does not reproduce the whole
career table). Use mode="source" for A only, mode="full" (default) for A+B.
"""

import json
import os
import re


def _nanori_sig(text):
    """Normalized full 名乗期間 (name-succession) segment, or None if it is not a
    reliable same-person key.

    Rejected as unreliable:
      - absent, or shorter than 30 chars
      - a *stub* table carrying no real era-date (only "未詳（）"): two different
        generations of one name (e.g. 松本名左衛門〈1〉 vs 〈2〉) both reduce to the
        bare name + 未詳, so an identical stub signature does NOT prove same
        person. Requiring at least one dated entry keeps genuine name-succession
        records (which always carry era-years) while dropping these stubs.
    """
    i = text.find("名乗期間")
    if i < 0:
        return None
    seg = re.sub(r"\s+", "", text[i:])
    if len(seg) <= 30:
        return None
    if not re.search(r"（\d{3,4}）", seg):  # no real date -> stub, not a key
        return None
    return seg


def _nanori_sig_struct(e):
    """Same-person key built from the *structured* kaimeihyou instead of parsing the
    free-text 名乗期間 block. `_nanori_sig(text)` keyed off full-width-paren era dates
    in `text`; that breaks the moment the text tail is reformatted (e.g. to JSON) and
    is redundant now that kaimeihyou is a first-class field. This reproduces the same
    equivalence classes on clean data — identical text tables parse to identical
    kaimeihyou — while being independent of how the tail is rendered.

    Falls back to the text parse when an entity carries no structured kaimeihyou.
    Returns a hashable signature or None (stub: no dated row, not a reliable key)."""
    km = (e.get("metadata") or {}).get("kaimeihyou")
    if not km:
        return _nanori_sig(e.get("text") or "")
    rows, dated = [], False
    for k in km:
        s, d = k.get("start_year"), k.get("end_year")
        if s or d:
            dated = True
        rows.append(
            (
                re.sub(r"\s", "", k.get("name") or ""),
                str(k.get("daime") or ""),
                s,
                d,
                k.get("name_type") or "",
            )
        )
    if not dated:  # stub table (only 未詳) — not a same-person key
        return None
    return tuple(rows)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the smaller id as the representative
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.p[hi] = lo


def build_canon_map(ents, mode="full"):
    """Map each numeric_id to its canonical (min) numeric_id.

    mode="none"   -> identity (no merging)
    mode="source" -> merge A only (same `idx`)
    mode="full"   -> merge A + B (same `idx`, or identical 名乗 table + birth/death)

    Returns (canon, stats) where canon[id] = canonical_id and stats is a dict
    with the merge counts for reporting.
    """
    ids = [int(e["numeric_id"]) for e in ents]
    canon = {i: i for i in ids}
    if mode == "none":
        return canon, {"mode": mode, "groups": 0, "merged": 0}

    # Dataset-provided clusters (e.g. Mahānāma name-variant clusters): when entities
    # carry metadata.cluster_id, that IS the canonical grouping — score by cluster
    # (a prediction of any cluster member counts). Bypasses the daime A/B rules.
    if any((e.get("metadata") or {}).get("cluster_id") is not None for e in ents):
        canon = {
            int(e["numeric_id"]): int((e.get("metadata") or {}).get("cluster_id", e["numeric_id"]))
            for e in ents
        }
        groups = len(set(canon.values()))
        return canon, {"mode": mode + "+cluster", "groups": groups, "merged": len(canon) - groups}

    uf = _UF()
    for i in ids:
        uf.find(i)

    # A: same source idx
    by_src = {}
    for e in ents:
        by_src.setdefault(e["idx"], []).append(int(e["numeric_id"]))
    for grp in by_src.values():
        for x in grp[1:]:
            uf.union(grp[0], x)

    # B: identical full 名乗 table + same birth/death year
    if mode == "full":
        by_sig = {}
        for e in ents:
            sig = _nanori_sig_struct(e)
            if not sig:
                continue
            m = e.get("metadata", {})
            key = (sig, m.get("birth_year"), m.get("death_year"))
            by_sig.setdefault(key, []).append(int(e["numeric_id"]))
        for grp in by_sig.values():
            for x in grp[1:]:
                uf.union(grp[0], x)

    canon = {i: uf.find(i) for i in ids}
    n_groups = len({v for v in canon.values()})
    merged = len(ids) - n_groups
    return canon, {"mode": mode, "groups": n_groups, "merged": merged, "total": len(ids)}


# --------------------------------------------------------------------------------------
# KB-level repair: filter each entities.jsonl's kaimeihyou by birth/death consistency.
#
# ARC source pages merged other generations' rename rows into some person records (see
# the note in dataset._ls_clean_kaimeihyou: 初代坂東三津五郎 1745-1782 ends up carrying
# 坂東八十助 1962-2000 / 三津五郎〈10〉 2001-2015). This pushes the same filter down to
# the data layer: drop any rename row whose whole span falls outside [birth-10, death+10].
# Deletion-only, deterministic, idempotent — re-running changes nothing. Entities with
# both birth and death unknown are left untouched.
# --------------------------------------------------------------------------------------

_NANORI_MARK = re.compile(r"名乗期間名前種別")


def _text_json_tail(e):
    """Replace an entity's trailing 「名乗期間」 free-text paragraph in `text` with the
    cleaned kaimeihyou serialized as JSON.

    In ritsumei entities the 名乗期間 paragraph is always at the end of `text` (marked by
    「名乗期間名前種別」). Keep the marker and the biography before it, then replace what
    follows with a JSON array — no format drift, same source as metadata.kaimeihyou, so the
    dirty rows (already removed at the kaimeihyou layer) disappear naturally. If kaimeihyou
    is empty, leave it untouched (avoids losing info when parsing failed). Returns the new
    text, or None to skip."""
    if not e.get("idx", "").startswith("ritsumei"):
        return None
    km = (e.get("metadata") or {}).get("kaimeihyou") or []
    if not km:
        return None
    t = e.get("text") or ""
    m = _NANORI_MARK.search(t)
    if not m:
        return None
    rows = []
    for k in km:
        r = {
            "name": (k.get("name") or "").strip(),
            "daime": k.get("daime"),
            "reading": (k.get("reading") or "").strip(),
            "start": k.get("start_year"),
            "end": k.get("end_year"),
        }
        nt = k.get("name_type") or ""
        if nt and nt != "→":
            r["type"] = nt
        rows.append(r)
    new = t[: m.end()] + json.dumps(rows, ensure_ascii=False)
    return new if new != t else None


def repair_file(path, do_text=False):
    from dataset import _ls_clean_kaimeihyou  # single rule source; lazy import avoids a cycle

    # symlink -> repair the real target, not the link itself
    real = os.path.realpath(path)
    rows = [json.loads(l) for l in open(real, encoding="utf-8")]
    ent = row = txt = 0
    for e in rows:
        km = (e.get("metadata") or {}).get("kaimeihyou")
        if km:
            clean = _ls_clean_kaimeihyou(e)
            if len(clean) != len(km):
                ent += 1
                row += len(km) - len(clean)
                e["metadata"]["kaimeihyou"] = clean
        if do_text:  # rebuild the trailing text as JSON from the (cleaned) kaimeihyou
            nt = _text_json_tail(e)
            if nt is not None:
                e["text"] = nt
                txt += 1
    if ent == 0 and txt == 0:
        print(f"  = {path}  (no change)")
        return 0, 0, 0
    if not os.path.exists(real + ".bak"):  # keep the very first backup
        import shutil

        shutil.copy2(real, real + ".bak")
    with open(real, "w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(
        f"  ✓ {path}  kaimeihyou entities {ent}/rows dropped {row}"
        + (f" | text→JSON {txt}" if do_text else "")
        + f"   (backup {os.path.basename(real)}.bak)"
    )
    return ent, row, txt


def repair_kaimeihyou(targets=None, do_text=False):
    """KB-level kaimeihyou repair: filter each entities.jsonl's rename table by birth/death
    consistency.

    `targets` is a list of entities.jsonl paths; when None, use the default minimal scope
    data/dataset_daime_eval/entities.jsonl (a symlink that covers daime_eval_full /
    daime_ft / daime_ft_w100). do_text=True also rewrites the trailing text into JSON."""
    if targets is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        targets = [os.path.join(root, "data", "dataset_daime_eval", "entities.jsonl")]
    print(
        "KB-level kaimeihyou repair"
        + (" (birth/death filter + trailing text→JSON)" if do_text else " (birth/death consistency filter, deletion-only)")
    )
    te = tr = tt = 0
    for p in targets:
        e, r, x = repair_file(p, do_text=do_text)
        te += e
        tr += r
        tt += x
    print(f"total: kaimeihyou entities {te} / dirty rows dropped {tr}" + (f" / text→JSON {tt}" if do_text else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entity-catalog canonicalization utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rk = sub.add_parser(
        "repair-kaimeihyou",
        help="KB-level repair: filter entities.jsonl kaimeihyou by birth/death consistency.",
    )
    rk.add_argument(
        "targets",
        nargs="*",
        help="entities.jsonl paths (default: data/dataset_daime_eval/entities.jsonl)",
    )
    rk.add_argument("--text", action="store_true", help="also rewrite the trailing text into JSON")
    args = parser.parse_args()

    if args.cmd == "repair-kaimeihyou":
        repair_kaimeihyou(args.targets or None, do_text=args.text)
