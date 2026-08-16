#!/usr/bin/env python3
"""Clean test of the proposed method: ORGANIZE the dataset's extra evidence and
PROVIDE it to the LLM.

Everything is held fixed — same candidates (same gather, same normalizer), same
single LLM selection call, same scoring — and only the candidate fact sheet varies:
  without : title + description
  with    : title + description + the organized evidence for that dataset
            (daime: birth/death + kaimeihyou name-at-year + dated name periods;
             mahanama: also_known_as name variants; HIPE: mined dates + aliases)
so the delta isolates evidence provision, not candidate recall or pipeline shape.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/workspace/Yakushahyoubanki_NEL"
sys.path.insert(0, f"{ROOT}/src")
import config  # noqa: E402,F401  (side effect: adds src/methods subpackages to sys.path)

for line in open(f"{ROOT}/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["LLM_CACHE"] = "1"

from llm_cache import wrap_client
from openai import OpenAI

CLIENT = wrap_client(OpenAI())
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# dataset -> scoring kind
DATASETS = [("daime_ft_w100", "canon"), ("mahanama", "cluster")] + [
    (d, "plain")
    for d in [
        "ajmc_en",
        "ajmc_de",
        "ajmc_fr",
        "hipe2020_en",
        "hipe2020_de",
        "hipe2020_fr",
        "newseye_de",
        "newseye_fi",
        "newseye_fr",
        "newseye_sv",
        "sonar_de",
        "topres19th_en",
    ]
]
if os.environ.get("ONLY_DS"):
    DATASETS = [x for x in DATASETS if x[0] in os.environ["ONLY_DS"].split(",")]

# fields that are "organized evidence" (everything beyond title/description).
# `type` is the Wikidata P31 label; it pairs with the mention's annotated Type
# (pers/work/loc/org), which the prompt states — that pairing is the generic
# corpora's query-conditioned discriminator, so both halves must be present.
# Order matters: fmt_evidence caps the rendered string at 260 chars, so the keyed
# fields (type, dates) must precede the open-ended alias list or a long
# `also_known_as` truncates the very evidence a query key pairs with.
EV_KEYS = [
    "type",
    "birth_year",
    "death_year",
    "name_at_mention_year",
    "names_in_context",
    "name_periods",
    "dates",
    "also_known_as",
]

# mention-side annotated type -> what it means, stated in the prompt
TYPEMAP = {
    "pers": "a PERSON",
    "per": "a PERSON",
    "work": "a WORK (text/play/poem)",
    "humanprod": "a WORK (human-made product)",
    "loc": "a PLACE",
    "org": "an ORGANISATION",
    "prod": "a PRODUCT",
    "building": "a BUILDING",
    "street": "a STREET",
}

# HIPE's NE-FINE-LIT tagset (metadata.fine_type, added by HipeBuilder.add_fine_types).
# The coarse tag cannot separate candidates that are all places; `loc.adm.town` vs
# `loc.adm.reg` can — a town and the canton sharing its name are different rows in
# Wikidata with different P31s. Prefer the fine tag wherever the corpus annotates
# one, fall back to coarse otherwise: a single rule, applied to every dataset.
FINEMAP = {
    "pers.ind": "an INDIVIDUAL PERSON",
    "pers.author": "an AUTHOR (a person)",
    "pers.editor": "an EDITOR (a person)",
    "pers.myth": "a MYTHOLOGICAL FIGURE",
    "pers.other": "a PERSON",
    "loc": "a PLACE",
    "loc.adm.town": "a TOWN or CITY",
    "loc.adm.reg": "a REGION / province / canton (NOT a town)",
    "loc.adm.nat": "a COUNTRY / nation (NOT a town)",
    "loc.adm.sup": "a SUPRANATIONAL region (e.g. a continent)",
    "loc.phys.geo": "a PHYSICAL GEOGRAPHIC feature",
    "loc.phys.hydro": "a BODY OF WATER (river, lake, sea)",
    "loc.phys.astro": "an ASTRONOMICAL object",
    "loc.oro": "a MOUNTAIN or orographic feature",
    "loc.fac": "a FACILITY or built structure",
    "org.adm": "an ADMINISTRATIVE body / authority",
    "org.ent": "an ORGANISATION / company",
    "org.ent.pressagency": "a PRESS AGENCY",
    "prod.media": "a MEDIA PRODUCT (newspaper, journal)",
    "prod.doctr": "a DOCTRINE or named product",
    "work.primlit": "a PRIMARY LITERARY WORK (a text, not its author)",
}


def mention_type_phrase(r):
    """The mention's annotated type, at the finest granularity the corpus provides."""
    fine = ((r.get("metadata") or {}).get("fine_type") or "").strip()
    if fine:
        return FINEMAP.get(fine, fine)
    mt = (r.get("type") or "").lower()
    return TYPEMAP.get(mt, mt) if mt else ""


# Corpora whose document date anchors the referent: someone named in an 1900
# newspaper was alive around 1900, so the document year is the key that pairs with
# a candidate's `dates`. NOT true of ajmc — a classical commentary printed in 1881
# discusses Homer (800 BC), so there its document date is a trap, not a key.
# metadata carries `date` ("1900-06-26"), never `year`, so it must be parsed out.
NEWSPAPER = {
    "hipe2020_en",
    "hipe2020_de",
    "hipe2020_fr",
    "newseye_de",
    "newseye_fi",
    "newseye_fr",
    "newseye_sv",
    "sonar_de",
    "topres19th_en",
}


def doc_year(r):
    """Document year from metadata.date, or None (newseye_fr ships 'NA' on ~half)."""
    m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str((r.get("metadata") or {}).get("date") or ""))
    return int(m.group(1)) if m else None


# The 'all' arm: hand the LLM every candidate-side annotation the dataset ships,
# unorganized. It is the ablation that asks whether shaping the evidence earns its
# keep, or whether dumping the raw metadata does just as well.
#
# Candidate-side ONLY. The mention's own metadata.qid (mahanama: cluster_id) IS the
# gold label — feeding it back would score ~100% and measure nothing. An entity's
# own qid is not a leak (it says nothing about which candidate is correct), but it
# is dropped anyway along with the housekeeping fields, which carry no signal.
ALL_SKIP = {
    "nil",
    "dataset",
    "language",
    "qid",
    "idx",
    "cluster_id",
    "cluster_head_eid",
    "detail_id",
    "source",
    "wikipedia_url",
}
# 4000 is where every dataset truncates 0% of candidates — the dump must not lose a
# fair comparison merely by being clipped. daime's full kaimeihyou table is what
# costs: ~2.9k tokens per 20-candidate prompt, vs ~0.3k for HIPE.
ALL_CAP = 4000


def fmt_all(e):
    """Every candidate-side metadata field, sorted, capped. No selection, no shaping."""
    bits = []
    for k, v in sorted((e.get("metadata") or {}).items()):
        if k in ALL_SKIP or v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v[:40])
        bits.append(f"{k}={v}")
    return "; ".join(bits)[:ALL_CAP]


def fmt_evidence(ev):
    """Render only the organized-evidence fields, compactly."""
    bits = []
    for k in EV_KEYS:
        v = ev.get(k)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            v = ", ".join(
                str(x)
                if not isinstance(x, dict)
                else f"{x.get('name', '')}({x.get('start_year', '')}-{x.get('end_year', '')})"
                for x in v[:6]
            )
        bits.append(f"{k}={v}")
    return "; ".join(bits)[:260]


def run(ds, kind, n, cap=20):
    import importlib

    import el_mode
    import llm_debate_disambiguator
    import llm_tool_disambiguator

    for m in (el_mode, llm_tool_disambiguator, llm_debate_disambiguator):
        importlib.reload(m)
    from llm_debate_disambiguator import build_evidence
    from llm_tool_disambiguator import EntityDB, tool_search_entities

    D = f"{ROOT}/data/dataset_{ds}"
    # prefer the richest entity file: real Wikidata dates > fetched aliases > base
    ef = next(
        f"{D}/{f}"
        for f in ("entities_evidence.jsonl", "entities_aliased.jsonl", "entities.jsonl")
        if os.path.exists(f"{D}/{f}")
    )
    ents = [json.loads(l) for l in open(ef)]
    byid = {int(e["numeric_id"]): e for e in ents}
    test = [json.loads(l) for l in open(f"{D}/test.jsonl")][:n]
    db = EntityDB(ents)

    if kind == "canon":
        from entity_canon import build_canon_map

        canon, _ = build_canon_map(ents, mode="full")
        ok = lambda p, g: canon.get(p, p) == canon.get(g, g)
    elif kind == "cluster":
        cl = {int(e["numeric_id"]): e["metadata"]["cluster_id"] for e in ents}
        ok = lambda p, g: cl.get(p, -1) == cl.get(g, -2)
    else:
        ok = lambda p, g: p == g

    def ask(r, arm):
        """arm: 'off' title+desc | 'org' + organized evidence | 'all' + raw dump."""
        yr = (r.get("metadata") or {}).get("year")
        yr = yr if isinstance(yr, int) else None
        if yr is None and ds in NEWSPAPER:
            yr = doc_year(r)
        cids = [c["entity_id"] for c in tool_search_entities(db, r["mention"])][:cap]
        if not cids:
            return None
        r["_nc"] = len(cids)
        ctx = (
            (r.get("context_left") or "")[-250:]
            + " [["
            + r["mention"]
            + "]] "
            + (r.get("context_right") or "")[:250]
        ) or (r.get("metadata") or {}).get("sentence", "")
        with_ev = arm != "off"
        lines = []
        for i, c in enumerate(cids):
            e = byid[c]
            s = f"{i + 1}. {e['title']} -- {(e.get('text') or e.get('entity') or '')[:170]}"
            if arm == "org":
                extra = fmt_evidence(build_evidence(db, c, yr, context=ctx))
            elif arm == "all":
                extra = fmt_all(e)
            else:
                extra = ""
            if extra:
                s += f"  [{extra}]"
            lines.append(s)
        head = f'Mention: "{r["mention"]}"'
        if yr:
            head += f"  (document year: {yr})"
        if with_ev:
            # the query keys: state what the candidate evidence is to be matched
            # against. Without these the evidence is present but unusable.
            # Only pose the type key when the KB's type actually varies — asking
            # mahanama (every entity a `person`) to match on it is pure noise.
            mt = mention_type_phrase(r)
            if mt and db.type_informative:
                head += (
                    f"\nThe mention is annotated as {mt}. Prefer a candidate whose type matches."
                )
            if yr and ds in NEWSPAPER:
                head += (
                    f"\nIt appears in a newspaper published in {yr}. Prefer a "
                    f"candidate whose `dates` are consistent with being "
                    f"referred to then."
                )
        p = (
            f"{head}\nContext: {ctx[:450]}\n\nCandidates:\n"
            + "\n".join(lines)
            + '\n\nReply ONLY JSON: {"choice": INTEGER}'
        )
        t = (
            CLIENT.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=30, messages=[{"role": "user", "content": p}]
            )
            .choices[0]
            .message.content
        )
        m = re.search(r'"choice"\s*:\s*(\d+)', t or "") or re.search(r"\b(\d+)\b", t or "")
        if not m:
            return None
        i = int(m.group(1)) - 1
        return cids[i] if 0 <= i < len(cids) else None

    def acc(arm):
        """-> {id(row): correct?}, so the ambiguous subset can be scored separately."""

        def one(r):
            p = ask(r, arm)
            return p is not None and ok(p, r["label_id"])

        res = {}
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(one, r): r for r in test}
            for f in as_completed(futs):
                res[id(futs[f])] = f.result()
        return res

    off, org, alle = acc("off"), acc("org"), acc("all")
    # A single-candidate mention has nothing to disambiguate: evidence cannot change
    # it either way, so it only dilutes Δ. Report the ambiguous subset separately —
    # that is the population the method actually addresses.
    amb = [r for r in test if r.get("_nc", 0) > 1]
    pct = lambda d, rows: 100 * sum(d[id(r)] for r in rows) / len(rows) if rows else 0.0
    return (
        pct(off, test),
        pct(org, test),
        pct(alle, test),
        len(test),
        pct(off, amb),
        pct(org, amb),
        pct(alle, amb),
        len(amb),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    print("三臂: off=title+desc | org=+组织后的证据 | all=+全部原始metadata(未整形)")
    print("全量 = 测试集所有样本; 歧义 = >=2 候选的子集(单候选无歧义可消,只稀释 Δ)\n")
    print(f"{'':15}{'|':>2}{'全量样本':^32}{'|':>2}{'歧义子集':^32}")
    print(
        f"{'dataset':<15}{'|':>2}{'n':>5}{'off':>6}{'org':>6}{'all':>6}"
        f"{'o-off':>7}{'a-off':>7}{'|':>2}{'n':>5}{'off':>6}{'org':>6}{'all':>6}"
        f"{'o-off':>7}{'a-off':>7}"
    )
    print("-" * 83)
    rows = []
    for ds, kind in DATASETS:
        os.environ["EL_FORCE_GENERIC"] = "1" if ds != "daime_ft_w100" else ""
        if ds == "daime_ft_w100":
            os.environ.pop("EL_FORCE_GENERIC", None)
        try:
            off, org, alle, n, aoff, aorg, aall, an = run(ds, kind, args.n)
            rows.append((ds, off, org, alle, n, aoff, aorg, aall, an))
            print(
                f"{ds:<15}{'|':>2}{n:>5}{off:>6.1f}{org:>6.1f}{alle:>6.1f}"
                f"{org - off:>+7.1f}{alle - off:>+7.1f}{'|':>2}{an:>5}{aoff:>6.1f}"
                f"{aorg:>6.1f}{aall:>6.1f}{aorg - aoff:>+7.1f}{aall - aoff:>+7.1f}",
                flush=True,
            )
        except Exception as e:
            print(f"{ds:<15}  FAIL {repr(e)[:60]}", flush=True)
    if rows:
        print("-" * 83)
        m = lambda f: sum(f(r) for r in rows) / len(rows)
        pos = lambda f: sum(1 for r in rows if f(r) > 0.5)
        for lbl, o, g, a in (("全量", 1, 2, 3), ("歧义", 5, 6, 7)):
            print(
                f"{lbl}均值:  org-off {m(lambda r: r[g] - r[o]):+5.1f} ({pos(lambda r: r[g] - r[o])}/{len(rows)} 为正)"
                f"   all-off {m(lambda r: r[a] - r[o]):+5.1f} ({pos(lambda r: r[a] - r[o])}/{len(rows)})"
                f"   org-all {m(lambda r: r[g] - r[a]):+5.1f} ({pos(lambda r: r[g] - r[a])}/{len(rows)})"
            )
