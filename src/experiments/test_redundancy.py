#!/usr/bin/env python3
"""Does the organized sheet earn its keep by ADDING facts, or by PRE-COMPUTING one?

daime's description is the entity's raw catalogue text, and it already contains the
whole kaimeihyou table verbatim — birth/death years appear in it 100% of the time,
the dated name periods 84.9%. So the `all` arm, which ships that table again, is
telling the model something it has already read: hence its +3.5 against the sheet's
+14.3. The sheet's one field that is NOT restating the description is
`name_at_mention_year` — not a new fact, but the answer to "which of these periods
covers the mention's year", worked out in advance.

If that reading is right, a sheet carrying ONLY the collapse should match or beat
the full sheet, since everything else is redundant bulk. Arms:

  off    title + description                      (the description already has it all)
  lean   + category + name_at_mention_year        (the collapse alone)
  org    + birth/death + name_periods too         (the current sheet)
  all    + the raw kaimeihyou table               (the dump)
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/workspace/Yakushahyoubanki_NEL"
for p in (f"{ROOT}/config", f"{ROOT}/src"):
    sys.path.insert(0, p)
import config  # noqa: E402,F401  (side effect: adds src/methods subpackages to sys.path)
for line in open(f"{ROOT}/.env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ["LLM_CACHE"] = "1"
os.environ.pop("EL_FORCE_GENERIC", None)

from llm_cache import wrap_client
from llm_debate_disambiguator import build_evidence
from llm_tool_disambiguator import EntityDB, tool_search_entities
from openai import OpenAI

from entity_canon import build_canon_map

CLIENT = wrap_client(OpenAI())
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
D = f"{ROOT}/data/dataset_daime_ft_w100"
ents = [json.loads(l) for l in open(f"{D}/entities.jsonl")]
byid = {int(e["numeric_id"]): e for e in ents}
test = [json.loads(l) for l in open(f"{D}/test.jsonl")][:230]
db = EntityDB(ents)
canon, _ = build_canon_map(ents, mode="full")
ok = lambda p, g: canon.get(p, p) == canon.get(g, g)

LEAN = ("category", "name_at_mention_year")


def sheet(c, yr, arm):
    if arm == "off":
        return ""
    ev = build_evidence(db, c, yr)
    if arm == "lean":
        return "; ".join(f"{k}={ev[k]}" for k in LEAN if ev.get(k) not in (None, "", [], {}))
    if arm == "org":
        return ev.render(cap=400)
    md = byid[c].get("metadata") or {}  # all: the raw table, unshaped
    return "; ".join(
        f"{k}={v}"
        for k, v in sorted(md.items())
        if k not in ("source", "detail_id", "wikipedia_url") and v not in (None, "", [], {})
    )[:4000]


def ask(r, arm):
    cids = [c["entity_id"] for c in tool_search_entities(db, r["mention"])][:20]
    if not cids:
        return None
    r["_nc"] = len(cids)
    yr = (r.get("metadata") or {}).get("year")
    yr = yr if isinstance(yr, int) else None
    lines = []
    for i, c in enumerate(cids):
        s = f"{i + 1}. {byid[c]['title']} -- {(byid[c].get('text') or '')[:170]}"
        e = sheet(c, yr, arm)
        if e:
            s += f"  [{e}]"
        lines.append(s)
    head = f'Mention: "{r["mention"]}"' + (f"  (year: {yr})" if yr else "")
    ctx = (
        (r.get("context_left") or "")[-250:]
        + " [["
        + r["mention"]
        + "]] "
        + (r.get("context_right") or "")[:250]
    )
    p = (
        f"{head}\nContext: {ctx[:450]}\n\nCandidates:\n"
        + "\n".join(lines)
        + '\n\nReply ONLY JSON: {"choice": INTEGER}'
    )
    try:
        t = (
            CLIENT.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=30, messages=[{"role": "user", "content": p}]
            )
            .choices[0]
            .message.content
        )
    except Exception:
        return None
    m = re.search(r'"choice"\s*:\s*(\d+)', t or "") or re.search(r"\b(\d+)\b", t or "")
    if not m:
        return None
    i = int(m.group(1)) - 1
    return (cids[i] if 0 <= i < len(cids) else None), len(p)


def run(arm):
    res, sz = {}, []

    def one(r):
        out = ask(r, arm)
        if out is None:
            return False, 0
        p, n = out
        return (p is not None and ok(p, int(r["label_id"]))), n

    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(one, r): r for r in test}
        for f in as_completed(futs):
            good, n = f.result()
            res[id(futs[f])] = good
            sz.append(n)
    return res, sum(sz) / len(sz)


if __name__ == "__main__":
    print(f"daime — 冗余消融 (n={len(test)}, {MODEL})\n")
    print(f"{'arm':<7}{'prompt字符':>10}{'准确率':>9}{'Δ':>8}")
    print("-" * 35)
    base = None
    for arm in ("off", "lean", "org", "all"):
        res, size = run(arm)
        a = 100 * sum(res.values()) / len(test)
        if base is None:
            base = a
        print(f"{arm:<7}{size:>10.0f}{a:>9.1f}{a - base:>+8.1f}", flush=True)
