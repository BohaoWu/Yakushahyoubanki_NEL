#!/usr/bin/env python3
"""Score the k20+evidence smoke: Δ (new − old) on the same first-100 mentions,
per dataset with the right matcher (daime=canon, mahanama=cluster, HIPE=plain)."""

import json
import os
import sys

ROOT = "/workspace/Yakushahyoubanki_NEL"
sys.path.insert(0, f"{ROOT}/src")
from entity_canon import build_canon_map

O = f"{ROOT}/experiment/smoke_k20"

# (dataset, old-prediction file, scoring kind)
JOBS = [
    ("daime_ft_w100", "zscot_gpt-4o-mini.jsonl", "canon"),
    ("mahanama", "zscot.jsonl", "cluster"),
] + [
    (d, "zscot_gpt-4o-mini.jsonl", "plain")
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


def matcher(ds, kind):
    ents = [json.loads(l) for l in open(f"{ROOT}/data/dataset_{ds}/entities.jsonl")]
    if kind == "cluster":
        cl = {int(e["numeric_id"]): e["metadata"]["cluster_id"] for e in ents}
        return lambda p, g: cl.get(p, -1) == cl.get(g, -2)
    if kind == "canon":
        canon, _ = build_canon_map(ents, mode="full")
        cid = lambda i: canon.get(i, i)
        return lambda p, g: cid(p) == cid(g)
    return lambda p, g: p == g


def acc(recs, idxs, m):
    r = {x["sample_idx"]: x for x in recs}
    ok = sum(
        1
        for i in idxs
        if i in r
        and r[i].get("gold_id") is not None
        and m(r[i].get("pred_id", -1), r[i]["gold_id"])
    )
    return 100 * ok / len(idxs)


print(f"\n{'dataset':<15}{'old':>8}{'new(k20+ev)':>13}{'Δ':>7}")
print("-" * 43)
deltas = []
for ds, oldfn, kind in JOBS:
    newf = f"{O}/{ds}_new.jsonl"
    oldf = f"{ROOT}/experiment/predictions_{ds}/{oldfn}"
    if not (os.path.exists(newf) and os.path.exists(oldf)):
        print(f"{ds:<15}  (missing: new={os.path.exists(newf)} old={os.path.exists(oldf)})")
        continue
    new = [json.loads(l) for l in open(newf)]
    idxs = [x["sample_idx"] for x in new]
    old = [json.loads(l) for l in open(oldf)]
    m = matcher(ds, kind)
    a_old, a_new = acc(old, idxs, m), acc(new, idxs, m)
    deltas.append((ds, a_new - a_old))
    print(f"{ds:<15}{a_old:>8.1f}{a_new:>13.1f}{a_new - a_old:>+7.1f}")
print("-" * 43)
if deltas:
    up = sum(1 for _, d in deltas if d > 0.5)
    print(f"涨(Δ>0.5): {up}/{len(deltas)}  | 均值 Δ {sum(d for _, d in deltas) / len(deltas):+.1f}")
