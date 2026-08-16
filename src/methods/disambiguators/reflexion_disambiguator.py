#!/usr/bin/env python3
"""Reflexion disambiguator for yakusya kabuki-actor entity linking.

A Reflexion-style derivative of MAD (Shinn et al., 2023). It REUSES MAD's
machinery by import only (candidate gathering, finalist selection, the structured
kaimeihyou/year evidence sheet) — the MAD code is NOT modified — but replaces the
3-role *debate* with a single agent's **self-reflection retry loop**:

  per mention (multi-candidate cases only; singletons take the tool baseline):
    reflections = []
    for trial in 1..max_trials:
      1) Solver : pick entity_id from the evidence, conditioned on past reflections
      2) Verifier (NO gold — self-evaluation): check the pick's year falls in the
         candidate's active period, its kaimeihyou name matches at that year, and
         the generation (daime) is right -> {ok, critique}
      3) if ok or last trial -> stop; else Reflect: turn the critique into a short
         lesson appended to `reflections`, and retry the solver

The verifier is gold-free (the original Reflexion used the answer key; that would
be test-time cheating), so this is deployable. Same evidence as MAD -> a clean
debate-vs-reflexion comparison on the 1479 代目 benchmark.
"""

# --- path bootstrap: methods split into subpackages but modules import each
# other by bare name, so put src/, src/methods and all method subfolders on sys.path ---
import os as _bos
import sys as _bsys

_bmroot = _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__)))
for _bd in (
    _bos.path.dirname(_bmroot),
    _bmroot,
    *(_bos.path.join(_bmroot, _bs) for _bs in _bos.listdir(_bmroot)),
):
    if (
        _bos.path.isdir(_bd)
        and not _bos.path.basename(_bd).startswith("__")
        and _bd not in _bsys.path
    ):
        _bsys.path.insert(0, _bd)
# --- end bootstrap ---
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# same-dir sibling modules (added to sys.path when run as a script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_debate_disambiguator import (  # noqa: E402  (MAD machinery, reused read-only)
    GENERIC_META,  # noqa
    _ask,
    _parse_pick,
    build_evidence,
    format_evidence_block,
    pick_finalists,
)
from llm_tool_disambiguator import EntityDB, disambiguate_one, format_sample  # noqa: E402

SOLVER_META = """You are an expert in entity linking for Japanese kabuki actor records. Given a mention (with its year and context) and the structured facts of finalist candidates (birth/death, the kaimeihyou stage-name history per period, a short bio), determine which candidate the mention refers to. Reason strictly from the given facts: the year must fall in the candidate's active period; at that year the candidate's kaimeihyou name must match the mention (possibly an abbreviation or a haimyou); same name with a different generation (daime/〈n〉) is a DIFFERENT person. Do not fabricate."""

SOLVE_PROMPT = """{sample}
Candidates and their facts:
{evidence}
{reflections}
Pick the single candidate (entity_id in {{{finalist_ids}}}) the mention refers to.
Output strictly as JSON, no markdown: {{"reasoning": "brief, tied to year/kaimeihyou/daime", "entity_id": INTEGER}}"""

VERIFIER_META = """You are a meticulous verifier for kabuki-actor entity linking. You are NOT given the answer key. Judge only whether a proposed choice is internally consistent with the structured facts."""

VERIFY_PROMPT = """The proposed answer is entity_id={pick}.
Mention and candidate facts:
{sample}
{evidence}

Check the proposal against the facts, then decide:
- Is the mention year clearly OUTSIDE entity_id={pick}'s known lifespan / active
  (kaimeihyou) period? (Unknown/undated is NOT a failure — only a concrete
  contradiction is.)
- Does its kaimeihyou name at that year plausibly match the mention?
- Among the OTHER candidates, is there a SPECIFIC one that fits the year /
  kaimeihyou / generation (daime/〈n〉) clearly BETTER than the proposed one?
Mark ok=false ONLY when there is a concrete contradiction, or another candidate is
a clearly better fit (name its id). Otherwise mark ok=true. Do not nitpick.
Output strictly as JSON: {{"ok": true/false, "critique": "the concrete contradiction or the clearly-better candidate id and why"}}"""

REFLECT_META = """You are a self-reflective entity-linking agent improving across trials."""

REFLECT_PROMPT = """Your previous answer entity_id={pick} was judged inconsistent.
Verifier's critique: {critique}

In 1-2 sentences, diagnose why that pick was likely wrong and state a concrete, high-level lesson for the next attempt (which year/kaimeihyou/daime cue to weigh). Do not name a new answer; just the lesson."""

REFLECTION_HEADER = "\nLessons from your previous failed attempts (use them):\n"


def _parse_json(text, default):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else default
    except Exception:
        return default


# ---- deterministic verifier: an INDEPENDENT error signal the solver didn't use ----
# (year consistency from the same birth/death + kaimeihyou periods used to resolve
#  golds in build_daime_eval; ±5y grace for the imprecise Edo dating)


def _active_window(e):
    m = e.get("metadata", {})
    ys = []
    for p in m.get("kaimeihyou") or []:
        ys += [y for y in (p.get("start_year"), p.get("end_year")) if y]
    if m.get("birth_year"):
        ys.append(m["birth_year"])
    if m.get("death_year"):
        ys.append(m["death_year"])
    return (min(ys), max(ys)) if ys else None


def _year_consistent(db, eid, year, grace=5):
    e = db.by_id.get(int(eid))
    w = _active_window(e) if e else None
    return bool(w and year is not None and w[0] - grace <= year <= w[1] + grace)


def verify_year(db, finalists, pick, year):
    """Deterministic gate. ok=False only when the pick is year-INCONSISTENT and at
    least one other finalist IS active at the mention year (a concrete, gold-free
    error signal). Otherwise ok=True (no reliable signal -> don't nag)."""
    if year is None:
        return True, ""
    yc = [e for e in finalists if _year_consistent(db, e, year)]
    if _year_consistent(db, pick, year) or not yc:
        return True, ""
    titles = ", ".join(f"{e}({db.by_id[int(e)]['title']})" for e in yc)
    return False, (
        f"entity_id={pick} is NOT active in year {year} (the mention year "
        f"falls outside its known lifespan/kaimeihyou period), whereas these "
        f"candidates ARE active at {year}: {titles}. Reconsider those."
    )


def reflexion_one(
    sample_str,
    evidence,
    finalists,
    db,
    year,
    client,
    model,
    judge_model,
    max_trials,
    verify_mode="deterministic",
):
    """Self-reflection retry loop over the finalists. Returns (final_id, transcript).

    verify_mode: 'deterministic' = gate on year consistency (independent signal,
    recommended); 'llm' = an LLM self-verifier (no independent signal)."""
    finalist_ids = ", ".join(str(e) for e in finalists)
    reflections, trail = [], []
    final_id = finalists[0]
    for t in range(1, max_trials + 1):
        refl_block = (
            (REFLECTION_HEADER + "\n".join(f"- {r}" for r in reflections)) if reflections else ""
        )
        solve = _ask(
            client,
            model,
            [
                {
                    "role": "system",
                    "content": (GENERIC_META if getattr(db, "generic", False) else SOLVER_META),
                },
                {
                    "role": "user",
                    "content": SOLVE_PROMPT.format(
                        sample=sample_str,
                        evidence=evidence,
                        reflections=refl_block,
                        finalist_ids=finalist_ids,
                    ),
                },
            ],
        )
        final_id, _conf = _parse_pick(solve, finalists, final_id)
        trail.append(f"[Solve T{t}] {solve}")
        if t == max_trials:
            break
        if verify_mode == "deterministic":
            ok, critique = verify_year(db, finalists, final_id, year)
        else:
            ver = _ask(
                client,
                judge_model,
                [
                    {"role": "system", "content": VERIFIER_META},
                    {
                        "role": "user",
                        "content": VERIFY_PROMPT.format(
                            pick=final_id, sample=sample_str, evidence=evidence
                        ),
                    },
                ],
            )
            v = _parse_json(ver, {"ok": True, "critique": ""})
            ok, critique = v.get("ok", True), v.get("critique", "")
        trail.append(f"[Verify T{t}] ok={ok} {critique}")
        if ok:
            break
        refl = _ask(
            client,
            model,
            [
                {"role": "system", "content": REFLECT_META},
                {
                    "role": "user",
                    "content": REFLECT_PROMPT.format(pick=final_id, critique=critique),
                },
            ],
        )
        reflections.append(refl)
        trail.append(f"[Reflect T{t}] {refl}")
    return final_id, {
        "trail": trail,
        "n_trials": t,
        "finalists": finalists,
        "reflected": len(reflections) > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--judge_model", default=None, help="verifier model (default = --model)")
    ap.add_argument("--k", type=int, default=5, help="number of year-ranked finalists")
    ap.add_argument("--max_trials", type=int, default=3, help="reflexion trials per mention")
    ap.add_argument(
        "--verify_mode",
        choices=["deterministic", "llm"],
        default="deterministic",
        help="reflection gate: deterministic year-consistency (independent "
        "signal, default) or an LLM self-verifier",
    )
    ap.add_argument(
        "--max_steps", type=int, default=8, help="tool-use steps for the baseline answer"
    )
    ap.add_argument("--num_workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key", default=None)
    args = ap.parse_args()
    judge_model = args.judge_model or args.model

    from openai import OpenAI

    ck = {}
    if args.base_url:
        ck["base_url"] = args.base_url
    if args.api_key:
        ck["api_key"] = args.api_key
    client = OpenAI(**ck)

    ents = [json.loads(l) for l in open(args.entities)]
    db = EntityDB(ents)
    print(f"[entities] {len(ents)} from {args.entities}")
    test = [json.loads(l) for l in open(args.test)]
    if args.limit:
        test = test[args.start : args.start + args.limit]
    print(
        f"[start] model={args.model} samples={len(test)} max_trials={args.max_trials} "
        f"workers={args.num_workers}"
    )

    def process_sample(i, sample):
        gold = int(sample["label_id"])
        base = disambiguate_one(sample, db, client, args.model, args.max_steps)
        a_id = base["pred_id"]
        from llm_tool_disambiguator import get_candidate_list  # noqa: E402 (lazy: avoids circular import)

        cands = get_candidate_list(
            db, sample, topk=None
        )  # CAND_METHOD env; default 'gather' == old
        is_multi = len(cands) >= 2
        info = None
        if is_multi:
            lead_id, finalists = pick_finalists(cands, a_id, k=args.k)
            if len(finalists) >= 2:
                year = (sample.get("metadata") or {}).get("year")
                evidence = format_evidence_block(
                    [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
                )
                final_id, info = reflexion_one(
                    format_sample(sample, db),
                    evidence,
                    finalists,
                    db,
                    year,
                    client,
                    args.model,
                    judge_model,
                    args.max_trials,
                    args.verify_mode,
                )
            else:
                final_id = a_id
        else:
            final_id = a_id
        return {
            "sample_idx": args.start + i,
            "mention": sample.get("mention"),
            "type": sample.get("type"),
            "gold_id": gold,
            "gold_title": sample.get("label_title", ""),
            "n_candidates": len(cands),
            "is_multi": is_multi,
            "pred_id_baseline": a_id,
            "pred_id": final_id,
            "reflected": bool(info and info["reflected"]),
            "n_trials": info["n_trials"] if info else 0,
            "transcript": info,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()
    correct = total = 0
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            correct += rec["pred_id"] == rec["gold_id"]
            total += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 10 == 0 or done + 1 == len(test):
                print(
                    f"  [{done + 1}/{len(test)}] acc={correct}/{total}={100 * correct / total:.1f}% "
                    f"elapsed={time.time() - t0:.0f}s"
                )
    out_f.close()
    print(f"\n[done] {correct}/{total} = {100 * correct / total:.1f}%")


if __name__ == "__main__":
    main()
