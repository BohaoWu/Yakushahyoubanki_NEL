#!/usr/bin/env python3
"""Merged disambiguators — dispatch: python cot_family.py <protocol> [args].

── bare ──
Single-shot EL baseline — the ablation floor for the agent scaffolding.

One LLM call over the SAME pre-gathered candidates + evidence that the tool agent
and the debate use, but with NO tool-use loop and NO debate. Comparing this
`bare` baseline against llm_tool (tool-use agent) and mad (multi-agent debate)
isolates how much the agent scaffolding adds on top of "just pick from the
candidates". Dual-mode: daime evidence/prompt for a 改名表 KB, generic otherwise.

Usage:
  python llm_bare_disambiguator.py --test t.jsonl --entities e.jsonl --out o.jsonl       --model gpt-4o-mini --num_workers 20 [--limit N]

── zscot ──
Zero-shot-CoT disambiguator for kabuki actor entity linking.

Transplants Zero-shot-CoT (Kojima et al., NeurIPS 2022 — "Large Language Models
are Zero-Shot Reasoners") onto daime disambiguation. The paper's method is a
two-stage prompt: (1) append "Let's think step by step." to elicit a reasoning
chain z, then (2) append an answer-extraction trigger ("Therefore, the answer is
...") to read the final answer out of z. We keep both stages verbatim; only the
answer space is an entity_id chosen from the finalist candidates:

  1. Reasoning extraction: <question> + "Let's think step by step." -> chain z
  2. Answer extraction:     <question> + z + "Therefore, the answer (entity_id)
     is" -> the picked entity_id (cleansed to a finalist).

<question> lists the mention (year + context) and the year-ranked finalist
candidates with their structured facts, exactly the evidence MAD/MAPS use. This
is the simplest LLM baseline here: a single agent, two calls, no debate / no
sampling / no multi-aspect prompting — a clean reference point for the richer
methods. `--no_cot` drops the CoT trigger (the paper's plain Zero-shot ablation:
one call, direct answer).

Reuses MAD's candidate gathering + structured evidence by import (unchanged);
writes the same unified eval JSONL as MAD (sample_idx / gold_id / pred_id / ...),
so run_all's `eval` scores it directly.

Usage:
  python llm_zscot_disambiguator.py       --test experiments/pair_disjoint/data/test.jsonl       --entities experiments/pair_disjoint/data/entities.jsonl       --out  experiments/pair_disjoint/predictions/zscot_gpt4omini.jsonl       --model gpt-4o-mini --num_workers 20

── plansolve ──
Plan-and-Solve disambiguator for kabuki actor entity linking.

Transplants Plan-and-Solve (PS) Prompting (Wang et al., ACL 2023 — "Plan-and-
Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language
Models") onto daime disambiguation. PS is Zero-shot-CoT with a richer trigger
that first devises a PLAN then carries it out — a lightweight, training-free
stand-in for the "high-level plan + low-level execution" two-layer idea. Two
stages, like Zero-shot-CoT; only the answer is an entity_id:

  1. Plan+solve: <question> + the PS trigger ("Let's first understand the problem
     and devise a plan ... Then, let's carry out the plan ... step by step.") ->
     a plan followed by its execution.
  2. Answer extraction: <question> + reasoning + "Therefore, the answer
     (entity_id) is" -> the picked entity_id (cleansed to a finalist).

`--plus` uses the PS+ trigger (prompt_301, adapted for this domain: "extract the
relevant cues — year, co-actors, name form — and devise a plan ..."). Reuses
MAD's candidate gathering + evidence by import; writes the same unified eval
JSONL as MAD, so run_all's `eval` scores it directly.

Usage:
  python llm_plansolve_disambiguator.py       --test experiments/pair_disjoint/data/test.jsonl       --entities experiments/pair_disjoint/data/entities.jsonl       --out  experiments/pair_disjoint/predictions/plansolve_gpt4omini.jsonl       --model gpt-4o-mini --num_workers 20
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
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from el_mode import GENERIC_META, era_hint_line, infer_era, sample_context
from llm_debate_disambiguator import (  # noqa: E402
    GENERIC_META,
    _ask,
    _parse_pick,
    build_evidence,
    format_evidence_block,
    pick_finalists,
)
from llm_tool_disambiguator import (  # noqa: E402
    ENTS_PATH,
    EntityDB,
    disambiguate_one,
    format_sample,
    tool_search_entities,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DAIME_META = """You are an entity-linking expert for Japanese kabuki actor records. Given a mention (year + context) and the structured facts of several candidate entities, decide which candidate the mention refers to. Same name at a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the right generation. Reason only from the given facts."""

BARE_PROMPT = """{sample}
Candidates and their facts:
{evidence}

Pick the single candidate (entity_id in {finalist_ids}) the mention refers to.
Output strictly as JSON, no markdown: {{"reasoning": "brief reasoning", "entity_id": INTEGER}}"""


def bare_one(i, sample, db, client, model, k, no_agent=False):
    gold = int(sample["label_id"])
    if no_agent:
        # generic candidate recall: surface/alias match only — no family (代目)
        # expansion and no year re-rank (what a non-daime EL system would do).
        cands = [c["entity_id"] for c in tool_search_entities(db, sample.get("mention", ""))]
    else:
        from llm_tool_disambiguator import get_candidate_list  # noqa: E402 (lazy: avoids circular import)

        cands = get_candidate_list(
            db, sample, topk=None
        )  # CAND_METHOD env; default 'gather' == old
    if not cands:
        return {
            "sample_idx": i,
            "mention": sample.get("mention"),
            "gold_id": gold,
            "pred_id": -1,
            "n_candidates": 0,
        }
    lead, finalists = pick_finalists(cands, cands[0], k=k)
    year = (sample.get("metadata") or {}).get("year")
    evidence = format_evidence_block([(str(e), build_evidence(db, e, year)) for e in finalists])
    meta = GENERIC_META if getattr(db, "generic", False) else DAIME_META
    ans = _ask(
        client,
        model,
        [
            {"role": "system", "content": meta},
            {
                "role": "user",
                "content": BARE_PROMPT.format(
                    sample=format_sample(sample, db),
                    evidence=evidence,
                    finalist_ids=", ".join(str(e) for e in finalists),
                ),
            },
        ],
    )
    pid, _conf = _parse_pick(ans, finalists, lead)
    return {
        "sample_idx": i,
        "mention": sample.get("mention"),
        "type": sample.get("type"),
        "gold_id": gold,
        "pred_id": pid,
        "n_candidates": len(cands),
    }


def main_bare():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--num_workers", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--no_agent",
        action="store_true",
        help="ablate the domain agent: force generic evidence+prompt "
        "(title/description only, no year/改名表) on the same candidates",
    )
    args = ap.parse_args()

    ents = [json.loads(l) for l in open(args.entities)]
    db = EntityDB(ents)
    if args.no_agent:
        db.generic = True  # drop the year/kaimeihyou evidence + daime prompt
    test = [json.loads(l) for l in open(args.test)]
    if args.limit:
        test = test[: args.limit]

    from llm_client import make_client

    client = make_client()

    print(f"[bare] model={args.model} samples={len(test)} generic={db.generic}")
    t0 = time.time()
    results = [None] * len(test)
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {
            ex.submit(bare_one, i, s, db, client, args.model, args.k, args.no_agent): i
            for i, s in enumerate(test)
        }
        done = cor = 0
        for fut in __import__("concurrent.futures", fromlist=["as_completed"]).as_completed(futs):
            r = fut.result()
            results[r["sample_idx"]] = r
            done += 1
            cor += int(r["pred_id"] == r["gold_id"])
            if done % 25 == 0 or done == len(test):
                print(
                    f"  [{done}/{len(test)}] acc={cor}/{done}={cor / done * 100:.1f}% "
                    f"({time.time() - t0:.0f}s)"
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = sum(r["pred_id"] == r["gold_id"] for r in results) / max(len(results), 1)
    print(f"[bare] final acc = {acc * 100:.1f}% -> {args.out}")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ERA_AGENT = bool(os.environ.get("EL_ERA_AGENT"))

COT_TRIGGER = "Let's think step by step."

ANSWER_TRIGGER = "Therefore, the answer (entity_id) is"

ZSCOT_META = """You are an entity-linking expert for Japanese kabuki actor records. Given a mention (year + context) and the structured facts of several finalist candidate entities, determine which candidate the mention refers to. Reason only from the given facts; same name at a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the right generation."""

QUESTION = """{sample}

Candidates and their facts:
{evidence}

Question: which candidate entity_id (from {{{finalist_ids}}}) does the mention refer to?"""


def run_zscot(sample, db, client, model, lead_id, finalists, cot=True):
    """Zero-shot-CoT over up to K finalists. Returns (final_id, transcript)."""
    md = sample.get("metadata") or {}
    year = md.get("year")
    evidence = format_evidence_block(
        [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    )
    finalist_ids_str = ", ".join(str(e) for e in finalists)
    question = QUESTION.format(
        sample=format_sample(sample, db), evidence=evidence, finalist_ids=finalist_ids_str
    )

    # Generic mode: replace the (invalid) document-year anchor with an LLM-inferred
    # referent era, appended as a soft preference. Abstains ('') when undatable.
    if _ERA_AGENT and getattr(db, "generic", False):
        era = infer_era(client, model, sample.get("mention", ""), sample_context(sample))
        question += era_hint_line(era)

    reasoning = ""
    if cot:
        # Stage 1: reasoning extraction.
        reasoning = _ask(
            client,
            model,
            [
                {
                    "role": "system",
                    "content": (GENERIC_META if getattr(db, "generic", False) else ZSCOT_META),
                },
                {"role": "user", "content": f"{question}\n{COT_TRIGGER}"},
            ],
        )
        # Stage 2: answer extraction (question + chain + answer trigger).
        answer_user = f"{question}\n{COT_TRIGGER}{reasoning}\n{ANSWER_TRIGGER}"
    else:
        # Plain Zero-shot ablation: one call, direct answer.
        answer_user = f"{question}\n{ANSWER_TRIGGER}"

    ans = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else ZSCOT_META),
            },
            {"role": "user", "content": answer_user},
        ],
    )
    final_id, _conf = _parse_pick(ans, finalists, lead_id)

    return final_id, {"reasoning": reasoning, "answer": ans, "finalists": finalists}


def main_zscot():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--no_cot",
        action="store_true",
        help="plain Zero-shot ablation: drop 'Let's think step by step' (one call)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--k", type=int, default=5, help="number of year-ranked finalists to reason among"
    )
    ap.add_argument(
        "--entities",
        default=ENTS_PATH,
        help="entity DB jsonl (must match the dataset variant of --test)",
    )
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key", default=None)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="concurrent samples (remote LLM API calls run in parallel)",
    )
    args = ap.parse_args()

    from llm_client import make_client

    client = make_client(base_url=args.base_url, api_key=args.api_key)

    ents = [json.loads(l) for l in open(args.entities)]
    db = EntityDB(ents)
    print(f"[entities] {len(ents)} from {args.entities}")
    test = [json.loads(l) for l in open(args.test)]
    if args.limit:
        test = test[args.start : args.start + args.limit]
    method = "zero_shot" if args.no_cot else "zero_shot_cot"
    print(f"[start] method={method} model={args.model} samples={len(test)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()

    def process_sample(i, sample):
        """Disambiguate one sample (baseline + Zero-shot-CoT on genuine ambiguity).
        Independent per sample -> safe to run concurrently (client is thread-safe)."""
        gold = int(sample["label_id"])
        base = disambiguate_one(sample, db, client, args.model, args.max_steps)
        a_id = base["pred_id"]

        from llm_tool_disambiguator import get_candidate_list  # noqa: E402 (lazy: avoids circular import)

        cands = get_candidate_list(
            db, sample, topk=None
        )  # CAND_METHOD env; default 'gather' == old
        is_multi = len(cands) >= 2

        transcript = None
        if is_multi:
            lead_id, finalists = pick_finalists(cands, a_id, k=args.k)
            if len(finalists) >= 2:
                final_id, transcript = run_zscot(
                    sample, db, client, args.model, lead_id, finalists, cot=not args.no_cot
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
            "flipped": final_id != a_id,
            "transcript": transcript,
        }

    base_correct = cot_correct = total = 0
    multi_total = multi_base_correct = multi_cot_correct = flips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            cot_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            cot_correct += cot_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_cot_correct += cot_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"{method}={cot_correct}/{total}={100 * cot_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline:  {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall {method}: {cot_correct}/{total} = {100 * cot_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline:  {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand {method}:  {multi_cot_correct}/{multi_total} = "
            f"{100 * multi_cot_correct / multi_total:.1f}%"
        )
    print(f"  {method} flipped {flips} answers")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PS_TRIGGER = (
    "Let's first understand the problem and devise a plan to solve the "
    "problem. Then, let's carry out the plan to solve the problem step by step."
)

PS_PLUS_TRIGGER = (
    "Let's first understand the problem, extract the relevant cues "
    "(the mention year, the co-actors in the passage, and the name "
    "form / kaimeihyou), and devise a plan. Then, let's carry out "
    "the plan, align each candidate's active period and name to the "
    "year step by step, and show the answer."
)

ANSWER_TRIGGER = "Therefore, the answer (entity_id) is"

PS_META = """You are an entity-linking expert for Japanese kabuki actor records. Given a mention (year + context) and the structured facts of several finalist candidate entities, determine which candidate the mention refers to. Reason only from the given facts; same name at a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the right generation."""

QUESTION = """{sample}

Candidates and their facts:
{evidence}

Question: which candidate entity_id (from {{{finalist_ids}}}) does the mention refer to?"""


def run_plansolve(sample, db, client, model, lead_id, finalists, plus=False):
    """Plan-and-Solve over up to K finalists. Returns (final_id, transcript)."""
    md = sample.get("metadata") or {}
    year = md.get("year")
    evidence = format_evidence_block(
        [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    )
    finalist_ids_str = ", ".join(str(e) for e in finalists)
    question = QUESTION.format(
        sample=format_sample(sample, db), evidence=evidence, finalist_ids=finalist_ids_str
    )
    trigger = PS_PLUS_TRIGGER if plus else PS_TRIGGER

    # Stage 1: plan + solve.
    reasoning = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else PS_META),
            },
            {"role": "user", "content": f"{question}\n{trigger}"},
        ],
    )
    # Stage 2: answer extraction (question + plan/execution + answer trigger).
    ans = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else PS_META),
            },
            {"role": "user", "content": f"{question}\n{trigger}{reasoning}\n{ANSWER_TRIGGER}"},
        ],
    )
    final_id, _conf = _parse_pick(ans, finalists, lead_id)

    return final_id, {"reasoning": reasoning, "answer": ans, "finalists": finalists}


def main_plansolve():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--plus",
        action="store_true",
        help="use the PS+ trigger (prompt_301): extract cues then plan",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--k", type=int, default=5, help="number of year-ranked finalists to reason among"
    )
    ap.add_argument(
        "--entities",
        default=ENTS_PATH,
        help="entity DB jsonl (must match the dataset variant of --test)",
    )
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--api_key", default=None)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="concurrent samples (remote LLM API calls run in parallel)",
    )
    args = ap.parse_args()

    from llm_client import make_client

    client = make_client(base_url=args.base_url, api_key=args.api_key)

    ents = [json.loads(l) for l in open(args.entities)]
    db = EntityDB(ents)
    print(f"[entities] {len(ents)} from {args.entities}")
    test = [json.loads(l) for l in open(args.test)]
    if args.limit:
        test = test[args.start : args.start + args.limit]
    method = "plansolve+" if args.plus else "plansolve"
    print(f"[start] method={method} model={args.model} samples={len(test)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()

    def process_sample(i, sample):
        """Disambiguate one sample (baseline + Plan-and-Solve on genuine
        ambiguity). Independent per sample -> safe to run concurrently."""
        gold = int(sample["label_id"])
        base = disambiguate_one(sample, db, client, args.model, args.max_steps)
        a_id = base["pred_id"]

        from llm_tool_disambiguator import get_candidate_list  # noqa: E402 (lazy: avoids circular import)

        cands = get_candidate_list(
            db, sample, topk=None
        )  # CAND_METHOD env; default 'gather' == old
        is_multi = len(cands) >= 2

        transcript = None
        if is_multi:
            lead_id, finalists = pick_finalists(cands, a_id, k=args.k)
            if len(finalists) >= 2:
                final_id, transcript = run_plansolve(
                    sample, db, client, args.model, lead_id, finalists, plus=args.plus
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
            "flipped": final_id != a_id,
            "transcript": transcript,
        }

    base_correct = ps_correct = total = 0
    multi_total = multi_base_correct = multi_ps_correct = flips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            ps_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            ps_correct += ps_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_ps_correct += ps_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"{method}={ps_correct}/{total}={100 * ps_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline:  {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall {method}: {ps_correct}/{total} = {100 * ps_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline:  {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand {method}:  {multi_ps_correct}/{multi_total} = "
            f"{100 * multi_ps_correct / multi_total:.1f}%"
        )
    print(f"  {method} flipped {flips} answers")


# --- dispatch ---
if __name__ == "__main__":
    import sys as _s

    _D = {"bare": main_bare, "zscot": main_zscot, "plansolve": main_plansolve}
    if len(_s.argv) < 2 or _s.argv[1] not in _D:
        _s.exit("usage: cot_family.py {bare|zscot|plansolve} [args...]")
    _D[_s.argv.pop(1)]()
