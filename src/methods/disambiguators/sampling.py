#!/usr/bin/env python3
"""Merged disambiguators — dispatch: python sampling.py <protocol> [args].

── selfcon ──
Self-Consistency disambiguator for kabuki actor entity linking.

Transplants Self-Consistency (Wang et al., ICLR 2023 — "Self-Consistency
Improves Chain of Thought Reasoning in Language Models") onto daime
disambiguation. The method: sample several CoT reasoning paths at temperature>0
from the SAME prompt, then marginalize over the paths by MAJORITY-VOTING the
final answers. We keep that exactly; only the answer is an entity_id chosen from
the finalist candidates:

  1. Sample N CoT paths: "reason step by step, then give the entity_id" drawn at
     temperature T from one grounded prompt -> N (reasoning, entity_id) paths.
  2. Marginalize: the final entity is the most-voted entity_id across the N
     paths. Ties break on deterministic year plausibility, then the year-rank
     lead — no external scorer (the whole point vs `rerank`, which QE-reranks).

Contrast with the other sampling method here (llm_rerank_disambiguator.py):
same temperature-sampling stage, but selection is plain majority vote instead of
a per-candidate QE judge — the pure Self-Consistency baseline.

Reuses MAD's candidate gathering + structured evidence by import (unchanged);
writes the same unified eval JSONL as MAD (sample_idx / gold_id / pred_id / ...),
so run_all's `eval` scores it directly.

Usage:
  python llm_selfcon_disambiguator.py       --test experiments/pair_disjoint/data/test.jsonl       --entities experiments/pair_disjoint/data/entities.jsonl       --out  experiments/pair_disjoint/predictions/selfcon_gpt4omini.jsonl       --model gpt-4o-mini --n_samples 5 --temperature 0.7 --num_workers 20

── cove ──
Chain-of-Verification (CoVe) disambiguator for kabuki actor entity linking.

Transplants Chain-of-Verification (Dhuliawala et al., 2023 — "Chain-of-
Verification Reduces Hallucination in Large Language Models") onto daime
disambiguation, following the reference implementation's four stages
(github.com/ritun16/chain-of-verification). Only the answer space changes to an
entity_id chosen from the finalist candidates:

  1. Baseline: pick a finalist entity_id for the mention (with a short reason).
  2. Plan verifications: from the mention + baseline pick, generate targeted
     verification questions about the pick's fit (year in active period? the
     kaimeihyou name at that year matches? the co-actors are contemporaries?).
  3. Execute verifications INDEPENDENTLY: answer each question using ONLY the
     deterministic structured evidence of the finalists — not free recall and
     not the baseline's own reasoning. This independence is CoVe's core: it
     surfaces where the baseline pick contradicts the facts.
  4. Refine: given the mention, the baseline pick and the verification Q&A,
     output the final entity_id (switching candidate if the checks say so).

The evidence fact sheet (year_active / kaimeihyou_at_year / bio) that grounds
stage 3 is exactly what MAD/MAPS use, reused by import. Writes the same unified
eval JSONL as MAD, so run_all's `eval` scores it directly.

Usage:
  python llm_cove_disambiguator.py       --test experiments/pair_disjoint/data/test.jsonl       --entities experiments/pair_disjoint/data/entities.jsonl       --out  experiments/pair_disjoint/predictions/cove_gpt4omini.jsonl       --model gpt-4o-mini --num_workers 20
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llm_debate_disambiguator import (  # noqa: E402  # noqa: E402
    GENERIC_META,
    _ask,
    _parse_pick,
    _year_score,
    build_evidence,
    format_evidence_block,
    pick_finalists,
)
from llm_tool_disambiguator import (  # noqa: E402
    ENTS_PATH,
    EntityDB,
    disambiguate_one,
    format_sample,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ask_t(client, model, messages, temperature, max_retries=6):
    """chat completion with a settable temperature (the shared _ask is temp-0
    only; sampling CoT paths needs temp>0). Exponential backoff on errors."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


SC_META = """You are an entity-linking expert for Japanese kabuki actor records. Given a mention (year + context) and the structured facts of several finalist candidate entities, determine which candidate the mention refers to. Reason only from the given facts; same name at a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the right generation."""

SC_PROMPT = """{sample}

Candidates and their facts:
{evidence}

Question: which candidate entity_id (from {{{finalist_ids}}}) does the mention refer to?
Let's think step by step, then end with the final answer as strict JSON on its own line: {{"entity_id": INTEGER}}"""


def run_selfcon(sample, db, client, model, lead_id, finalists, n_samples=5, temperature=0.7):
    """Sample N CoT paths, majority-vote the answers. Returns (final_id, transcript)."""
    md = sample.get("metadata") or {}
    year = md.get("year")
    evidence = format_evidence_block(
        [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    )
    finalist_ids_str = ", ".join(str(e) for e in finalists)
    msgs = [
        {"role": "system", "content": (GENERIC_META if getattr(db, "generic", False) else SC_META)},
        {
            "role": "user",
            "content": SC_PROMPT.format(
                sample=format_sample(sample, db), evidence=evidence, finalist_ids=finalist_ids_str
            ),
        },
    ]

    paths = []
    for _ in range(n_samples):
        ans = _ask_t(client, model, msgs, temperature)
        pid, _conf = _parse_pick(ans, finalists, lead_id)
        paths.append((pid, ans))

    votes = Counter(p for p, _ in paths if p in finalists)
    if not votes:
        return lead_id, {"votes": {}, "paths": [a for _, a in paths], "note": "no valid path"}

    # Majority vote; ties -> deterministic year plausibility -> year-rank lead.
    top = max(votes.values())
    tied = [eid for eid, c in votes.items() if c == top]
    final_id = (
        tied[0]
        if len(tied) == 1
        else max(tied, key=lambda e: (_year_score(db, e, year), e == lead_id))
    )
    transcript = {
        "votes": {str(k): v for k, v in votes.items()},
        "paths": [a for _, a in paths],
        "finalists": finalists,
    }
    return final_id, transcript


def main_selfcon():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="number of sampled CoT reasoning paths to vote over",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="sampling temperature for the CoT paths"
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
    print(
        f"[start] model={args.model} samples={len(test)} "
        f"n_samples={args.n_samples} temp={args.temperature}"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()

    def process_sample(i, sample):
        """Disambiguate one sample (baseline + self-consistency on genuine
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
                final_id, transcript = run_selfcon(
                    sample,
                    db,
                    client,
                    args.model,
                    lead_id,
                    finalists,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
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

    base_correct = sc_correct = total = 0
    multi_total = multi_base_correct = multi_sc_correct = flips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            sc_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            sc_correct += sc_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_sc_correct += sc_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"selfcon={sc_correct}/{total}={100 * sc_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline: {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall selfcon:  {sc_correct}/{total} = {100 * sc_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline: {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand selfcon:  {multi_sc_correct}/{multi_total} = "
            f"{100 * multi_sc_correct / multi_total:.1f}%"
        )
    print(f"  selfcon flipped {flips} answers")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COVE_META = """You are an entity-linking expert for Japanese kabuki actor records. A mention (year + context) must be linked to the correct finalist candidate entity. Same name at a different generation (daime / 〈n〉) is a DIFFERENT person — the year decides the generation. Reason strictly from the provided structured facts."""

BASELINE_PROMPT = """{sample}

Candidates and their facts:
{evidence}

Which candidate entity_id (from {{{finalist_ids}}}) does the mention refer to?
Output strict JSON, no markdown: {{"reasoning": "one or two sentences", "entity_id": INTEGER}}"""

PLAN_PROMPT = """Mention: {mention} (year {year})
Baseline answer: entity_id={pick}

Create a short numbered list of verification questions that check whether the mention really refers to entity_id={pick}. Focus each question on ONE checkable fact:
- is the year {year} inside that candidate's active period / lifespan?
- does that candidate's kaimeihyou name at {year} match the mention surface?
- are the co-actors in the context contemporaries of that candidate?
Output only the numbered questions."""

EXECUTE_PROMPT = """Answer each verification question using ONLY the structured facts below. Do not use outside knowledge and do not assume the baseline answer is correct. Be concise; if the facts do not support a "yes", say "no" and state what the facts show.

Structured facts for all finalist candidates:
{evidence}

Verification questions:
{questions}

Answer them as a numbered list, one line each."""

REFINE_PROMPT = """{sample}

Candidates and their facts:
{evidence}

Baseline answer: entity_id={pick}
Verification questions & answers:
{verifications}

Using the verification results, give the final answer: the entity_id (from {{{finalist_ids}}}) the mention truly refers to. If the checks contradict the baseline, switch to the candidate the facts support.
Output strict JSON, no markdown: {{"reasoning": "brief", "entity_id": INTEGER}}"""


def run_cove(sample, db, client, model, lead_id, finalists):
    """Chain-of-Verification over up to K finalists. Returns (final_id, transcript)."""
    md = sample.get("metadata") or {}
    year = md.get("year")
    mention = sample.get("mention", "")
    evidence = format_evidence_block(
        [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    )
    finalist_ids_str = ", ".join(str(e) for e in finalists)
    sample_str = format_sample(sample, db)

    # Stage 1: baseline pick.
    base_ans = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else COVE_META),
            },
            {
                "role": "user",
                "content": BASELINE_PROMPT.format(
                    sample=sample_str, evidence=evidence, finalist_ids=finalist_ids_str
                ),
            },
        ],
    )
    base_id, _c = _parse_pick(base_ans, finalists, lead_id)

    # Stage 2: plan verification questions.
    questions = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else COVE_META),
            },
            {
                "role": "user",
                "content": PLAN_PROMPT.format(mention=mention, year=year, pick=base_id),
            },
        ],
    )

    # Stage 3: execute verifications independently on the evidence only.
    verifications = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else COVE_META),
            },
            {
                "role": "user",
                "content": EXECUTE_PROMPT.format(evidence=evidence, questions=questions),
            },
        ],
    )

    # Stage 4: refine.
    refine_ans = _ask(
        client,
        model,
        [
            {
                "role": "system",
                "content": (GENERIC_META if getattr(db, "generic", False) else COVE_META),
            },
            {
                "role": "user",
                "content": REFINE_PROMPT.format(
                    sample=sample_str,
                    evidence=evidence,
                    pick=base_id,
                    verifications=verifications,
                    finalist_ids=finalist_ids_str,
                ),
            },
        ],
    )
    final_id, _c = _parse_pick(refine_ans, finalists, base_id)

    transcript = {
        "baseline_pick": base_id,
        "baseline": base_ans,
        "questions": questions,
        "verifications": verifications,
        "refine": refine_ans,
        "finalists": finalists,
    }
    return final_id, transcript


def main_cove():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--k", type=int, default=5, help="number of year-ranked finalists to verify among"
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
    print(f"[start] model={args.model} samples={len(test)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()

    def process_sample(i, sample):
        """Disambiguate one sample (baseline + CoVe on genuine ambiguity).
        Independent per sample -> safe to run concurrently."""
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
                final_id, transcript = run_cove(sample, db, client, args.model, lead_id, finalists)
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

    base_correct = cove_correct = total = 0
    multi_total = multi_base_correct = multi_cove_correct = flips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            cove_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            cove_correct += cove_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_cove_correct += cove_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"cove={cove_correct}/{total}={100 * cove_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline: {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall cove:     {cove_correct}/{total} = {100 * cove_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline: {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand cove:     {multi_cove_correct}/{multi_total} = "
            f"{100 * multi_cove_correct / multi_total:.1f}%"
        )
    print(f"  cove flipped {flips} answers")


# --- dispatch ---
if __name__ == "__main__":
    import sys as _s

    _D = {"selfcon": main_selfcon, "cove": main_cove}
    if len(_s.argv) < 2 or _s.argv[1] not in _D:
        _s.exit("usage: sampling.py {selfcon|cove} [args...]")
    _D[_s.argv.pop(1)]()
