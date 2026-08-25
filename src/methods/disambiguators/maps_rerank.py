#!/usr/bin/env python3
"""Merged disambiguators — dispatch: python maps_rerank.py <protocol> [args].

── maps ──
MAPS-style disambiguator for kabuki actor entity linking.

Transplants the MAPS translation strategy (He et al., TACL 2024 — Multi-Aspect
Prompting and Selection) onto entity-linking / daime disambiguation. MAPS
translates by (1) mining knowledge — keywords / topics / demonstrations, (2)
integrating each knowledge type into a separate candidate translation, then (3)
selecting the best candidate with a quality estimator (COMET-QE). We keep the
same three stages but the output space is an entity_id, not a sentence:

  1. Aspect mining (1 LLM call): extract three orthogonal disambiguation VIEWS
     of the mention's context, each a distinct MAPS "knowledge" type:
       - contemporaries : verbatim co-actor names sharing the playbill (the
                           target's same-year peers → pin the generation)
       - chronology     : how the mention year aligns to kaimeihyou periods /
                           lifespans (which generation was active then)
       - nomenclature   : the name FORM itself (haimyou / yago / abbreviation)
                           and which generation historically held it
  2. Aspect integration (base + one call per view): each view independently
     picks a finalist entity_id from the SAME deterministic evidence, grounded
     on its own mined knowledge → up to 4 candidate picks (MAPS "integration").
  3. Selection (MAPS "knowledge selection"): if the picks agree, take the
     consensus for free; otherwise a QE-style selector LLM reads every view's
     (pick, rationale) and chooses the best-supported entity_id. Deterministic
     tie-break = year-plausibility score.

Reuses the MAD/tool infrastructure by import (candidate gathering, structured
kaimeihyou/year evidence, verbatim co-star extraction) with that code unchanged,
so this file only adds the mine→integrate→select control flow. Output is the
same unified eval JSONL as MAD (sample_idx / gold_id / pred_id / ...), so
run_all's `eval` scores it directly.

Usage:
  python llm_maps_disambiguator.py       --test experiments/pair_disjoint/data/test.jsonl       --entities experiments/pair_disjoint/data/entities.jsonl       --out  experiments/pair_disjoint/predictions/maps_gpt4omini.jsonl       --model gpt-4o-mini --num_workers 20

── rerank ──
Sample-and-rerank disambiguator for kabuki actor entity linking.

Transplants the MAPS paper's `Rerank` baseline (He et al., TACL 2024) onto daime
disambiguation. In MT, Rerank samples several translations from the LLM (temp>0)
plus one greedy, then an EXTERNAL reference-free QE model (COMET-QE) scores each
candidate and the best is kept. We keep the same two stages; only the output is
an entity_id, not a sentence, and the QE scorer is an LLM judge (there is no
learned COMET-QE for JA entity linking — see `--qe` note below):

  1. Sample (MAPS "sampling"): draw N stochastic disambiguation picks at
     temperature T from the SAME grounded prompt, plus one greedy pick (temp 0),
     over the year-ranked finalist evidence -> a bag of candidate entity_ids.
  2. QE rerank (MAPS "knowledge selection", metric=comet_qe): an independent LLM
     judge scores each DISTINCT sampled candidate 0-100 for how well it fits the
     mention + context (reference-free, one call per candidate, exactly as
     COMET-QE scores each src↔mt pair independently). Pick the top; ties break on
     self-consistency vote count, then deterministic year plausibility.

Contrast with the MAPS-EL method (llm_maps_disambiguator.py): there, diversity
comes from multi-aspect prompts and selection reads the aspects' rationales;
here, diversity comes from temperature sampling one prompt and selection is a
per-candidate QE score — the same Rerank-vs-MAPS contrast the paper draws.

Reuses MAD's candidate gathering + structured evidence by import (unchanged);
writes the same unified eval JSONL as MAD (sample_idx / gold_id / pred_id / ...),
so run_all's `eval` scores it directly.

Usage:
  python llm_rerank_disambiguator.py       --test experiments/pair_disjoint/data/test.jsonl       --entities experiments/pair_disjoint/data/entities.jsonl       --out  experiments/pair_disjoint/predictions/rerank_gpt4omini.jsonl       --model gpt-4o-mini --n_samples 3 --temperature 0.7 --num_workers 20
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
    _interpretation_block,
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


def _parse_json(text):
    """Best-effort JSON object from an LLM reply (first { .. last })."""
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except Exception:
        return None


MINE_META = """You are an expert reader of Edo-period Japanese kabuki playbills and actor reviews (役者評判記). The raw text is terse: mixed kana/kanji, abbreviations, theater jargon, and role-vs-actor pairings.
You prepare structured cues for an entity-linking system that must pick the correct generation (daime / 〈n〉) of an actor.
STRICT RULE: copy every actor name, yago (house name) and haimyou (poetic name) EXACTLY as written in the original Japanese — never translate, romanize or normalize a name. Only prose explanation is in English."""

MINE_PROMPT = """Year: {year}
Target mention: {mention}
Raw context (an Edo-period kabuki playbill / review):
{sample}

Extract THREE independent views that each help pin the target's generation. Return strict JSON, no markdown:
{{
  "contemporaries": "a JSON list of the OTHER actors in the SAME passage, each name copied VERBATIM in Japanese (exclude role names and the target itself); these same-year peers are the key generation signal",
  "chronology": "1-2 English sentences on what the year {year} implies — which active period / lifespan window it points to",
  "nomenclature": "1-2 English sentences on the mention's name FORM (is it an abbreviation, a haimyou, a yago?) and which generation historically bore that exact name"
}}"""


def mine_aspects(client, model, sample):
    """One LLM call → the three MAPS-style knowledge views. None on failure."""
    md = sample.get("metadata") or {}
    ans = _ask(
        client,
        model,
        [
            {"role": "system", "content": MINE_META},
            {
                "role": "user",
                "content": MINE_PROMPT.format(
                    year=md.get("year"),
                    mention=sample.get("mention", ""),
                    sample=format_sample(sample),
                ),
            },
        ],
    )
    obj = _parse_json(ans)
    if not isinstance(obj, dict):
        return None
    cs = obj.get("contemporaries")
    if isinstance(cs, list):
        obj["contemporaries"] = "、".join(str(x) for x in cs)
    return obj


ASPECTS = [
    ("base", None, "Decide purely from the structured evidence and context."),
    (
        "contemporaries",
        "contemporaries",
        "Focus on the co-actors sharing this playbill: which finalist was active "
        "alongside exactly these contemporaries?",
    ),
    (
        "chronology",
        "chronology",
        "Focus on the year: which finalist's active period / lifespan contains it?",
    ),
    (
        "nomenclature",
        "nomenclature",
        "Focus on the name form: which finalist held this exact name (kaimeihyou / "
        "haimyou / yago) at this time?",
    ),
]

INTEGRATE_META = """You are an entity-linking expert for Japanese kabuki actor records. Given a mention (year + context), the structured facts of several finalist candidate entities, and one specific analytical focus, decide which candidate the mention refers to. Reason only from the given facts; same name with a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the right generation."""

INTEGRATE_PROMPT = """{sample}

Candidates and their facts:
{evidence}

Analytical focus: {focus}
{cue}
Choose the single entity_id (from {{{finalist_ids}}}) the mention refers to.
Output strict JSON, no markdown: {{"reasoning": "one or two sentences", "confidence": 0.0-1.0, "entity_id": INTEGER}}"""


def integrate_pick(
    client,
    model,
    sample_str,
    evidence,
    finalist_ids_str,
    finalists,
    lead_id,
    focus,
    cue,
    generic=False,
):
    """One view's grounded pick → (entity_id, confidence, reasoning, raw)."""
    cue_block = f"Mined cue: {cue}\n" if cue else ""
    ans = _ask(
        client,
        model,
        [
            {"role": "system", "content": (GENERIC_META if generic else INTEGRATE_META)},
            {
                "role": "user",
                "content": INTEGRATE_PROMPT.format(
                    sample=sample_str,
                    evidence=evidence,
                    focus=focus,
                    cue=cue_block,
                    finalist_ids=finalist_ids_str,
                ),
            },
        ],
    )
    pid, conf = _parse_pick(ans, finalists, lead_id)
    obj = _parse_json(ans) or {}
    return pid, conf, (obj.get("reasoning") or "").strip(), ans


SELECT_META = """You are the selector in an entity-linking pipeline for Japanese kabuki actors. Several analysts each proposed a candidate entity for the same mention, viewing it from a different angle (contemporaries, chronology, name form). Judge which proposal is best supported by the evidence and pick the final entity."""

SELECT_PROMPT = """Mention and context:
{sample}

Candidates and their facts:
{evidence}

The analysts' proposals (each from a different view):
{proposals}

As the selector, choose the entity_id (from {{{finalist_ids}}}) best supported by the evidence. A name at a different generation (daime / 〈n〉) is a DIFFERENT person — align generation via the year.
Output strict JSON, no markdown: {{"reasoning": "brief", "confidence": 0.0-1.0, "entity_id": INTEGER}}"""


def select_final(
    client, model, sample_str, evidence, finalist_ids_str, finalists, picks, fallback, generic=False
):
    """MAPS selection over the view picks. Consensus is free; otherwise an LLM
    selector (QE analog) decides, deterministic year-score breaks ties."""
    votes = [p[0] for p in picks if p[0] is not None]
    if votes and len(set(votes)) == 1:
        return votes[0], "consensus", None  # all views agree → no extra call
    proposals = "\n".join(
        f"- [{name}] entity_id={pid}: {reason}"
        for (name, _key, _focus), (pid, _conf, reason, _raw) in zip(ASPECTS, picks)
    )
    sel = _ask(
        client,
        model,
        [
            {"role": "system", "content": (GENERIC_META if generic else SELECT_META)},
            {
                "role": "user",
                "content": SELECT_PROMPT.format(
                    sample=sample_str,
                    evidence=evidence,
                    proposals=proposals,
                    finalist_ids=finalist_ids_str,
                ),
            },
        ],
    )
    pid, _conf = _parse_pick(sel, finalists, fallback)
    return pid, "selector", sel


def run_maps(sample, db, client, model, lead_id, finalists, select_model=None, mine=True):
    """MAPS mine→integrate→select over up to K finalists.
    Returns (final_id, method, transcript)."""
    select_model = select_model or model
    md = sample.get("metadata") or {}
    year = md.get("year")
    evidences = [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    evidence = format_evidence_block(evidences)
    finalist_ids_str = ", ".join(str(e) for e in finalists)
    sample_str = format_sample(sample, db)

    # Stage 1: mine the three views (also enrich the shown context, like TransMAD).
    aspects = mine_aspects(client, model, sample) if mine else None
    if aspects:
        interp = {
            "clarification": aspects.get("chronology", ""),
            "co_stars": [c for c in aspects.get("contemporaries", "").split("、") if c],
            "notes": aspects.get("nomenclature", ""),
        }
        sample_str = sample_str + _interpretation_block(interp)

    # Stage 2: one grounded pick per view (base always runs; views need mining).
    picks = []
    for name, key, focus in ASPECTS:
        cue = (aspects or {}).get(key) if key else None
        if key and not cue:
            picks.append((None, None, "", ""))  # view unavailable (mining failed)
            continue
        picks.append(
            integrate_pick(
                client,
                model,
                sample_str,
                evidence,
                finalist_ids_str,
                finalists,
                lead_id,
                focus,
                cue,
                generic=getattr(db, "generic", False),
            )
        )

    # Stage 3: select the final entity.
    final_id, how, sel_raw = select_final(
        client,
        select_model,
        sample_str,
        evidence,
        finalist_ids_str,
        finalists,
        picks,
        fallback=lead_id,
        generic=getattr(db, "generic", False),
    )

    # Deterministic safety net: if selection returned nothing usable, take the
    # view pick with the best year plausibility.
    if final_id not in finalists:
        scored = [(p[0], _year_score(db, p[0], year)) for p in picks if p[0] in finalists]
        final_id = max(scored, key=lambda x: x[1])[0] if scored else lead_id

    transcript = {
        "aspects": aspects,
        "picks": {
            name: {"entity_id": p[0], "confidence": p[1], "reasoning": p[2]}
            for (name, _, _), p in zip(ASPECTS, picks)
        },
        "selection": how,
        "selector_raw": sel_raw,
        "finalists": finalists,
    }
    return final_id, how, transcript


def main_maps():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--select_model", default=None, help="model for the selection stage (default = --model)"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--k", type=int, default=5, help="number of year-ranked finalists the views choose among"
    )
    ap.add_argument(
        "--no_mine",
        action="store_true",
        help="skip aspect mining (base pick only + selection); ablation",
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
        """Disambiguate one sample (baseline + MAPS on genuine ambiguity).
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
                final_id, _how, transcript = run_maps(
                    sample,
                    db,
                    client,
                    args.model,
                    lead_id,
                    finalists,
                    select_model=args.select_model,
                    mine=not args.no_mine,
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

    base_correct = maps_correct = total = 0
    multi_total = multi_base_correct = multi_maps_correct = flips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            maps_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            maps_correct += maps_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_maps_correct += maps_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"maps={maps_correct}/{total}={100 * maps_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline: {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall maps:     {maps_correct}/{total} = {100 * maps_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline: {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand maps:     {multi_maps_correct}/{multi_total} = "
            f"{100 * multi_maps_correct / multi_total:.1f}%"
        )
    print(f"  maps flipped {flips} answers")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_json(text):
    """Best-effort JSON object from an LLM reply (first { .. last })."""
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except Exception:
        return None


def _ask_t(client, model, messages, temperature, max_retries=6):
    """chat completion with a settable temperature (the shared _ask is temp-0
    only; sampling needs temp>0). Exponential backoff on transient errors."""
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


PICK_META = """You are an entity-linking expert for Japanese kabuki actor records. Given a mention (year + context) and the structured facts of several finalist candidate entities, decide which candidate the mention refers to. Reason only from the given facts; same name at a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the right generation."""

PICK_PROMPT = """{sample}

Candidates and their facts:
{evidence}

Choose the single entity_id (from {{{finalist_ids}}}) the mention refers to.
Output strict JSON, no markdown: {{"reasoning": "one or two sentences", "confidence": 0.0-1.0, "entity_id": INTEGER}}"""


def sample_picks(
    client,
    model,
    sample_str,
    evidence,
    finalist_ids_str,
    finalists,
    lead_id,
    n_samples,
    temperature,
    generic=False,
):
    """N stochastic picks (temp>0) + 1 greedy pick (temp 0) from the same prompt.
    Returns a list of entity_ids (fallbacks resolved to a finalist)."""
    msgs = [
        {"role": "system", "content": (GENERIC_META if generic else PICK_META)},
        {
            "role": "user",
            "content": PICK_PROMPT.format(
                sample=sample_str, evidence=evidence, finalist_ids=finalist_ids_str
            ),
        },
    ]
    picks = []
    for i in range(n_samples + 1):
        temp = 0.0 if i == n_samples else temperature  # last draw is greedy
        ans = _ask_t(client, model, msgs, temp)
        pid, _conf = _parse_pick(ans, finalists, lead_id)
        picks.append(pid)
    return picks


QE_META = """You are a reference-free quality estimator for Japanese kabuki actor entity linking — the entity-linking analogue of COMET-QE. You are given a mention (with its year and context) and ONE candidate entity's structured facts. Score, independently of any other candidate, how well this candidate fits the mention. Judge strictly on evidence: does the year fall in the candidate's active period / lifespan, does the candidate's kaimeihyou name at that year match the mention, do co-actor / house cues agree? A name at the wrong generation (daime / 〈n〉) is a POOR fit even if the surface name matches."""

QE_PROMPT = """Mention and context:
{sample}

Candidate entity facts:
{evidence}

Score how well THIS candidate fits the mention, from 0 (clearly wrong) to 100 (certainly correct). Judge this candidate on its own, not relative to others.
Output strict JSON, no markdown: {{"reasoning": "brief", "score": INTEGER 0-100}}"""


def qe_score(client, model, sample_str, db, eid, year):
    """One independent QE call scoring candidate `eid` 0-100. -1 on parse fail."""
    ev = json.dumps(build_evidence(db, eid, year), ensure_ascii=False, indent=2)
    ans = _ask_t(
        client,
        model,
        [
            {"role": "system", "content": QE_META},
            {"role": "user", "content": QE_PROMPT.format(sample=sample_str, evidence=ev)},
        ],
        temperature=0.0,
    )
    obj = _parse_json(ans) or {}
    try:
        return int(obj.get("score"))
    except (TypeError, ValueError):
        return -1


def run_rerank(
    sample, db, client, model, lead_id, finalists, qe_model=None, n_samples=3, temperature=0.7
):
    """MAPS-style sample -> QE-rerank over up to K finalists.
    Returns (final_id, transcript)."""
    qe_model = qe_model or model
    md = sample.get("metadata") or {}
    year = md.get("year")
    evidences = [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    evidence = format_evidence_block(evidences)
    finalist_ids_str = ", ".join(str(e) for e in finalists)
    sample_str = format_sample(sample, db)

    # Stage 1: sample a bag of candidate picks.
    picks = sample_picks(
        client,
        model,
        sample_str,
        evidence,
        finalist_ids_str,
        finalists,
        lead_id,
        n_samples,
        temperature,
        generic=getattr(db, "generic", False),
    )
    votes = Counter(p for p in picks if p in finalists)
    distinct = list(votes)
    if not distinct:
        return lead_id, {"picks": picks, "votes": {}, "qe": {}, "note": "no valid pick"}

    # Stage 2: external QE judge scores each DISTINCT candidate independently.
    qe = {eid: qe_score(client, qe_model, sample_str, db, eid, year) for eid in distinct}

    # Select: max QE score; ties -> more self-consistency votes -> year plausibility.
    def rank_key(eid):
        return (qe[eid], votes[eid], _year_score(db, eid, year))

    final_id = max(distinct, key=rank_key)

    transcript = {
        "picks": picks,
        "votes": {str(k): v for k, v in votes.items()},
        "qe": {str(k): v for k, v in qe.items()},
        "finalists": finalists,
    }
    return final_id, transcript


def main_rerank():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--qe_model", default=None, help="model for the QE judge stage (default = --model)"
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="stochastic samples per mention (a greedy sample is always added)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature for the stochastic picks",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--k", type=int, default=5, help="number of year-ranked finalists to sample/rerank among"
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
        """Disambiguate one sample (baseline + sample/rerank on genuine ambiguity).
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
                final_id, transcript = run_rerank(
                    sample,
                    db,
                    client,
                    args.model,
                    lead_id,
                    finalists,
                    qe_model=args.qe_model,
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

    base_correct = rr_correct = total = 0
    multi_total = multi_base_correct = multi_rr_correct = flips = 0

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            rr_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            rr_correct += rr_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_rr_correct += rr_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"rerank={rr_correct}/{total}={100 * rr_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline: {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall rerank:   {rr_correct}/{total} = {100 * rr_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline: {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand rerank:   {multi_rr_correct}/{multi_total} = "
            f"{100 * multi_rr_correct / multi_total:.1f}%"
        )
    print(f"  rerank flipped {flips} answers")


# --- dispatch ---
if __name__ == "__main__":
    import sys as _s

    _D = {"maps": main_maps, "rerank": main_rerank}
    if len(_s.argv) < 2 or _s.argv[1] not in _D:
        _s.exit("usage: maps_rerank.py {maps|rerank} [args...]")
    _D[_s.argv.pop(1)]()
