#!/usr/bin/env python3
"""Multi-Agent-Debate (MAD) disambiguator for kabuki actor entity linking.

Prototype that grafts the MAD (Multi-Agents Debate) idea onto the existing
tool-use disambiguator (llm_tool_disambiguator.py):

  1. Run the existing tool-use agent loop -> baseline answer A (+ candidate set).
  2. If the mention has <=1 candidate -> trivial, keep A (no debate, no cost).
  3. If >1 candidate (genuine ambiguity) -> run a 3-role debate:
       - Affirmative argues for A (the tool loop's pick)
       - Negative argues for an alternative candidate B (best by year)
       - Moderator decides the final entity_id
     Both debaters are grounded on the SAME deterministically pre-fetched
     structured evidence (year_active / kaimeihyou_at_year / get_entity) for the
     two finalists, so no per-debater tool loop is needed.

This isolates the value of debate: only the genuinely ambiguous subset pays the
extra ~3 LLM calls, and we report baseline vs debate accuracy in one run.

Usage:
  python llm_debate_disambiguator.py \
      --test /workspace/BLINK/dataset_relinked_yr/test.jsonl \
      --out  /workspace/BLINK/predictions_tooldis/debate_gpt4omini.jsonl \
      --model gpt-4o-mini --limit 30
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

# Reuse everything from the tool disambiguator that lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re  # noqa: E402

from llm_tool_disambiguator import (  # noqa: E402
    ENTS_PATH,
    EntityDB,
    disambiguate_one,
    format_sample,
    parse_final_answer,
    tool_find_by_alias_year,
    tool_find_family,
    tool_get_entity,
    tool_search_entities,
    tool_year_active,
)

# Generation / paren markers stripped to derive a family base name from a mention.
_GEN_PAT = re.compile(r"〈\d+〉")


def mention_base(mention: str) -> str:
    """Best-effort family base name from a mention surface (for find_family)."""
    b = _GEN_PAT.sub("", mention or "")
    b = re.sub(r"\([^)]*\)$", "", b)
    b = re.sub(r"（[^）]*）$", "", b)
    return b.strip()


# -----------------------------------------------------------------------------
# Candidate gathering (local, no API) — decides single vs multi candidate
# -----------------------------------------------------------------------------


def _year_score(db, eid, year):
    """How plausible is `year` for entity `eid`? Higher = better.

     3 = mention-year falls inside a kaimeihyou period using a matching name
     2 = year within [birth, death]
     1 = year within a loose active window (any kaimeihyou period)
     0 = no year evidence either way (e.g. missing dates)
    -1 = year clearly outside the entity's lifespan
    """
    if not isinstance(year, int):
        return 0
    info = tool_get_entity(db, eid)
    by, dy = info.get("birth_year"), info.get("death_year")
    active = tool_year_active(db, eid).get("name_periods") or []
    for p in active:
        s, e = p.get("start_year"), p.get("end_year")
        if s and e and s <= year <= e:
            return 3
    if by and dy and by <= year <= dy:
        return 2
    if by and dy and not (by - 5 <= year <= dy + 5):
        return -1
    if active:
        return 1
    return 0


def gather_candidates(db, sample):
    """Return unique candidate ids for a mention, ranked by year plausibility.

    Combines alias-at-year matches (strongest), exact/alias/prefix title
    matches, then re-ranks the whole pool by a year-plausibility score so that
    the entities actually alive/active in the mention year float to the top.
    This is what makes a 2-way debate viable over a 20-candidate pool.
    """
    mention = sample.get("mention", "")
    md = sample.get("metadata") or {}
    year = md.get("year")

    # generic (non-daime) mode: surface/alias match only — no year-alias hits and
    # no year re-rank (this is the "without the yakusya agent" candidate path).
    if getattr(db, "generic", False):
        return [c["entity_id"] for c in tool_search_entities(db, mention)]

    year_hits = []
    if isinstance(year, int):
        year_hits = [c["entity_id"] for c in tool_find_by_alias_year(db, mention, year)]
    search_hits = [c["entity_id"] for c in tool_search_entities(db, mention)]
    # find_family catches lineage members (other generations of the same base
    # name) that surface matching misses -> lifts pool recall toward ~100%.
    family_hits = [c["entity_id"] for c in tool_find_family(db, mention_base(mention))]

    ordered, seen = [], set()
    for eid in year_hits + search_hits + family_hits:
        if eid not in seen:
            seen.add(eid)
            ordered.append(eid)

    # Stable re-rank by year plausibility (desc); ties keep original order.
    scored = [(eid, _year_score(db, eid, year)) for eid in ordered]
    scored.sort(key=lambda x: -x[1])
    return [eid for eid, _ in scored]


def pick_finalists(candidates, a_id, k=5):
    """Pick up to K finalists for the debate (year-ranked).

    The baseline answer A (if a real candidate) is always kept and placed first
    so the affirmative side has it to defend; the rest fill by year rank.
    Returns (lead_id, finalists) where finalists is the ordered K-list.
    """
    lead = a_id if (a_id is not None and a_id != -1 and a_id in candidates) else candidates[0]
    finalists = [lead] + [eid for eid in candidates if eid != lead]
    finalists = finalists[: max(2, k)]
    return lead, finalists


# -----------------------------------------------------------------------------
# Deterministic evidence pre-fetch (shared by both debaters)
# -----------------------------------------------------------------------------


def build_evidence(db, entity_id, year, context=None):
    """The candidate fact sheet fed to the reasoning module — see EvidenceAgent.

    `year` and `context` are the query keys the sheet is collapsed against: a sheet
    is only evidence relative to a query. Daime collapses its kaimeihyou table with
    `year`, mahanama its epithet cluster with `context`. Pass whatever the dataset's
    own annotation can pair with; the rest is ignored."""
    return db.agent.get_evidence_organized(entity_id, year=year, context=context)


def format_evidence_block(evidences):
    """evidences: list of (label, Evidence).

    Rendered through Evidence.render rather than json.dumps: the sheet has an order
    (keyed fields first, so a truncation never eats the decisive one) and per-field
    budgets, and dumping raw JSON silently discarded both. Title and description stay
    on the candidate's own line — render() covers only the evidence fields."""
    blocks = []
    for label, ev in evidences:
        head = f"Candidate {label}: {ev.get('title', '')}"
        desc = (ev.get("description") or "")[:400]
        body = ev.render(cap=400) if hasattr(ev, "render") else json.dumps(ev, ensure_ascii=False)
        blocks.append(
            "\n".join(
                x for x in (head, f"  {desc}" if desc else "", f"  {body}" if body else "") if x
            )
        )
    return "\n\n".join(blocks)


# -----------------------------------------------------------------------------
# Debate prompts (MAD-style, specialized for EL grounding)
# -----------------------------------------------------------------------------

PLAYER_META = """You are a debater specializing in entity linking for Japanese kabuki actor records. You are given a mention (with its year and context) and the structured facts of several finalist candidate entities (birth/death years, the kaimeihyou stage-name history per period, a short bio).
Your goal is to determine which candidate the mention refers to. Reason strictly from the provided facts, focusing on:
- whether the year falls within the candidate's lifespan / active period
- whether, at that year, the candidate's kaimeihyou name matches the mention (the mention may be an abbreviation or a haimyou/poetic name)
- co-actor / yago (house name) / lineage cues in the context
Do not fabricate facts; use only the given evidence."""

MODERATOR_META = """You are the moderator of an entity-linking debate for Japanese kabuki actors. Two debaters argue over which candidate a mention refers to. Based on the structured facts and context, decide which candidate is correct."""

# Generic (non-daime) prompts live in el_mode; re-exported here so the reasoning
# methods keep importing GENERIC_META from llm_debate unchanged.
from el_mode import (  # noqa: E402
    GENERIC_META,  # noqa: F401  (unused here; re-exported for the reasoning methods)
    MODERATOR_META_GENERIC,
    MODERATOR_PROMPT_GENERIC,
    PLAYER_META_GENERIC,
)

AFFIRMATIVE_PROMPT = """{sample}
Candidates and their facts:
{evidence}

You are the affirmative side, arguing that this mention refers to **candidate entity_id={lead_id}**. Give the strongest evidence-based reasons supporting it (year match, kaimeihyou, context). Concise and forceful, 3-5 sentences."""

NEGATIVE_PROMPT = """{aff_ans}

You are the negative side and disagree with the affirmative. Review **each** of the other candidates one by one (excluding the affirmative's entity_id={lead_id}), giving a one-sentence assessment of how well each fits the year / kaimeihyou / context, then pick the **single most likely correct one** to rebut the affirmative. Begin your answer by explicitly writing "I argue for entity_id=XXX". Be concise: one sentence per candidate plus a conclusion."""

# Multi-round rebuttal (lever 2). Each side sees the opponent's latest turn.
AFF_REBUT_PROMPT = """The negative side just argued:
{neg_ans}

You are still the affirmative side. Rebut the negative's specific points one by one, and restate why entity_id={lead_id} is correct (tied closely to the year / kaimeihyou / context evidence). Concise and forceful."""

NEG_REBUT_PROMPT = """The affirmative side just rebutted:
{aff_ans}

You are still the negative side. Respond to the affirmative's rebuttal. If you still believe another candidate is more correct, restate "I argue for entity_id=XXX" and give decisive evidence; if persuaded, you may change your position. Concise and forceful."""

MODERATOR_PROMPT = """The mention and all candidate evidence:
{sample}
Candidates and their facts:
{evidence}

Debate transcript (multiple rounds):
{debate_log}

As the moderator, based on the facts, select from the candidates (entity_id in {{{finalist_ids}}}) the one the mention truly refers to.
Important: same name with a different generation (daime / 〈n〉) is a DIFFERENT person — use the year to align to the correct generation.
Output strictly as JSON, no markdown: {{"reasoning": "brief reasoning", "confidence": 0.0-1.0, "entity_id": INTEGER}}"""

# Adversarial verification of the moderator's pick (lever 2).
VERIFY_PROMPT = """The moderator's preliminary choice is entity_id={pick}.
Mention and candidate evidence:
{sample}
{evidence}

You are a skeptic. Try your best to **refute** this choice: does the year really fall within its active period? Does the kaimeihyou name really match? Is there another candidate (entity_id in {{{finalist_ids}}}) that fits the evidence better (especially is the generation / daime mistaken)?
Output strictly as JSON: {{"challenge": "your objection", "better_candidate_id": INTEGER or null}}"""

REVISE_PROMPT = """The skeptic objects:
{challenge}

Re-examine your decision. If the objection holds, switch to the candidate that better fits the evidence; if not, keep your original choice.
Output strictly as JSON, no markdown: {{"reasoning": "brief reasoning", "confidence": 0.0-1.0, "entity_id": INTEGER}}"""


# The chat-with-retries helper now lives in llm_client; re-exported as _ask so the
# reasoning methods (cot_family / sampling / maps_rerank / reflexion) keep importing it
# from here unchanged.
from llm_client import ask as _ask  # noqa: E402


def parse_json(text):
    """Best-effort JSON object from an LLM reply. Tries the outermost {..} span
    first, then a minimal {.."entity_id"..} block. Returns a dict or None.
    (Previously undefined -> the parse branch below silently fell through to the
    regex, so confidence was always None; this restores real JSON parsing.)"""
    if not text:
        return None
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        pass
    m = re.search(r'\{[^{}]*"entity_id"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _parse_pick(text, finalists, fallback):
    obj = None
    try:
        obj = parse_json(text)
    except Exception:
        pass
    pid = obj.get("entity_id") if isinstance(obj, dict) else parse_final_answer(text)
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        pid = None
    conf = obj.get("confidence") if isinstance(obj, dict) else None
    if pid not in finalists:
        pid = fallback
    return pid, conf


# -----------------------------------------------------------------------------
# TransMAD: interpreter that clarifies the Edo-period playbill + extracts co-stars
# -----------------------------------------------------------------------------

INTERPRET_META = """You are an expert reader of Edo-period Japanese kabuki playbills and actor reviews (役者評判記). The raw text is terse: mixed kana/kanji, abbreviations, theater jargon, and role-vs-actor pairings.
Your job is to clarify it for an entity-linking debate that must pick the correct generation (daime / 〈n〉) of an actor.
STRICT RULE: copy every actor name, yago (house name) and haimyou (poetic name) EXACTLY as written in the original Japanese — never translate, romanize or normalize a name. Only your prose explanation is in English."""

INTERPRET_PROMPT = """Year: {year}
Target mention: {mention}
Raw context (an Edo-period kabuki playbill / review):
{sample}

Return strict JSON, no markdown:
1. "clarification": 2-4 English sentences — what kind of record this is and what it says about the target mention's role/standing.
2. "co_stars": a JSON list of the OTHER actors appearing in the SAME passage (the target's same-playbill contemporaries). Copy each name VERBATIM in Japanese. Exclude role names and the target mention itself. These contemporaries are the key signal for which generation the mention is.
3. "notes": any year / lineage / house cue useful to separate generations.

Output: {{"clarification": "...", "co_stars": ["...", "..."], "notes": "..."}}"""


def interpret_context(client, model, sample):
    """One LLM call: clarify the archaic context + extract verbatim co-star names.
    Returns a dict {clarification, co_stars, notes} or None on parse failure."""
    md = sample.get("metadata") or {}
    msgs = [
        {"role": "system", "content": INTERPRET_META},
        {
            "role": "user",
            "content": INTERPRET_PROMPT.format(
                year=md.get("year"), mention=sample.get("mention", ""), sample=format_sample(sample)
            ),
        },
    ]
    ans = _ask(client, model, msgs)
    try:
        return json.loads(ans[ans.index("{") : ans.rindex("}") + 1])
    except Exception:
        return None


def _interpretation_block(interp):
    """Render the interpreter output as an English-prose + verbatim-Japanese block
    appended to the sample shown to every debater/moderator."""
    cs = "、".join(interp.get("co_stars") or [])
    parts = [f"\n\n[Interpretation] {interp.get('clarification', '').strip()}"]
    if cs:
        parts.append(
            f"[Co-stars in this same playbill (contemporaries — use to pin the generation)] {cs}"
        )
    if interp.get("notes"):
        parts.append(f"[Notes] {interp['notes'].strip()}")
    return "\n".join(parts)


def run_debate(
    sample,
    db,
    client,
    model,
    lead_id,
    finalists,
    rounds=2,
    judge_model=None,
    verify=True,
    translate=False,
):
    """Multi-round debate over up to K finalists (levers 2+3).

    - Negative evaluates every non-lead candidate then picks the strongest (L3).
    - `rounds` exchanges of aff/neg rebuttal before the moderator decides (L2).
    - Moderator runs on `judge_model` (default = model) (L2).
    - Adversarial verify: a skeptic challenges the pick; moderator revises (L2).
    - translate (TransMAD): prepend an interpreter pass that clarifies the archaic
      context and extracts verbatim co-star names, shown to every role.
    Returns (final_id, confidence, transcript).
    """
    judge_model = judge_model or model
    _gen = getattr(db, "generic", False)  # generic (non-daime) dataset -> generic prompts
    player_meta = PLAYER_META_GENERIC if _gen else PLAYER_META
    mod_meta = MODERATOR_META_GENERIC if _gen else MODERATOR_META
    mod_prompt = MODERATOR_PROMPT_GENERIC if _gen else MODERATOR_PROMPT
    md = sample.get("metadata") or {}
    year = md.get("year")
    evidences = [(str(eid), build_evidence(db, eid, year)) for eid in finalists]
    evidence = format_evidence_block(evidences)
    finalist_ids = ", ".join(str(e) for e in finalists)
    sample_str = format_sample(sample, db)
    if translate and not _gen:  # co-star interpretation is a daime-generation cue
        interp = interpret_context(client, model, sample)
        if interp:
            sample_str = sample_str + _interpretation_block(interp)

    aff_prompt = AFFIRMATIVE_PROMPT.format(sample=sample_str, evidence=evidence, lead_id=lead_id)
    aff_hist = [{"role": "system", "content": player_meta}, {"role": "user", "content": aff_prompt}]
    neg_hist = [{"role": "system", "content": player_meta}, {"role": "user", "content": aff_prompt}]

    debate_log = []
    aff_ans = _ask(client, model, aff_hist)
    aff_hist.append({"role": "assistant", "content": aff_ans})
    debate_log.append(f"[Affirmative R1] {aff_ans}")

    neg_hist.append({"role": "assistant", "content": aff_ans})
    neg_hist.append(
        {"role": "user", "content": NEGATIVE_PROMPT.format(aff_ans=aff_ans, lead_id=lead_id)}
    )
    neg_ans = _ask(client, model, neg_hist)
    neg_hist.append({"role": "assistant", "content": neg_ans})
    debate_log.append(f"[Negative R1] {neg_ans}")

    # Additional rebuttal rounds.
    for r in range(2, rounds + 1):
        aff_hist.append(
            {"role": "user", "content": AFF_REBUT_PROMPT.format(neg_ans=neg_ans, lead_id=lead_id)}
        )
        aff_ans = _ask(client, model, aff_hist)
        aff_hist.append({"role": "assistant", "content": aff_ans})
        debate_log.append(f"[Affirmative R{r}] {aff_ans}")

        neg_hist.append({"role": "user", "content": NEG_REBUT_PROMPT.format(aff_ans=aff_ans)})
        neg_ans = _ask(client, model, neg_hist)
        neg_hist.append({"role": "assistant", "content": neg_ans})
        debate_log.append(f"[Negative R{r}] {neg_ans}")

    # Moderator decision.
    mod_ans = _ask(
        client,
        judge_model,
        [
            {"role": "system", "content": mod_meta},
            {
                "role": "user",
                "content": mod_prompt.format(
                    sample=sample_str,
                    evidence=evidence,
                    finalist_ids=finalist_ids,
                    debate_log="\n\n".join(debate_log),
                ),
            },
        ],
    )
    final_id, conf = _parse_pick(mod_ans, finalists, lead_id)

    transcript = {"debate_log": debate_log, "moderator": mod_ans, "finalists": finalists}

    # Adversarial verification + optional revision.
    if verify:
        challenge = _ask(
            client,
            judge_model,
            [
                {"role": "system", "content": mod_meta},
                {
                    "role": "user",
                    "content": VERIFY_PROMPT.format(
                        pick=final_id,
                        sample=sample_str,
                        evidence=evidence,
                        finalist_ids=finalist_ids,
                    ),
                },
            ],
        )
        revise = _ask(
            client,
            judge_model,
            [
                {"role": "system", "content": mod_meta},
                {
                    "role": "user",
                    "content": mod_prompt.format(
                        sample=sample_str,
                        evidence=evidence,
                        finalist_ids=finalist_ids,
                        debate_log="\n\n".join(debate_log),
                    ),
                },
                {"role": "assistant", "content": mod_ans},
                {"role": "user", "content": REVISE_PROMPT.format(challenge=challenge)},
            ],
        )
        final_id, conf = _parse_pick(revise, finalists, final_id)
        transcript["challenge"] = challenge
        transcript["revise"] = revise

    return final_id, conf, transcript


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of year-ranked finalists the moderator chooses among",
    )
    ap.add_argument(
        "--entities",
        default=ENTS_PATH,
        help="Entity DB jsonl (must match the dataset variant of --test)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of aff/neg debate rounds before the moderator decides",
    )
    ap.add_argument(
        "--judge_model", default=None, help="Model for moderator/verifier (default = --model)"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Enable adversarial verification of the moderator's pick. "
        "OFF by default: on the full set it was net-negative "
        "(flipped 11 correct -> wrong vs only 2 wrong -> correct).",
    )
    ap.add_argument(
        "--translate",
        action="store_true",
        help="TransMAD: add an interpreter pass that clarifies the archaic "
        "Edo-period context and extracts verbatim co-star names, shown "
        "to every debater (helps pin the generation via contemporaries).",
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
        """Disambiguate one sample (baseline + optional debate). Independent per
        sample, so safe to run concurrently — the OpenAI client is thread-safe."""
        gold = int(sample["label_id"])

        # 1) baseline tool-use answer A
        base = disambiguate_one(sample, db, client, args.model, args.max_steps)
        a_id = base["pred_id"]

        # 2) decide single vs multi candidate
        from llm_tool_disambiguator import get_candidate_list  # noqa: E402 (lazy: avoids circular import)

        cands = get_candidate_list(
            db, sample, topk=None
        )  # CAND_METHOD env; default 'gather' == old
        is_multi = len(cands) >= 2

        # 3) debate on genuine ambiguity. Finalists come from the year-ranked
        #    pool, so debate runs even when the baseline tool loop failed (-1).
        transcript = None
        if is_multi:
            lead_id, finalists = pick_finalists(cands, a_id, k=args.k)
            if len(finalists) >= 2:
                final_id, _conf, transcript = run_debate(
                    sample,
                    db,
                    client,
                    args.model,
                    lead_id,
                    finalists,
                    rounds=args.rounds,
                    judge_model=args.judge_model,
                    verify=args.verify,
                    translate=args.translate,
                )
            else:
                final_id = a_id
        else:
            final_id = a_id

        rec = {
            "sample_idx": args.start + i,
            "mention": sample.get("mention"),
            "type": sample.get("type"),
            "gold_id": gold,
            "gold_title": sample.get("label_title", ""),
            "n_candidates": len(cands),
            "is_multi": is_multi,
            "pred_id_baseline": a_id,
            "pred_id": final_id,
            "debated": transcript is not None,
            "flipped": final_id != a_id,
            "transcript": transcript,
        }
        return rec

    base_correct = deb_correct = total = 0
    multi_total = multi_base_correct = multi_deb_correct = 0
    flips = 0  # debate changed the answer

    from concurrent.futures import ThreadPoolExecutor

    # executor.map yields results in input order, so the output JSONL stays
    # contiguous by sample_idx (keeps the resume harness's line-count logic valid)
    # while up to num_workers samples are processed concurrently.
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            base_ok = rec["pred_id_baseline"] == rec["gold_id"]
            deb_ok = rec["pred_id"] == rec["gold_id"]
            base_correct += base_ok
            deb_correct += deb_ok
            total += 1
            if rec["is_multi"]:
                multi_total += 1
                multi_base_correct += base_ok
                multi_deb_correct += deb_ok
            if rec["flipped"]:
                flips += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 5 == 0 or done + 1 == len(test):
                el = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] "
                    f"base={base_correct}/{total}={100 * base_correct / total:.1f}% "
                    f"debate={deb_correct}/{total}={100 * deb_correct / total:.1f}% "
                    f"(multi={multi_total}, flips={flips}) elapsed={el:.0f}s"
                )

    out_f.close()
    print("\n[done]")
    print(f"  overall baseline: {base_correct}/{total} = {100 * base_correct / total:.1f}%")
    print(f"  overall debate:   {deb_correct}/{total} = {100 * deb_correct / total:.1f}%")
    if multi_total:
        print(
            f"  multi-cand baseline: {multi_base_correct}/{multi_total} = "
            f"{100 * multi_base_correct / multi_total:.1f}%"
        )
        print(
            f"  multi-cand debate:   {multi_deb_correct}/{multi_total} = "
            f"{100 * multi_deb_correct / multi_total:.1f}%"
        )
    print(f"  debate flipped {flips} answers")


if __name__ == "__main__":
    main()
