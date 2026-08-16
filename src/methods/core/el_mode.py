"""Dual-mode entity linking for the LLM disambiguator suite — one place.

The suite serves two tasks from the same code:

  * **daime** (kabuki 代目 disambiguation): the KB carries a 改名表 (`kaimeihyou`)
    per entity. Same surface name spans many generations; the decisive signal is
    the mention's injected `[年:YYYY]` token → which generation was active/using
    that name. Keep the original year/generation prompts.

  * **generic** (any other Wikidata-QID EL benchmark: ajmc / hipe2020 / newseye /
    sonar / topres19th …): no 改名表, no year signal. Standard EL — pick the entity
    the mention refers to from context + each candidate's description.

Detection is automatic: `is_generic(entities)` is true iff no entity has a
`kaimeihyou`. `EntityDB.generic` caches it; every method selects its prompt with
`<generic_prompt> if db.generic else <daime_prompt>`. yakusya data always resolves
to daime mode, so its behaviour is unchanged.

This module owns the generic side: the detector, the generic prompt strings, and
the generic candidate-evidence shape. The daime prompts stay with their methods.
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


import os as _os
import sys as _sys

_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _SRC not in _sys.path:
    _sys.path.append(_SRC)

# The evidence organizers, the leak list and the KB detectors all live in
# src/evidence_agent.py now — one implementation, verified field-for-field against
# the one this module used to carry (9298 candidates, 14 corpora, 100% identical)
# before it was removed. This module keeps only the generic PROMPTS.
from evidence_agent import (  # noqa: E402,F401
    EVIDENCE_NOTE,
    Evidence,
    EvidenceAgent,
)


def is_generic(entities):
    """True for a non-daime KB (no 改名表 anywhere) → use the generic prompts."""
    return EvidenceAgent._detect_kind(entities) != "daime"


# --- era-hint agent (generic analog of the daime playbill-year signal) ---------
# On daime the mention carries a playbill year [年:YYYY]; on generic corpora
# there is none, and the *document* date is the wrong anchor (a mention can refer
# to an ancient or timeless entity). Instead we ask an LLM to infer the referent's
# own era from context, and abstain ('') when it is undatable.
_ERA_PROMPT = (
    "Read the mention and its context. In what time period did the "
    "entity the mention refers to most likely live, exist, or occur? Judge from "
    "the context, not from any publication date. Answer with a rough year, a "
    "century, or a short range; if the referent is ancient, mythological, or has "
    'no meaningful date, answer "timeless". Mention: "{m}"\nContext: {c}\n'
    'Output strictly JSON: {{"era": "..."}}'
)


def infer_era(client, model, mention, context):
    """LLM era hint for the mention's referent; '' when timeless/undatable."""
    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=60,
            messages=[
                {"role": "user", "content": _ERA_PROMPT.format(m=mention, c=(context or "")[:600])}
            ],
        )
        txt = r.choices[0].message.content
    except Exception:
        return ""
    m = _re.search(r'"era"\s*:\s*"([^"]*)"', txt or "")
    era = (m.group(1) if m else "").strip()
    return "" if (not era or "timeless" in era.lower()) else era


def era_hint_line(era):
    """Prompt line injecting an inferred era, or '' when abstaining."""
    if not era:
        return ""
    return (
        f"\nInferred from the context, the mention's referent most likely "
        f"belongs to this period: {era}. When two candidates are otherwise "
        f"plausible, prefer the one whose dates are consistent with it."
    )


def sample_context(sample):
    """Mention + trimmed left/right context string for era inference."""
    m = sample.get("mention", "")
    return (
        f"{(sample.get('context_left') or '')[-300:]} [[{m}]] "
        f"{(sample.get('context_right') or '')[:300]}"
    )


# --- generic prompts (no year / 代目 / kaimeihyou reasoning) -------------------

# Tool-use agent (llm_tool) system prompt.
GENERIC_SYSTEM_PROMPT = """You are an entity-linking expert. Given a mention, its
surrounding context, and a knowledge base of candidate entities (each with a title
and description), find the entity_id the mention refers to.

Workflow:
1. Use search_entities to find candidates whose title or alias matches the mention
   (try the full mention and, if OCR-garbled or partial, shorter fragments).
2. Use get_entity to read each candidate's description; pick the one whose
   description and type best fit the surrounding context.
3. Reason only from the tool results — do not guess.

Output a single JSON object (no markdown):
{"reasoning": "brief reasoning", "entity_id": INTEGER}

Never output "unknown" / "not found". If search_entities returns any candidate,
choose the most likely one and output its entity_id. If none, retry search_entities
with a different fragment of the mention.
"""

# What the evidence sheet contains and how to weigh it lives with the agent that
# produces it (src/evidence_agent.py) — this module only splices it into the roles.
_EVIDENCE_NOTE = EVIDENCE_NOTE

# Single-agent reasoning methods (zscot / cove / rerank / selfcon / plansolve /
# maps / reflexion) system role.
GENERIC_META = (
    """You are an entity-linking expert. Given a mention with its surrounding context, and several candidate entities, determine which candidate the mention refers to. """
    + _EVIDENCE_NOTE
    + """ Reason only from the context and the given evidence; do not fabricate facts."""
)

# Debate roles (mad / transmad).
PLAYER_META_GENERIC = (
    """You are a debater specializing in entity linking. You are given a mention (with its surrounding context) and several candidate entities. Determine which candidate the mention refers to. """
    + _EVIDENCE_NOTE
    + """ Reason strictly from the context and the given evidence; do not fabricate facts."""
)

MODERATOR_META_GENERIC = (
    """You are the moderator of an entity-linking debate. Two debaters argue over which candidate a mention refers to. """
    + _EVIDENCE_NOTE
    + """ Based on the context and each candidate's evidence, decide which candidate is correct."""
)

MODERATOR_PROMPT_GENERIC = """The mention and all candidate evidence:
{sample}
Candidates and their facts:
{evidence}

Debate transcript (multiple rounds):
{debate_log}

As the moderator, select from the candidates (entity_id in {{{finalist_ids}}}) the one the mention truly refers to, based on the context and each candidate's description.
Output strictly as JSON, no markdown: {{"reasoning": "brief reasoning", "confidence": 0.0-1.0, "entity_id": INTEGER}}"""
