#!/usr/bin/env python3
"""Tool-use LLM disambiguator for kabuki actor entity linking.

Architecture (2C from the plan):
  - LLM (GPT-4o / GPT-5) calls deterministic tools to look up entity DB facts
    (search by surface, kaimeihyou periods, family members, year-active info)
  - The LLM combines tool outputs with mention context to pick the right
    entity. NOT a fine-tuned model; pure inference-time reasoning.

The tools expose entity-database knowledge (kaimeihyou, birth/death). They
do NOT reveal gold labels — same info is available to any retrieval-style
system, so this is fair (no leakage).

Usage:
  python llm_tool_disambiguator.py \
      --test /workspace/BLINK/dataset_relinked_yr/test.jsonl \
      --out /workspace/BLINK/predictions_tooldis/gpt4o.jsonl \
      --model gpt-4o \
      --limit 100   # set to None for full
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
import difflib
import glob
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

import config  # noqa: E402  (config.WS = workspace root, YAKUSYA_WORKSPACE-overridable)

ENTS_PATH = f"{config.WS}/BLINK/dataset_relinked/entities.jsonl"

# -----------------------------------------------------------------------------
# Tool implementations (no LLM here, just DB lookup)
# -----------------------------------------------------------------------------


def _normalize(name: str) -> str:
    s = re.sub(r"\s+", "", name or "")
    s = re.sub(r"[☆★○●◎◇◆□■▽▼△▲]", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"（[^）]*）", "", s)
    s = re.sub(r"〈[^〉]*〉", "", s)
    return s.strip()


def _fold_diacritics(s: str) -> str:
    """NFKD decompose + drop combining marks: é→e, ä→a, ö→o, ü→u, å→a, ç→c…"""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalize_generic(name: str, lang: str = "") -> str:
    """Language-aware Latin-script normalizer for generic (Wikidata) KBs. Always
    lowercases, collapses (not deletes) whitespace, and strips surrounding
    punctuation; then applies per-language folding, because the right rule differs
    by language:

      de -> ß becomes ss, then fold umlauts (ä→a ö→o ü→u)
      fr -> fold accents (é/è/ê→e, ç→c, à→a …)
      sv -> fold å/ä/ö (Swedish sorts them last but variants/OCR conflate them)
      fi -> KEEP ä/ö (distinct Finnish letters, folding would merge real words)
      sa -> Sanskrit in SLP1: KEEP case (capitals are distinct phonemes, e.g.
            A≠a long/short vowel, D≠d retroflex) and do NOT fold — lowercasing
            would collapse different Sanskrit sounds into one surface.
      en / other -> fold strays (harmless on ASCII, catches loanword accents)

    The daime `_normalize` deletes all spaces and keeps case, which mangles Latin
    titles/aliases and breaks surface/alias matching on the HIPE benchmarks."""
    lg = (lang or "").lower()[:2]
    s = name or ""
    if lg != "sa":  # SLP1 is case-significant; every other language folds case
        s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(".,;:!?'\"()[]{}«»")
    if lg == "de":
        s = _fold_diacritics(s.replace("ß", "ss"))
    elif lg in ("fi", "sa"):
        pass  # fi: keep ä/ö;  sa: keep case + SLP1 letters as-is
    else:  # fr, sv, en, unknown
        s = _fold_diacritics(s)
    return s


class EntityDB:
    def __init__(self, ents):
        self.ents = ents
        self.by_id = {int(e["numeric_id"]): e for e in ents}
        # daime task iff the KB has 改名表; otherwise a generic multi-candidate EL
        # benchmark -> callers pick the generic prompts (no year/代目 reasoning).
        # EL_FORCE_GENERIC=1 forces generic mode on a daime KB (agent-ablation).
        self.generic = bool(os.environ.get("EL_FORCE_GENERIC")) or is_generic(ents)
        # Dominant KB language picks the per-language folding rule (de/fr/sv/fi/en).
        from collections import Counter as _Counter

        langs = _Counter((e.get("metadata", {}) or {}).get("language", "") for e in ents)
        self.lang = (langs.most_common(1)[0][0] if langs else "") or ""
        # Dataset kind selects the per-dataset evidence organizer (see el_mode):
        #   daime    — KB has a 改名表 (kaimeihyou)
        #   mahanama — KB ships name-variant clusters (metadata.cluster_id)
        #   generic  — plain Wikidata-QID KB (HIPE benchmarks)
        _has_cluster = any((e.get("metadata") or {}).get("cluster_id") is not None for e in ents)
        self.kind = "daime" if not is_generic(ents) else "mahanama" if _has_cluster else "generic"
        # The evidence agent for this KB. It lives on the db because every caller
        # that needs a candidate's evidence already has one, and because the two
        # halves of the method — the sheet and the query key stated against it —
        # have to come from the same place or they drift apart.
        #
        # `kind` is passed rather than re-detected: EL_FORCE_GENERIC is the ablation
        # that runs the daime KB through the generic organizer to measure what its
        # own 改名表 is worth, and nothing in the data marks that run as different.
        _kind = "generic" if (self.generic and self.kind == "daime") else self.kind
        self.agent = EvidenceAgent(ents, kind=_kind)
        self.type_informative = self.agent.type_informative
        # Latin-script, language-aware normalizer for generic KBs; daime keeps the
        # JA normalizer (deletes spaces, keeps case, strips 〈n〉/parens).
        self.norm = (lambda s: _normalize_generic(s, self.lang)) if self.generic else _normalize
        # title (normalized) → list of entity_ids
        self.by_title_norm: dict[str, list[int]] = defaultdict(list)
        # alias (kaimeihyou name / Wikidata alias) normalized → list of entity_ids
        self.by_alias_norm: dict[str, list[int]] = defaultdict(list)
        self._n_kaimeihyou = 0
        for e in ents:
            self.by_title_norm[self.norm(e["title"])].append(int(e["numeric_id"]))
            md = e.get("metadata", {}) or {}
            kh = md.get("kaimeihyou") or []
            self._n_kaimeihyou += len(kh)
            for p in kh:
                n = self.norm(p.get("name", ""))
                if n:
                    self.by_alias_norm[n].append(int(e["numeric_id"]))
            # generic (non-daime) KBs carry Wikidata aliases instead of kaimeihyou
            for a in md.get("aliases") or []:
                n = self.norm(a)
                if n:
                    self.by_alias_norm[n].append(int(e["numeric_id"]))
        # family base → list of entity_ids
        self.by_base: dict[str, list[int]] = defaultdict(list)
        gen_pat = re.compile(r"〈(\d+)〉")
        for e in ents:
            t = e["title"]
            b = re.sub(r"\([^)]*\)$", "", gen_pat.sub("", t)).strip()
            self.by_base[b].append(int(e["numeric_id"]))

    def candidates(self, sample, method=None, topk=30):
        """Candidate entity_ids for one mention `sample`, via the named generation
        method (see get_candidate_list for the strategy list). The query-side entry
        point: build an EntityDB from a dataset's entities, then ask it for a
        mention's pool — ``db.candidates(sample, method="gather")``."""
        return get_candidate_list(self, sample, method=method, topk=topk)


def tool_search_entities(db: EntityDB, mention_surface: str) -> list[dict]:
    """Return candidates whose title (normalized) matches mention, plus
    entities for which mention is one of their historical kaimeihyou names."""
    mn = db.norm(mention_surface) if hasattr(db, "norm") else _normalize(mention_surface)
    by_title = db.by_title_norm.get(mn, [])
    by_alias = db.by_alias_norm.get(mn, [])
    extra = []
    if getattr(db, "generic", False) and len(mn) >= 2:
        # Generic mode: exact key-match rarely fires (OCR-garbled / variant
        # surfaces), so substring-match the mention against BOTH titles and
        # Wikidata aliases. This is the generic analog of daime's kaimeihyou-at-
        # year candidate gathering and is what lifts gold-in-candidates recall.
        for k, ids in db.by_title_norm.items():
            if k != mn and (mn in k or (len(k) >= 3 and k in mn)):
                extra.extend(ids)
        for k, ids in db.by_alias_norm.items():
            if k != mn and (mn in k or (len(k) >= 3 and k in mn)):
                extra.extend(ids)
        extra = extra[:30]
    elif not by_title and not by_alias and len(mn) >= 2:
        # daime: original title-only prefix/substring fallback (unchanged).
        for k, ids in db.by_title_norm.items():
            if k.startswith(mn) or mn in k:
                extra.extend(ids)
        extra = extra[:20]
    seen, out = set(), []
    for eid in by_title + by_alias + extra:
        if eid in seen:
            continue
        seen.add(eid)
        e = db.by_id[eid]
        out.append(
            {
                "entity_id": eid,
                "title": e["title"],
                "category": e.get("metadata", {}).get("category", ""),
            }
        )
    return out[:30]


def tool_find_by_alias_year(db: EntityDB, alias: str, year: int) -> list[dict]:
    """Entities for which `alias` is a kaimeihyou name AT `year`."""
    mn = _normalize(alias)
    out = []
    for eid in db.by_alias_norm.get(mn, []):
        e = db.by_id[eid]
        for p in e.get("metadata", {}).get("kaimeihyou") or []:
            if _normalize(p.get("name", "")) != mn:
                continue
            s, ed = p.get("start_year"), p.get("end_year")
            if s and ed and s <= year <= ed:
                out.append(
                    {
                        "entity_id": eid,
                        "title": e["title"],
                        "period": f"{s}-{ed}",
                    }
                )
                break
    return out[:30]


def tool_get_entity(db: EntityDB, entity_id: int) -> dict:
    e = db.by_id.get(int(entity_id))
    if not e:
        return {"error": f"no entity with id {entity_id}"}
    md = e.get("metadata", {}) or {}
    return {
        "entity_id": int(e["numeric_id"]),
        "title": e["title"],
        "category": md.get("category"),
        "birth_year": md.get("birth_year"),
        "death_year": md.get("death_year"),
        "daime": md.get("daime"),
        "reading": md.get("reading"),
        "description": (e.get("text") or "")[:600],
    }


def tool_year_active(db: EntityDB, entity_id: int) -> dict:
    e = db.by_id.get(int(entity_id))
    if not e:
        return {"error": f"no entity with id {entity_id}"}
    md = e.get("metadata", {}) or {}
    periods = []
    for p in md.get("kaimeihyou") or []:
        if p.get("start_year") or p.get("end_year"):
            periods.append(
                {
                    "name": p.get("name"),
                    "start_year": p.get("start_year"),
                    "end_year": p.get("end_year"),
                    "daime": p.get("daime"),
                }
            )
    return {
        "entity_id": int(e["numeric_id"]),
        "title": e["title"],
        "birth_year": md.get("birth_year"),
        "death_year": md.get("death_year"),
        "name_periods": periods,
    }


def tool_kaimeihyou_at_year(db: EntityDB, entity_id: int, year: int) -> dict:
    e = db.by_id.get(int(entity_id))
    if not e:
        return {"error": f"no entity with id {entity_id}"}
    md = e.get("metadata", {}) or {}
    for p in md.get("kaimeihyou") or []:
        s, ed = p.get("start_year"), p.get("end_year")
        if s and ed and s <= year <= ed:
            return {"name_at_year": p.get("name"), "period": f"{s}-{ed}", "daime": p.get("daime")}
    return {"name_at_year": None, "note": "no kaimeihyou period covers this year"}


def tool_find_family(db: EntityDB, base_name: str) -> list[dict]:
    bn = base_name.strip()
    ids = db.by_base.get(bn, [])
    if not ids:
        # try normalization (strip parens / spaces)
        nm = _normalize(base_name)
        for k, v in db.by_base.items():
            if _normalize(k) == nm:
                ids = v
                break
    out = []
    for eid in ids[:30]:
        e = db.by_id[eid]
        md = e.get("metadata", {}) or {}
        out.append(
            {
                "entity_id": eid,
                "title": e["title"],
                "daime": md.get("daime"),
                "birth_year": md.get("birth_year"),
                "death_year": md.get("death_year"),
            }
        )
    return out


# -----------------------------------------------------------------------------
# Tool registry for OpenAI function calling
# -----------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": "Find candidate entities by mention surface form. Returns up to 30 candidates whose canonical title matches the mention, or who have used the mention as a historical stage name (kaimeihyou).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mention_surface": {
                        "type": "string",
                        "description": "The mention as it appears in the text",
                    }
                },
                "required": ["mention_surface"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_by_alias_year",
            "description": "Find entities that used `alias` as a kaimeihyou stage name during a period containing `year`. Best disambiguator for historical mentions.",
            "parameters": {
                "type": "object",
                "properties": {"alias": {"type": "string"}, "year": {"type": "integer"}},
                "required": ["alias", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity",
            "description": "Get full record for an entity: title, category, birth/death year, description (up to 600 chars).",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "integer"}},
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "year_active",
            "description": "Get all stage-name periods (kaimeihyou) and birth/death years for an entity. Use this to check whether a year falls within when the entity was active.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "integer"}},
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kaimeihyou_at_year",
            "description": "What stage name (kaimeihyou) did entity_id use at year? Returns null if none of its known periods cover that year.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "integer"}, "year": {"type": "integer"}},
                "required": ["entity_id", "year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_family",
            "description": "Find all generations/sub-lineages sharing the same family base name (e.g. '松本幸四郎' returns 〈1〉 through 〈10〉 plus sub-lineages).",
            "parameters": {
                "type": "object",
                "properties": {"base_name": {"type": "string"}},
                "required": ["base_name"],
            },
        },
    },
]


def tool_search_entities_dense(db: EntityDB, mention_surface: str) -> list[dict]:
    """search_entities as surface-match FIRST, dense (BGE/E5) retrieval appended.

    Surface hits lead so the agent keeps its "the first result is the literal match"
    signal (pure dense broke that — it buries the exact match under semantic near-
    misses); dense only fills the tail to recover mentions surface can't match (OCR /
    variant surfaces). Same return shape; enabled by TOOL_SEARCH_DENSE=1. Uses the
    shared dense route (below) so the KB is encoded once, not per tool call."""
    if not hasattr(db, "entities"):
        db.entities = list(db.by_id.values())
    surface = tool_search_entities(db, mention_surface)  # exact/alias/substring first
    have = {c["entity_id"] for c in surface}
    out = list(surface)
    for eid in _dense(db, {"mention": mention_surface}, topk=30):
        if eid in have:
            continue
        e = db.by_id.get(eid)
        if e:
            out.append(
                {
                    "entity_id": eid,
                    "title": e["title"],
                    "category": e.get("metadata", {}).get("category", ""),
                }
            )
    return out[:30]


def _search_tool(db, args):
    mention = args["mention_surface"]
    if os.environ.get("TOOL_SEARCH_DENSE") == "1":
        return tool_search_entities_dense(db, mention)
    return tool_search_entities(db, mention)


TOOL_DISPATCH = {
    "search_entities": lambda db, args: _search_tool(db, args),
    "find_by_alias_year": lambda db, args: tool_find_by_alias_year(
        db, args["alias"], int(args["year"])
    ),
    "get_entity": lambda db, args: tool_get_entity(db, int(args["entity_id"])),
    "year_active": lambda db, args: tool_year_active(db, int(args["entity_id"])),
    "kaimeihyou_at_year": lambda db, args: tool_kaimeihyou_at_year(
        db, int(args["entity_id"]), int(args["year"])
    ),
    "find_family": lambda db, args: tool_find_family(db, args["base_name"]),
}

# Fair-mode tools: hide all structured year-period lookups. LLM can read entity
# description text (natural-language bio) and family structure but NOT the
# structured kaimeihyou table (which is what the relinking algorithm used to
# derive gold labels). Birth/death years are still returned via get_entity as
# they're standard biographical facts.
FAIR_TOOL_NAMES = {"search_entities", "get_entity", "find_family"}


# -----------------------------------------------------------------------------
# Agent loop
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert on entity linking for Japanese kabuki hyoubanki (actor
critique books). Given a mention, its year and its context, find the matching entity_id.

What makes kabuki hard:
- One name (e.g. "松本幸四郎") is used by many generations in turn — use `year` to tell the 代目 apart
- An actor renames himself several times in a life (kaimeihyou records the name he held in each period)
- A mention may be a short form ("松本", "幸四郎") or a haimyou (poetry name, e.g. "素朝", "飛雀")

Workflow:
1. Use search_entities or find_by_alias_year to find candidates
2. Use year_active / kaimeihyou_at_year to check whether a candidate was alive in the given
   year and whether he held that name then
3. If needed, use get_entity and read the description for lineage / yagou / performances
4. Reason from the facts the tools return — do not guess

When you have gathered everything, output a JSON object (no markdown code block):
{"reasoning": "your brief reasoning", "entity_id": INTEGER}

**Never output "cannot determine" or "not found".** If the tools returned any candidate at all
(from search_entities / find_family / find_by_alias_year), **you must pick the single most likely
one and output its entity_id**, even if you are not fully certain. For 99% of kabuki mentions at
least one reasonable candidate exists.

Selection priority:
1. An entity hit by find_by_alias_year (year matches)
2. A family member from find_family (estimate the 代目 from year)
3. The first entity returned by search_entities (closest literal match)
4. A member of the same lineage hinted at by co-performing actors in the context

If the tools return no candidate at all, retry search_entities with a different fragment of the
mention (e.g. drop the family name and use the given name alone).
"""

# Generic (non-daime) prompts + detector live in el_mode (single source of the
# dual-mode contract). Imported here for the agent loop's mode switch.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.append(_SRC)
from el_mode import GENERIC_SYSTEM_PROMPT, is_generic  # noqa: E402,F401

from evidence_agent import (  # noqa: E402
    EvidenceAgent,
    doc_year,
    mention_type_phrase,
)


def format_sample(sample: dict, db=None) -> str:
    """The query side of the prompt: the mention, and the keys its candidates'
    evidence is to be matched against.

    `Year` reads the corpus's own field via doc_year — HIPE stores `date`, not
    `year`, so the old `md.get('year')` printed `?` on all twelve of those corpora
    and left every candidate life span paired with nothing. `Type` likewise takes
    the finest tag annotated, since a coarse `LOC` cannot separate candidates that
    are all places.

    `db` is optional only for callers that have no EntityDB to hand; without it the
    key lines are omitted and the evidence goes back to being inert."""
    md = sample.get("metadata") or {}
    cl = (sample.get("context_left") or "")[-200:]
    cr = (sample.get("context_right") or "")[:200]
    # EL_NO_QUERY_KEYS restores the pre-agent query side wholesale: `metadata.year`
    # (absent on every HIPE corpus, which stores `date`) and the coarse NE tag. The
    # ablation has to cover these, not just the restated key lines — the system
    # prompt already says to prefer a matching type, so suppressing the restatement
    # alone changes nothing, and an A/B on it measures redundancy, not the method.
    off = bool(os.environ.get("EL_NO_QUERY_KEYS"))
    yr = md.get("year") if off else doc_year(sample)
    ty = (
        sample.get("type", "?") if off else (mention_type_phrase(sample) or sample.get("type", "?"))
    )
    head = (
        f"Mention: 「{sample['mention']}」\n"
        f"Year:    {yr if yr else '?'}\n"
        f"Book:    {md.get('book_name', '?')}\n"
        f"Type:    {ty}\n"
    )
    if db is not None:
        keys = db.agent.query_key_lines(sample)
        if keys:
            head += keys + "\n"
    return head + f"\nContext_left: ...{cl}\nContext_right: {cr}...\n"


_TOOL_TOKEN_RE = re.compile(r"<[｜|][^<>]+[｜|]>")

# Tool name allowed in TOOL_DISPATCH
_TOOL_NAMES = {
    "search_entities",
    "find_by_alias_year",
    "get_entity",
    "year_active",
    "kaimeihyou_at_year",
    "find_family",
}


def parse_raw_tool_calls(text: str):
    """Parse raw-format tool calls from assistant content (DeepSeek/Ollama style).
    Robust to multiple variants:
      <｜tool call begin｜>name<｜tool sep｜>{json}<｜tool call end｜>
      <｜tool_call_begin｜>name<｜tool_sep｜>{json}<｜tool_call_end｜>
      Strange nested begin-only variants seen with V3.1 (no proper sep tag).
    Strategy: strip all <｜...｜> tokens to get the raw stream, then match
    tool_name followed by a JSON object."""
    out = []
    if not text or ("<｜" not in text and "<|" not in text):
        return out
    # Replace all tool-tag tokens with a single delimiter
    stripped = _TOOL_TOKEN_RE.sub("§", text)
    # Now look for: § name § {json} (with optional § between)
    # Function-name + JSON object pairs
    pat = re.compile(r"([A-Za-z_][\w]*)\s*§+\s*(\{(?:[^{}]|\{[^{}]*\})*\})")
    for m in pat.finditer(stripped):
        name = m.group(1)
        if name not in _TOOL_NAMES:
            continue
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        out.append((name, args))
    return out


def parse_final_answer(text: str) -> int | None:
    """Extract entity_id from final JSON response."""
    # try strict JSON
    m = re.search(r'\{[^{}]*"entity_id"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return int(obj.get("entity_id", -1))
        except Exception:
            pass
    # fallback: search for "entity_id": N
    m = re.search(r'"entity_id"\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    m = re.search(r"entity_id[=\s]+(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def disambiguate_one(sample, db, client, model, max_steps=8, fair=False) -> dict:
    """Run the agent loop for one sample. Returns dict with pred_id + meta.
    fair=True removes the 3 tools that directly leak kaimeihyou structured data."""
    messages = [
        {
            "role": "system",
            "content": GENERIC_SYSTEM_PROMPT
            if getattr(db, "generic", False)
            else SYSTEM_PROMPT,
        },
        {"role": "user", "content": format_sample(sample, db)},
    ]
    tools_to_use = (
        [t for t in TOOL_DEFS if t["function"]["name"] in FAIR_TOOL_NAMES] if fair else TOOL_DEFS
    )
    n_tool_calls = 0
    for step in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_to_use,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as e:
            return {"pred_id": -1, "error": f"api: {e}", "steps": step, "tool_calls": n_tool_calls}

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                fn = TOOL_DISPATCH.get(name)
                if fn is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = fn(db, args)
                    except Exception as e:
                        result = {"error": f"tool {name}: {e}"}
                n_tool_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue
        # No structured tool_calls — check for raw-text tool calls
        # (DeepSeek / some Ollama servers return <｜tool call begin｜>... in content)
        text = msg.content or ""
        raw_calls = parse_raw_tool_calls(text)
        if raw_calls:
            # Append assistant text + each tool result as plain user messages.
            messages.append({"role": "assistant", "content": text})
            for name, args in raw_calls:
                fn = TOOL_DISPATCH.get(name)
                if fn is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = fn(db, args)
                    except Exception as e:
                        result = {"error": f"tool {name}: {e}"}
                n_tool_calls += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool `{name}` returned:\n{json.dumps(result, ensure_ascii=False)}\n\n"
                            + "Continue reasoning, or give your final answer (strict JSON)."
                        ),
                    }
                )
            continue
        # final answer
        pid = parse_final_answer(text)
        return {
            "pred_id": pid if pid is not None else -1,
            "answer_text": text,
            "steps": step + 1,
            "tool_calls": n_tool_calls,
        }

    return {
        "pred_id": -1,
        "error": "max_steps_exceeded",
        "steps": max_steps,
        "tool_calls": n_tool_calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument(
        "--entities",
        default=None,
        help="entity DB jsonl (defaults to ENTS_PATH; pass to match the "
        "--test dataset variant, e.g. dataset_relinked_yr_v2)",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="concurrent samples (remote LLM API calls run in parallel)",
    )
    ap.add_argument(
        "--fair",
        action="store_true",
        help="Strip leaky tools (year_active, kaimeihyou_at_year, find_by_alias_year)",
    )
    ap.add_argument(
        "--base_url",
        default=None,
        help="OpenAI-compatible endpoint base URL (e.g. http://localhost:8000/v1 for local vLLM)",
    )
    ap.add_argument(
        "--api_key",
        default=None,
        help="API key (defaults to OPENAI_API_KEY env; use 'EMPTY' for local vLLM)",
    )
    args = ap.parse_args()

    from llm_client import make_client

    client = make_client(base_url=args.base_url, api_key=args.api_key)

    ents = [json.loads(l) for l in open(args.entities or ENTS_PATH)]
    db = EntityDB(ents)
    test = [json.loads(l) for l in open(args.test)]
    if args.limit:
        test = test[args.start : args.start + args.limit]
    print(
        f"[start] model={args.model} samples={len(test)} max_steps={args.max_steps} "
        f"fair={args.fair} workers={args.num_workers}"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()
    correct, total = 0, 0
    total_calls = 0

    def process_sample(i, sample):
        """Disambiguate one sample (tool-use loop). Independent per sample, so
        safe to run concurrently — the OpenAI client is thread-safe."""
        r = disambiguate_one(sample, db, client, args.model, args.max_steps, fair=args.fair)
        return {
            "sample_idx": args.start + i,
            "mention": sample.get("mention"),
            "type": sample.get("type"),
            "gold_id": int(sample["label_id"]),
            "gold_title": sample.get("label_title", ""),
            "pred_id": r["pred_id"],
            "steps": r.get("steps"),
            "tool_calls": r.get("tool_calls"),
            "answer_text": r.get("answer_text", "")[:500],
            "error": r.get("error"),
        }

    from concurrent.futures import ThreadPoolExecutor

    # executor.map keeps output in input order (contiguous sample_idx) while up to
    # num_workers samples run concurrently.
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        for done, rec in enumerate(ex.map(lambda a: process_sample(*a), list(enumerate(test)))):
            if rec["pred_id"] == rec["gold_id"]:
                correct += 1
            total += 1
            total_calls += rec["tool_calls"] or 0
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if (done + 1) % 10 == 0 or done + 1 == len(test):
                elapsed = time.time() - t0
                print(
                    f"  [{done + 1}/{len(test)}] acc={correct}/{total}={100 * correct / total:.1f}% "
                    f"avg_tool_calls={total_calls / total:.1f} elapsed={elapsed:.0f}s"
                )
    out_f.close()
    print(f"\n[done] final acc = {correct}/{total} = {100 * correct / total:.1f}%")


# ============================================================================
# Candidate-list provider (folded in from the former methods/core/candidate_list.py).
#
# One entry point, get_candidate_list(db, sample, method, topk) — also reachable as
# EntityDB.candidates(sample, method) — dispatches to every candidate-generation
# strategy so a single switch (CAND_METHOD env) changes what every disambiguator and
# every chatel exporter retrieves. Each generator takes (db, sample, topk) and returns
# entity_ids (deduped, order = its own ranking):
#
#   gather   surface + alias-at-year (kaimeihyou) + family + year rerank — the default,
#            daime-aware, ~100% recall on daime.
#   surface  tool_search_entities: normalized title/alias exact + substring match.
#   name     base-name pool: every generation sharing the mention's base name.
#   fuzzy    character n-gram / edit-distance over normalized titles+aliases — OCR /
#            spelling variants that dense/exact retrieval misses.
#   blink    the dataset's frozen BLINK chatel top-k (no GPU; [] if absent).
#   dense    BGE/E5 dense retrieval — stronger multilingual than BLINK's mBERT.
#   union / union_dense  de-duped union of the cheap routes (+ dense).
#
# gather_candidates / mention_base live in llm_debate_disambiguator, which imports
# EntityDB from here, so they are imported lazily to avoid the import cycle.
# ============================================================================


def _gather(db, s, topk):
    from llm_debate_disambiguator import gather_candidates

    return [c["entity_id"] if isinstance(c, dict) else c for c in gather_candidates(db, s)][:topk]


def _surface(db, s, topk):
    return [c["entity_id"] for c in tool_search_entities(db, s.get("mention", ""))][:topk]


def _name(db, s, topk):
    """Every entity whose base name equals the mention's — all generations."""
    from llm_debate_disambiguator import mention_base

    base = mention_base(s.get("mention", ""))
    if not hasattr(db, "_base_index"):
        idx = defaultdict(list)
        for e in db.entities if hasattr(db, "entities") else []:
            idx[mention_base(e.get("title", ""))].append(int(e["numeric_id"]))
        db._base_index = idx
    return db._base_index.get(base, [])[:topk]


def _fuzzy(db, s, topk, cutoff=0.72):
    """Nearest normalized title/alias keys by character similarity — OCR/variant safe."""
    mn = db.norm(s.get("mention", "")) if hasattr(db, "norm") else s.get("mention", "").lower()
    if not mn:
        return []
    if not hasattr(db, "_fuzzy_keys"):
        db._fuzzy_keys = list(set(db.by_title_norm) | set(db.by_alias_norm))
    close = difflib.get_close_matches(mn, db._fuzzy_keys, n=topk * 2, cutoff=cutoff)
    out = []
    for k in close:
        for eid in db.by_title_norm.get(k, []) + db.by_alias_norm.get(k, []):
            if eid not in out:
                out.append(eid)
    return out[:topk]


_BLINK_CACHE = {}


def _blink(db, s, topk):
    """Read this dataset's frozen BLINK chatel top-k, aligned by sample index."""
    ds = getattr(db, "dataset_name", None) or os.environ.get("YAKUSYA_DATASET", "")
    if ds not in _BLINK_CACHE:
        hits = glob.glob(f"{config.PROJ_EXP}/experiments_{ds}/*chatel*.json")
        _BLINK_CACHE[ds] = json.load(open(hits[0])) if hits else None
    chat = _BLINK_CACHE[ds]
    if chat is None:
        return []
    i = s.get("_idx")
    rec = chat.get(f"yakusya_test_{i}") if i is not None else None
    titles = ((rec or {}).get("entities") or {}).get("entity_candidates") or [[]]
    titles = titles[0] if titles else []
    t2id = getattr(db, "_title_index", None)
    if t2id is None:
        t2id = defaultdict(list)
        for e in db.entities if hasattr(db, "entities") else []:
            t2id[e.get("title", "")].append(int(e["numeric_id"]))
        db._title_index = t2id
    out = []
    for t in titles[:topk]:
        for eid in t2id.get(t, []):
            if eid not in out:
                out.append(eid)
    return out


_DENSE = {}  # model_name -> (SentenceTransformer, entity_ids, entity_matrix) per db


def _dense(db, s, topk, model_name=None):
    """BGE/E5 dense retrieval: nearest entities to the mention by embedding cosine.

    Entity library (title + description) is encoded once per db and cached; each
    mention is then a single encode + top-k nearest neighbour. Stronger multilingual
    than BLINK's mBERT bi-encoder — meant for the low-recall OCR/multilingual corpora
    where exact/alias match fails."""
    import numpy as np

    model_name = model_name or os.environ.get("DENSE_MODEL", "BAAI/bge-m3")
    # E5 models require "query:" / "passage:" prefixes; BGE does not.
    e5 = "e5" in model_name.lower()
    qpfx, dpfx = ("query: ", "passage: ") if e5 else ("", "")
    key = id(db)
    if _DENSE.get(key, (None,))[0] != model_name:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cuda")
        ents = db.entities if hasattr(db, "entities") else []
        ids = [int(e["numeric_id"]) for e in ents]
        docs = [
            dpfx
            + ((e.get("title", "") or "") + " " + (e.get("text", "") or e.get("entity", "") or ""))[
                :512
            ]
            for e in ents
        ]
        mat = model.encode(
            docs,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        _DENSE[key] = (model_name, model, ids, mat)
    _, model, ids, mat = _DENSE[key]
    q = model.encode(
        [qpfx + s.get("mention", "")], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    sims = mat @ q
    top = np.argsort(-sims)[:topk]
    return [ids[i] for i in top]


def _union_of(members):
    def f(db, s, topk):
        seen, out = set(), []
        for m in members:
            for eid in _GENERATORS[m](db, s, topk):
                if eid not in seen:
                    seen.add(eid)
                    out.append(eid)
        return out[:topk] if topk else out

    return f


_GENERATORS = {
    "gather": _gather,
    "surface": _surface,
    "name": _name,
    "fuzzy": _fuzzy,
    "blink": _blink,
    "dense": _dense,
    # union = the cheap routes (no GPU); union_dense adds BGE/E5 dense retrieval.
    "union": _union_of(("gather", "surface", "name", "fuzzy", "blink")),
    "union_dense": _union_of(("gather", "surface", "name", "fuzzy", "blink", "dense")),
}


_DISK_CACHE = {}  # (dataset, method) -> {content_key: [ids]}, loaded once


def _cache_path(db, method):
    ds = getattr(db, "dataset_name", None) or os.environ.get("YAKUSYA_DATASET", "unknown")
    d = f"{config.PROJ_EXP}/candidate_cache"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{ds}__{method}.json"


def _content_key(s):
    """A stable per-mention key from sample content — NOT the row index, so any caller
    that loads the same test.jsonl hits the same cache entry without needing _idx."""
    md = s.get("metadata") or {}
    return "|".join(
        [
            s.get("mention", ""),
            (s.get("context_left") or "")[-60:],
            (s.get("context_right") or "")[:60],
            str(md.get("year", "")),
            str(s.get("label_id", "")),
        ]
    )


def get_candidate_list(db, sample, method=None, topk=30):
    """Candidate entity_ids for `sample`, via the named generation method.

    method: one of _GENERATORS; defaults to CAND_METHOD env, else 'gather' (which
    reproduces the current pipeline exactly). topk=0 or None means uncapped.

    For GPU methods (dense/union_dense) a precomputed on-disk cache is used when
    present (CAND_CACHE=1 default), keyed by sample CONTENT — so every method process
    reuses one BGE encoding instead of each re-encoding the KB, with zero changes to
    the callers. Build the cache once with precompute_candidates()."""
    method = method or os.environ.get("CAND_METHOD", "gather")
    gen = _GENERATORS.get(method)
    if gen is None:
        raise ValueError(f"unknown candidate method {method!r}; have {list(_GENERATORS)}")
    if os.environ.get("CAND_CACHE", "1") == "1" and "dense" in method:
        key = (getattr(db, "dataset_name", ""), method)
        if key not in _DISK_CACHE:
            p = _cache_path(db, method)
            _DISK_CACHE[key] = json.load(open(p)) if os.path.exists(p) else None
        cache = _DISK_CACHE[key]
        if cache is not None:
            hit = cache.get(_content_key(sample))
            if hit is not None:
                return hit[:topk] if topk else hit
    return gen(db, sample, topk or 10**9)


def precompute_candidates(db, test, method, topk=30):
    """Run `method` over every sample once and write its ids to the disk cache."""
    out = {}
    for i, s in enumerate(test):
        s.setdefault("_idx", i)
        out[_content_key(s)] = _GENERATORS[method](db, s, topk or 10**9)
    p = _cache_path(db, method)
    json.dump(out, open(p, "w"))
    return p


if __name__ == "__main__":
    main()
