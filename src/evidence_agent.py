"""The evidence agent: a dataset's non-text annotation, and the two ways to hand it
to a model.

Every corpus here answers "who is this mention?" partly from text and partly from
something else: daime ships a 改名表 (kaimeihyou) of dated stage names, the HIPE
corpora ship Wikidata types and life spans, Mahānāma ships epithet clusters. That
"something else" is what this module packages.

`EvidenceAgent` is the entry point and offers three ways to read a KB record:

    get_evidence_all(entity_id)                    every field, values read verbatim
    get_evidence_organized(entity_id, sample)      hand-written collapse against the query
    get_evidence_organized_by_llm(entity_id)       LLM-selected evidence fields, no collapse

They form a ladder of how much judgement is applied, and where it comes from. `all`
enumerates the schema and reads every value — a corpus-agnostic dump that needs no
human to know the field names, and the arm that already captures most of the benefit
everywhere but daime. `organized_by_llm` drops the bookkeeping an LLM judged useless
but reshapes nothing. `organized` alone collapses a stored structure
against the query — `name_at_mention_year` does not exist until a year is supplied,
nor `names_in_context` until a context is. Both LLM arms decide once off the schema
and copy values verbatim; only the last needs the query, and the gap between it and
the other two is what a query key buys. Evidence is only evidence relative to a
question.

That difference is also where the value is. Across 14 corpora the two arms are
nearly tied (+5.0 vs +4.8 over a no-evidence baseline, ambiguous subsets), because
most KBs here are flat — three scalar Wikidata fields, so "organizing" them is close
to an identity function. They separate exactly where the raw annotation is a
structure the model cannot query itself: on daime's kaimeihyou table, organized is
+14.3 against the dump's +3.5. And on mahanama, whose entities carry up to 1474
epithets, the dump is a 78k-character prompt scoring -16.5 — evidence provision
reallocates the model's attention rather than adding to it, so bulk actively
displaces the description it was already reading correctly.

See experiment/RESULTS_evidence_provision.md for the numbers.

`Evidence` is a dict subclass on purpose. The corpora's raw annotation has no common
shape — a nested table, a scalar string, a 1474-entry list — so a type that tried to
hold all of them would degrade into exactly this dict anyway. It earns its keep by
owning two things a bare dict cannot:

  * **the leak list.** Some fields sitting right next to the evidence ARE the answer:
    `label_id`, and `metadata.qid` / `metadata.cluster_id`, which is what the scorer
    compares. Handing those back reads as ~100% accuracy and measures nothing. The
    filter belongs with the data, not re-derived at each call site.
  * **the rendering contract.** `render` truncates, so an overflowing field is a
    field the model never sees; ORDER puts the keyed fields first for that reason.
    `name_at_mention_year` must precede `name_periods`, or daime's decisive value is
    pushed out by the very table it was collapsed from.

Cardinality is the thing to get right. Evidence about a *candidate* (its dates, its
type, its name table) belongs to the ENTITY and is shared by every mention that
retrieves it — daime's 230 test mentions draw on 5000 entities across 3173 candidate
pairings, so attaching it per example would copy each sheet ~17.6 times. Evidence
about the *query* (the mention's year, its annotated type, the document's date)
belongs to the mention. Hence two constructors, and `index()` for the entity side.

What this module does NOT do is collapse. `name_at_mention_year` does not exist until
a year is supplied, nor `names_in_context` until a context is; those are pairings of a
query key against a candidate field, and only the caller holds both sides. Collapsing
is where the value is — on daime, handing over the raw table scores +3.5 against the
collapsed sheet's +14.3 — so it stays at query time, where the key is known.
"""

import json
import os
import re as _re

# --- the query side: what a candidate's field gets matched against ------------
# A sheet is only evidence relative to a question, so the sheet is half the work:
# something has to tell the model what to compare it with. These name the mention's
# own annotation in words the candidate's `type` can be matched to.

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
    "person": "a PERSON",
    "location": "a PLACE",
    "misc": "an ENTITY",
}

# HIPE's NE-FINE-LIT tagset. Granularity decides this key: a coarse `LOC` cannot
# separate candidates that are all places, and hipe2020_de scores -1.7 with it; its
# own `loc.adm.town` vs `loc.adm.reg` separates a town from the canton sharing its
# name, and the same corpus scores +6.3. Prefer the finest tag the corpus annotates.
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

# Corpora where the document's own date anchors the referent: a person named in an
# 1900 newspaper was alive around 1900, so the date pairs with a candidate's `dates`.
# ajmc is deliberately absent — a classical commentary printed in 1881 discusses
# Homer, so there the publication date is a trap, not a key. Read off the row's own
# `metadata.dataset` (corpus provenance), not off a directory name.
DATE_ANCHORS = frozenset({"hipe2020", "newseye", "sonar", "topres19th"})


# How to READ the sheet, for the system role of any method that is handed one. It
# belongs here because it describes this module's own output — kept next to the
# fields it names, so the two cannot drift apart.
#
# Note it already poses the type key ("prefer a candidate whose `type` matches it"),
# which is why `query_key_lines` is largely redundant for methods that carry this
# note: restating the key changed 0 of 79 predictions on hipe2020_de. The two are
# kept separate anyway — this is the standing instruction in the system role, that
# is the per-mention statement of what the key's value actually IS, and a method can
# use either, both, or neither.
EVIDENCE_NOTE = (
    "Each candidate carries organized evidence: a description, its "
    "`type`, and when available `also_known_as` name variants and `dates`. "
    "The mention's own annotated Type is given with the mention: **prefer a "
    "candidate whose `type` matches it** (a mention typed pers is a human, one "
    "typed work is a text/play/poem, loc a place, org an organisation) — this is "
    "usually the deciding signal. Then use the context; a mention may be a name "
    "variant rather than the title, so check `also_known_as`, and when the context "
    "implies a period prefer a candidate whose `dates` fit it."
)


# --- the LLM field selector (get_evidence_all) --------------------------------
# The LLM is asked which FIELDS of a structured KB carry evidence — once per corpus,
# off the schema — and the values are then read straight out of the record. It never
# sees, rewrites or summarises a value.
#
# The obvious alternative, handing it each entity's annotation to rewrite, was built
# and measured first: it scores exactly the dump (+5.2 vs +3.0 on daime, llm-all
# -0.4 across six corpora) at one call per entity. The field-level reason is the
# interesting part — asked to tidy a record it dropped `match_*` bookkeeping (26/40,
# correct) but also `kaimeihyou` (17/40) and `daime` (31/40), which are the whole
# task. Denied the query it cannot tell that a long messy name table is the answer;
# it reads as exactly the noise it was told to remove. It also silently normalised
# 松本　七蔵 to 松本 七蔵 — a rewritten name is a broken key.
#
# Selecting fields has neither failure mode: a field is kept or not, and what is kept
# is copied verbatim. This is what NOISE hand-encodes, derived instead of maintained.
_LLM_FIELDS_PROMPT = (
    "A knowledge base of entities stores these metadata fields. For each you get the "
    "field name and example values from real records.\n\n"
    "An entity-linking model must decide which candidate a mention refers to. Say "
    "which fields carry EVIDENCE for that decision — a fact about the entity itself "
    "(its dates, names, type, role, what it is) — and which are BOOKKEEPING about how "
    "the record was built, imported or matched, which say nothing about which "
    "candidate is right.\n\n"
    "Keep a field when it could help tell two same-named candidates apart. A field "
    "that looks long or messy may still be the decisive one — judge by what it means, "
    "not by how tidy it is.\n\n"
    "Reply ONLY with the evidence field names, comma-separated, no prose.\n\n"
    "Fields:\n"
)
_LLM_FIELD_SAMPLES = 3  # example values per field, enough to show what it holds
_LLM_VAL_CAP = 120  # per example value, so the schema prompt stays small
_LLM_OUT_TOK = 300


def mention_type_phrase(sample):
    """The mention's annotated type, at the finest granularity its corpus provides."""
    fine = ((sample.get("metadata") or {}).get("fine_type") or "").strip()
    if fine:
        return FINEMAP.get(fine, fine)
    mt = (sample.get("type") or "").lower()
    return TYPEMAP.get(mt, mt) if mt else ""


def doc_year(sample):
    """The document's year, where its date anchors the referent; else None.

    The metadata key is `date` ("1900-06-26") and never `year` — reading `year` on a
    HIPE row silently yields nothing, which left every temporal claim on those
    corpora paired with no key at all."""
    md = sample.get("metadata") or {}
    if isinstance(md.get("year"), int):
        return md["year"]  # daime: the playbill year
    if md.get("dataset") not in DATE_ANCHORS:
        return None
    m = _re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", str(md.get("date") or ""))
    return int(m.group(1)) if m else None  # newseye_fr ships 'NA' on ~half


class Evidence(dict):
    """One entity's or one mention's non-text annotation, leak-filtered."""

    # --- what must never be handed to a model ---------------------------------
    # The gold, under each of its names. `qid` is the gold on the HIPE corpora and
    # `cluster_id` on Mahānāma, whose scorer compares cluster membership.
    MENTION_LEAK = frozenset(
        {
            "label",
            "label_id",
            "label_title",  # the answer
            "qid",
            "cluster_id",  # the answer, in metadata
            "gold_resolved_from",
            "resolve_by",  # daime: how the gold was decided
            "orig_label_id",
            "orig_label_title",
        }
    )
    # An entity's own qid is not a leak — it says nothing about which candidate is
    # right — but cluster_id / cluster_head_eid ARE the scoring key, and revealing
    # which candidates score as equivalent is not something a method could know.
    ENTITY_LEAK = frozenset({"cluster_id", "cluster_head_eid"})
    # Carries no signal about which candidate is right; only costs context.
    # The `match_*` / `query_name` family is daime's dataset-construction audit trail
    # (how each KB entity was matched to its Ritsumeikan source record) — provenance
    # of the KB itself, not a fact about the person it describes.
    NOISE = frozenset(
        {
            "nil",
            "dataset",
            "language",
            "idx",
            "qid",
            "detail_id",
            "source",
            "wikipedia_url",
            "origin_split",
            "heldout",
            "source_file",
            "start",
            "end",
            "match_candidates_count",
            "match_flags",
            "match_flags_binary",
            "match_method",
            "match_reasons",
            "match_score",
            "query_name",
            "mentions_count",
            "collapse_fix",
            "recovered",
            "link_name",
            "disambig",
            "gold_relinked_from",
        }
    )

    # --- the rendering contract -----------------------------------------------
    # NOTE the two vocabularies. `from_entity` yields the corpus's own raw field
    # names — `aliases`, `kaimeihyou` — because that is what the KB stores. ORDER
    # names the COLLAPSED sheet — `also_known_as`, `name_at_mention_year` — because
    # that is what a query key produces. They are different stages, so `render()` is
    # empty on a raw bag by design, and `render_all()` is the one that fits it.
    # Renaming raw->collapsed is the collapse's job, and the collapse needs the key.
    #
    # ORDER itself is load-bearing: keyed fields before open-ended ones, because
    # `render` truncates and the overflow is invisible to the model.
    ORDER = (
        "type",
        "birth_year",
        "death_year",
        "name_at_mention_year",
        "names_in_context",
        "name_periods",
        "dates",
        "also_known_as",
    )
    # Rendered on the candidate's own line, not inside the evidence block.
    HEADER = ("entity_id", "title", "description")
    # `also_known_as` is a flat set of surface forms — each is another chance to
    # match, so withholding is loss. `name_periods` is the structure the collapse
    # exists to replace, so it stays an illustration: shipping all of it is what the
    # naive dump does, and on daime that is +3.5 against the organized sheet's +14.3.
    CAPS = {"also_known_as": 40, "name_periods": 6, "names_in_context": 8}

    # --- construction ---------------------------------------------------------
    @classmethod
    def from_entity(cls, e):
        """A candidate's annotation: everything in metadata that is neither the
        scoring key nor bookkeeping. Query-independent, so safe to build at load."""
        md = e.get("metadata") or {}
        ev = cls(
            entity_id=int(e["numeric_id"]),
            title=e.get("title") or "",
            description=e.get("text") or e.get("entity") or "",
        )
        for k, v in md.items():
            if k in cls.ENTITY_LEAK or k in cls.NOISE:
                continue
            if v not in (None, "", [], {}):
                ev[k] = v
        return ev

    @classmethod
    def from_mention(cls, r):
        """A query's annotation: the keys a candidate field can be paired against —
        the mention's annotated type, its playbill year, its document date. The gold
        never enters; see MENTION_LEAK."""
        md = r.get("metadata") or {}
        ev = cls()
        if r.get("type"):
            ev["type"] = r["type"]
        for k, v in md.items():
            if k in cls.MENTION_LEAK or k in cls.NOISE:
                continue
            if v not in (None, "", [], {}):
                ev[k] = v
        return ev

    @classmethod
    def index(cls, entities):
        """entity numeric_id -> Evidence. Built once per dataset load; every mention
        that retrieves an entity reads the same sheet."""
        return {int(e["numeric_id"]): cls.from_entity(e) for e in entities}

    # --- use ------------------------------------------------------------------
    def _fmt(self, k, v):
        if not isinstance(v, list):
            return str(v)
        return ", ".join(
            f"{x.get('name', '')}({x.get('start_year', '')}-{x.get('end_year', '')})"
            if isinstance(x, dict)
            else str(x)
            for x in v[: self.CAPS.get(k, 6)]
        )

    def render(self, cap=260, order=None):
        """Compact `k=v; k=v` line, keyed fields first, truncated to `cap`."""
        bits = [
            f"{k}={self._fmt(k, self[k])}"
            for k in (order or self.ORDER)
            if self.get(k) not in (None, "", [], {})
        ]
        return "; ".join(bits)[:cap]

    def render_all(self, cap=4000):
        """Every field, unordered and unselected — the naive dump, for the control
        arm. Not a fallback: on mahanama it is a 78k-char prompt scoring -16.5, since
        evidence provision reallocates the model's attention rather than adding to it.

        Fields render alphabetically unless an `_order` was attached (get_evidence_all
        sets one when an LLM ranked them); then that order wins, since the point of
        the ordering is to put the decisive field where the model looks first."""
        keys = getattr(self, "_order", None) or sorted(self)
        bits = [
            f"{k}={self._fmt(k, self[k])}"
            for k in keys
            if k not in self.HEADER and self.get(k) not in (None, "", [], {})
        ]
        return "; ".join(bits)[:cap]

    def keys_used(self):
        """Fields a query key produced — what makes this a sheet and not a dump."""
        return [
            k
            for k in ("name_at_mention_year", "names_in_context", "type", "dates")
            if self.get(k) not in (None, "", [], {})
        ]


class EvidenceAgent:
    """Holds one dataset's non-text annotation and serves it two ways.

    Built from the entity list alone. Which corpus this is, and which of its fields
    can discriminate, are DETECTED from the data rather than looked up by name —
    deliberately. `metadata.dataset` is absent on daime (0% of its entities carry
    one), so a name-keyed registry breaks on the project's own primary corpus; and
    detection is what let Mahānāma drop in with no registration at all.
    """

    def __init__(self, entities, kind=None, client=None, model=None):
        # Detect BEFORE indexing: `cluster_id` is what identifies a mahanama KB, and
        # Evidence.index strips it as the scoring key.
        # `kind` overrides detection — the ablation that runs daime through the
        # generic organizer to measure what its own evidence is worth needs to say so
        # explicitly; nothing in the data distinguishes that run from a real one.
        self.kind = kind or self._detect_kind(entities)
        self.type_informative = self._type_varies(entities)
        self.index = Evidence.index(entities)
        # `client` lets get_evidence_all derive its field list instead of inheriting
        # NOISE. Optional, and the collapse path never reads it, so every existing
        # caller keeps its behaviour. One call per agent, so a plain client is fine.
        self.client = client
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._ev_fields = None  # get_evidence_organized_by_llm: selected fields
        self._all_order = None  # get_evidence_all: LLM ordering of all fields
        self._schema = self._sample_schema(self.index)

    @staticmethod
    def _sample_schema(index):
        """field -> a few example values. What the field selector judges a field by:
        `daime=2` and `match_score=0.83` are indistinguishable by name alone."""
        schema = {}
        for ev in index.values():
            for k, v in ev.items():
                if k in Evidence.HEADER:
                    continue
                s = schema.setdefault(k, [])
                if len(s) < _LLM_FIELD_SAMPLES:
                    s.append(v)
        return schema

    # --- detection ------------------------------------------------------------
    @staticmethod
    def _detect_kind(entities):
        """daime (has a kaimeihyou) | mahanama (name-variant clusters) | generic."""
        for e in entities:
            if (e.get("metadata") or {}).get("kaimeihyou"):
                return "daime"
        for e in entities:
            if (e.get("metadata") or {}).get("cluster_id") is not None:
                return "mahanama"
        return "generic"

    @staticmethod
    def _type_varies(entities):
        """Does `type` take more than one value in this KB?

        A key is only evidence when it varies. Mahānāma's KB types every entity it
        annotates the same way for 90% of mentions (person), so asking a model to
        match on it is noise that measurably costs accuracy; AJMC's 58 distinct types
        separate Homer (human) from the Odyssey (literary work) and there it decides
        the answer. Read off the KB, never off held-out labels, so applying it per
        dataset is not tuning on the test set."""
        seen = set()
        for e in entities:
            md = e.get("metadata") or {}
            t = md.get("wd_type") or e.get("category") or e.get("type") or md.get("type")
            if t:
                seen.add(t)
                if len(seen) > 1:
                    return True
        return False

    # --- the two ways to serve it ---------------------------------------------
    def evidence_fields(self):
        """The metadata fields that carry evidence, decided ONCE off the schema.

        The LLM sees field names and example values, never a whole record, and
        answers with names only — so this costs one call per corpus and cannot touch
        a value. Without a client it falls back to NOISE, the hand-written answer to
        the same question, which is what makes the client optional rather than a
        second code path."""
        if self._ev_fields is not None:
            return self._ev_fields
        names = sorted(self._schema)
        if self.client is None or not names:
            self._ev_fields = set(names)  # index is already NOISE-filtered
            return self._ev_fields
        lines = []
        for k in names:
            vals = "; ".join(str(v)[:_LLM_VAL_CAP] for v in self._schema[k])
            lines.append(f"- {k}: {vals}")
        try:
            t = (
                self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=_LLM_OUT_TOK,
                    messages=[{"role": "user", "content": _LLM_FIELDS_PROMPT + "\n".join(lines)}],
                )
                .choices[0]
                .message.content
                or ""
            )
        except Exception:
            self._ev_fields = set(names)  # a failed call must not empty the sheet
            return self._ev_fields
        want = {w.strip().strip("`\"'") for w in _re.split(r"[,\n]", t)}
        keep = {k for k in names if k in want}
        # An answer naming nothing we asked about is a malformed answer, not a corpus
        # with no evidence. Fall back rather than hand every candidate a blank sheet.
        self._ev_fields = keep or set(names)
        return self._ev_fields

    def all_fields(self):
        """The names of every evidence field the corpus carries, enumerated ONCE.

        This is the whole "agent" side of the dump arm: point it at any dataset and
        it reports which fields exist, so the reader can copy their values. For a
        structured KB that is just the schema keys — no LLM needed, and no LLM used.
        An earlier version had the model RANK the fields; measured, the ranking put
        daime's decisive `kaimeihyou` last and bought nothing (dump ~= organized on
        13/14 corpora), so it is gone. What generalizes is the enumerate-then-read
        loop, which needs no per-corpus human knowledge of the field names."""
        if self._all_order is None:
            self._all_order = sorted(self._schema)
        return self._all_order

    def get_evidence_all(self, entity_id):
        """Every field the dataset holds for the candidate — the automated dump.

        Two parts, and neither needs a human to know the schema: `all_fields`
        enumerates the field names off the data, and the loop below reads each value
        verbatim out of the record. That makes "dump the candidate's evidence" a
        corpus-agnostic operation — the arm that already captures most of the benefit
        on 13/14 corpora (only daime needs the query-conditioned collapse on top).

        Callers render it with `.render_all()`."""
        raw = self.index.get(int(entity_id))
        if raw is None:
            return Evidence()
        order = self.all_fields()
        if self.client is None:
            return raw  # sorted-order dump, unchanged
        ev = Evidence(
            entity_id=raw.get("entity_id"),
            title=raw.get("title"),
            description=raw.get("description"),
        )
        for k in order:
            if raw.get(k) not in (None, "", [], {}):
                ev[k] = raw[k]
        ev._order = order
        return ev

    def get_evidence_organized_by_llm(self, entity_id, sample=None, pool=None):
        """The candidate's sheet, with the discriminating attribute DISCOVERED per
        mention rather than fixed off the schema.

        Two regimes:

        * ``sample`` + ``pool`` given (and a client): the query-conditioned collapse,
          GENERALIZED. daime's ``_collapse_daime`` hard-codes "year x kaimeihyou ->
          name-at-year"; here the LLM is instead shown the mention-in-context and the
          candidate briefs and asked to name the ONE attribute that separates them
          and each candidate's value on it. That discovered ``axis``/value pair is
          attached as a first-class field, so the reasoner reads a decisive line even
          on a corpus whose schema carries no structured column (HIPE's KB is just
          title+description+aliases). One discovery call per mention, cached; the
          per-candidate lookup is free.
        * no ``sample`` (or no client): the earlier schema-level field selection.

        Callers render it with ``.render`` / ``.render_all``."""
        raw = self.index.get(int(entity_id))
        if raw is None:
            return Evidence()
        if sample is not None and pool and self.client is not None:
            disc = self._discover_axis(sample, pool)
            ev = Evidence(
                entity_id=raw.get("entity_id"),
                title=raw.get("title"),
                description=(raw.get("description") or "")[:400],
            )
            axis = (disc or {}).get("axis")
            val = ((disc or {}).get("values") or {}).get(str(entity_id))
            if axis and val:
                ev[axis] = val
            v = _aliases(raw)
            if v:
                ev["also_known_as"] = v
            return ev
        keep = self.evidence_fields()
        if keep == set(self._schema):
            return raw  # nothing dropped: hand back the sheet
        ev = Evidence(
            entity_id=raw.get("entity_id"),
            title=raw.get("title"),
            description=raw.get("description"),
        )
        for k, v in raw.items():
            if k in Evidence.HEADER or k in keep:
                ev[k] = v
        return ev

    def _discover_axis(self, sample, pool):
        """One LLM call per mention: name the attribute that separates the candidates
        given the context, plus each candidate's value on it. Cached per (mention,
        context, pool). Returns {'axis': str, 'values': {id_str: value}, 'cue': str}
        or {} on failure. The LLM discovers the axis; it never picks the answer."""
        pool = [int(x) for x in pool]
        sig = (
            sample.get("mention", ""),
            (sample.get("context_left", "") or "")[-120:],
            (sample.get("context_right", "") or "")[:120],
            tuple(pool[:20]),
        )
        cache = self.__dict__.setdefault("_disc_cache", {})
        if sig in cache:
            return cache[sig]
        briefs = []
        for eid in pool[:20]:
            r = self.index.get(int(eid)) or {}
            briefs.append(f"  [{eid}] {r.get('title', '')} — {(r.get('description') or '')[:160]}")
        ctx = (
            (sample.get("context_left", "") or "")[-300:]
            + " «"
            + sample.get("mention", "")
            + "» "
            + (sample.get("context_right", "") or "")[:300]
        )
        prompt = (
            "A mention shares its surface form with several candidate entities. From the "
            "context and the candidate briefs, identify the SINGLE attribute that best "
            "separates the correct candidate from the others HERE, and give each "
            'candidate\'s value on it (verbatim from its brief, or "unknown"). Also state '
            "the phrase in the context that points to the answer. Do not pick the answer.\n\n"
            f'Mention: "{sample.get("mention", "")}" (type: {sample.get("type", "?")})\n'
            f"Context: {ctx}\n\nCandidates:\n"
            + "\n".join(briefs)
            + '\n\nJSON only: {"axis": "...", "values": {"<id>": "..."}, "cue": "..."}'
        )
        out = {}
        try:
            t = (
                self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                .choices[0]
                .message.content
                or ""
            )
            m = _re.search(r"\{.*\}", t, _re.S)
            if m:
                out = json.loads(m.group(0))
                out["values"] = {str(k): v for k, v in (out.get("values") or {}).items()}
        except Exception:
            out = {}
        cache[sig] = out
        return out

    def get_evidence_all_organized(self, entity_id, sample=None, *, year=None, context=None):
        """get_evidence_all, plus a GENERIC collapse driven by structure, not names.

        Reads every field like the dump, then — for any field shaped like a table of
        dated periods, paired with a year found in the query — surfaces the value
        active at that year as a computed field, placed first so a truncation cannot
        eat it. This is `_collapse_daime` with the corpus-specific parts (the field
        name `kaimeihyou`, the key `year`) replaced by detectors, so ONE body runs on
        every dataset: it collapses daime's kaimeihyou automatically and is a no-op on
        corpora that carry no such structure, where it equals the plain dump.

        No LLM and no client — the organizing is deterministic code; only its output
        is fed to the model. Callers render it with `.render_all()`."""
        raw = self.index.get(int(entity_id))
        if raw is None:
            return Evidence()
        query = dict(Evidence.from_mention(sample)) if sample else {}
        if year is not None:
            query["year"] = year
        if context is not None:
            query["context"] = context
        qyear = _query_year(query)
        ev = Evidence(
            entity_id=raw.get("entity_id"),
            title=raw.get("title"),
            description=raw.get("description"),
        )
        computed, passthru = [], []
        for k, v in raw.items():
            if k in Evidence.HEADER:
                continue
            if qyear is not None and _is_period_list(v):
                active = _period_at(v, qyear)
                if active is not None:
                    ck = f"{k}_at_{qyear}"
                    ev[ck] = active
                    computed.append(ck)
            ev[k] = v
            passthru.append(k)
        ev._order = computed + sorted(passthru)  # decisive computed values first
        return ev

    def get_evidence_organized(self, entity_id, sample=None, *, year=None, context=None):
        """The candidate's sheet, selected and collapsed against the query's keys.

        Pass `sample` (the mention row) and the keys are read off it; or pass `year`
        / `context` directly when the caller already extracted them. Those keys are
        what turn a stored structure into a single decisive value — without any, this
        degrades to the selection-only sheet, which is the weaker arm."""
        raw = self.index.get(int(entity_id))
        if raw is None:
            return Evidence()
        keys = Evidence.from_mention(sample) if sample else Evidence()
        if year is not None:
            keys["year"] = year
        if context is not None:
            keys["context"] = context
        ev = Evidence(
            entity_id=raw.get("entity_id"),
            title=raw.get("title"),
            description=(raw.get("description") or "")[:400],
        )
        if self.kind == "daime":
            # No type gate here: daime's `category` (役者 / 作者 / …) is a field the
            # corpus always carries, and the gate exists for KBs where the type is a
            # constant. It reads False on daime only because the detector looks for a
            # top-level `category` while daime keeps it in metadata — gating on that
            # would silently drop a field the old path always emitted.
            ev["category"] = raw.get("category")
            self._collapse_daime(raw, keys, ev)
            return ev
        if self.type_informative:
            ev["type"] = raw.get("wd_type") or raw.get("category") or raw.get("type")
        if self.kind == "mahanama":
            self._collapse_mahanama(raw, keys.get("context") or _context_of(sample), ev)
        else:
            self._collapse_generic(raw, ev)
        return ev

    def query_key_lines(self, sample):
        """The lines telling the model what to match the sheets against, or ''.

        The other half of the method. A candidate's `type` and `dates` are inert
        until something states the mention's own type and the document's year — and
        for twelve corpora nothing did, so their temporal evidence sat there as a key
        with no lock. Each line is emitted only when its key exists AND the sheets
        carry the field it pairs with.

        EL_NO_QUERY_KEYS=1 suppresses them: the ablation that leaves the evidence in
        place but takes away what it is to be compared with, which is what separates
        "the sheet is useless here" from "nothing posed the question"."""
        if os.environ.get("EL_NO_QUERY_KEYS"):
            return ""
        lines = []
        mt = mention_type_phrase(sample)
        if mt and self.type_informative:
            lines.append(
                f"The mention is annotated as {mt}. Prefer a candidate whose type matches."
            )
        yr = doc_year(sample)
        if yr and (sample.get("metadata") or {}).get("dataset") in DATE_ANCHORS:
            lines.append(
                f"It appears in a document published in {yr}. Prefer a "
                f"candidate whose `dates` are consistent with being "
                f"referred to then."
            )
        return "\n".join(lines)

    # --- per-corpus collapses -------------------------------------------------
    # Each turns ONE stored structure into ONE value by pairing it with a query key.
    # New corpus = new collapse, selected by _detect_kind — not a subclass, and not a
    # registry keyed on a dataset name the data may not carry.

    def _collapse_daime(self, raw, keys, ev):
        """kaimeihyou x playbill year -> the name that generation used that year.

        The decisive one: same surface name spans many generations, and only the year
        says which was using it. The model cannot do this itself — handed the raw
        table it scores +3.5, handed this value +14.3."""
        ev["birth_year"] = raw.get("birth_year")
        ev["death_year"] = raw.get("death_year")
        # Only the four fields that bear on "which generation, when" — the raw table
        # rows also carry readings and un-parsed era strings, which are noise here.
        periods = [
            {
                "name": p.get("name"),
                "start_year": p.get("start_year"),
                "end_year": p.get("end_year"),
                "daime": p.get("daime"),
            }
            for p in (raw.get("kaimeihyou") or [])
            if p.get("start_year") or p.get("end_year")
        ]
        year = keys.get("year")
        if isinstance(year, int):
            for p in periods:
                s, e = p["start_year"], p["end_year"]
                if s and e and s <= year <= e:
                    ev["name_at_mention_year"] = p["name"]
                    break
        ev["name_periods"] = periods or None

    def _collapse_mahanama(self, raw, ctx, ev):
        """epithet cluster x verse -> the epithets this verse actually utters.

        Whole-token and length-gated, not substring: SLP1 has no word boundaries a
        regex can trust and sandhi fuses adjacent words, so short epithets ("I",
        "aja", "pati") otherwise hit inside unrelated words and fire on every verse.
        Even done right this key is weak here (42% precision against an 86% baseline)
        — mahanama is the corpus where evidence cannot reach the model's own reading."""
        if ctx:
            toks = {t for t in _re.split(r"[\s.,;:'\"()\[\]|-]+", ctx.lower()) if t}
            hits = [a for a in _aliases(raw, cap=None) if len(a) >= 5 and a.lower() in toks]
            if hits:
                ev["names_in_context"] = hits[:8]
            return
        v = _aliases(raw)
        if v:
            ev["also_known_as"] = v

    def _collapse_generic(self, raw, ev):
        """Wikidata KB: the life span pairs with the document's date, the aliases with
        the mention's surface. Both are already scalars, so there is little to
        collapse — which is exactly why this arm barely beats the raw dump."""
        span = raw.get("span") or extract_span(raw.get("description") or "")
        if span:
            ev["dates"] = span
        v = _aliases(raw)
        if v:
            ev["also_known_as"] = v


# 8, matching the sheet the frozen results were measured on. Note this is a *list*
# cap, distinct from Evidence.CAPS, which bounds how many of them get RENDERED. The
# two are independent knobs and both bite: on sonar_de's Berlin (10 aliases) the KB
# holds more than either shows. Raising it changes nothing measurable — tested, the
# means move 0.0 — so it stays where the numbers were taken.
_ALIAS_CAP = 8


# --- generic structural organize (get_evidence_all_organized) ----------------
# The daime collapse, generalized off DATA STRUCTURE instead of field names, so one
# body runs on any corpus. It fires where the shape exists — a field that is a list
# of dated periods, paired with a year read from the query — and is inert elsewhere,
# which is why it degrades to the plain dump on the 13 corpora that carry no such
# table. Nothing here names `kaimeihyou` or `year`; both are detected.
_YEAR_RE = _re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def _as_year(v):
    """An int year in a plausible range, or one parsed out of a string, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 1000 <= v <= 2100 else None
    if isinstance(v, str):
        m = _YEAR_RE.search(v)
        return int(m.group(1)) if m else None
    return None


def _query_year(query):
    """A year from the query: an explicit `year`, else the first year any value yields."""
    y = _as_year(query.get("year"))
    if y is not None:
        return y
    for v in query.values():
        y = _as_year(v)
        if y is not None:
            return y
    return None


def _period_bounds(d):
    """(start, end) years of a period dict, from keys that name a start/end, else None."""
    start = end = None
    for k, v in d.items():
        y = _as_year(v)
        if y is None:
            continue
        kl = k.lower()
        if any(s in kl for s in ("start", "begin", "from")):
            start = y
        elif any(s in kl for s in ("end", "stop", "until", "to")):
            end = y
    return (start, end) if (start is not None or end is not None) else None


def _period_label(d):
    """The value a period bracket resolves to — its name, not its dates or reading.

    Prefers a key that IS a name/title/label/value over one that merely contains the
    word, and both over a bare non-year string, so daime's `name` wins over `reading`
    (which comes first in the dict) and over `name_type`."""
    strs = [(k, v) for k, v in d.items() if isinstance(v, str) and v and _as_year(v) is None]
    for want in ("name", "title", "label", "value"):
        for k, v in strs:
            if k.lower() == want:
                return v
    for want in ("name", "title", "label", "value"):
        for k, v in strs:
            if want in k.lower():
                return v
    return strs[0][1] if strs else None


def _is_period_list(v):
    """A list of >=2 dicts, of which >=2 carry a datable start/end — a periods table."""
    return (
        isinstance(v, list)
        and len(v) >= 2
        and all(isinstance(x, dict) for x in v)
        and sum(1 for x in v if _period_bounds(x)) >= 2
    )


def _period_at(periods, year):
    """The label of the period whose [start, end] brackets `year`, else None."""
    for p in periods:
        b = _period_bounds(p)
        if not b:
            continue
        s, e = b
        if s is not None and e is not None and s <= year <= e:
            return _period_label(p)
    return None


def _aliases(raw, cap=_ALIAS_CAP):
    """The candidate's aliases minus its own title (which is never a variant)."""
    title = (raw.get("title") or "").strip().lower()
    v = [a for a in (raw.get("aliases") or []) if a and a.strip().lower() != title]
    return v[:cap] if cap else v


def _context_of(sample):
    if not sample:
        return ""
    md = sample.get("metadata") or {}
    return (
        (sample.get("context_left") or "")[-250:] + " " + (sample.get("context_right") or "")[:250]
    ).strip() or md.get("sentence", "")


def extract_span(desc):
    """A short temporal span regexed out of a description, or '' — the fallback when
    Wikidata ships no date claim. Handles '(1813-1869)', '(b. 1813)', '(1881)',
    '16th century', 'fl. 1500', bare 4-digit years."""
    d = desc or ""
    for pat, fmt in (
        (
            r"\((?:b\.?\s*)?(\d{3,4})\s*[–-]\s*(\d{3,4})?\)",
            lambda m: f"{m.group(1)}–{m.group(2)}" if m.group(2) else f"b. {m.group(1)}",
        ),
        (r"\(\s*(\d{3,4})\s*\)", lambda m: m.group(1)),
        (r"\b(\d{1,2})(?:st|nd|rd|th)\s+century\b", lambda m: m.group(0)),
        (r"\bfl\.?\s*(\d{3,4})\b", lambda m: f"fl. {m.group(1)}"),
        (r"\b(1[0-9]{3}|20[0-2][0-9])\b", lambda m: m.group(1)),
    ):
        m = _re.search(pat, d, _re.I)
        if m:
            return fmt(m)
    return ""
