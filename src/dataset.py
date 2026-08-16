#!/usr/bin/env python3
"""Unified Dataset class: read / clean / save / split any NEL dataset.

One abstraction over the two on-disk layouts this project uses, resolved by the
config.DATASETS path registry:

  - "yakusya" 4-file layout: <dir>/{train,valid,test,entities}.jsonl, numeric_id
    gold. Supports the 4 disjoint re-splits (pair/entity/temporal/book).
  - "hipe" per-language layout: <dir>/<lang>/<split>.jsonl, Wikidata-QID gold,
    no entities.jsonl. Fixed splits; the yakusya re-splits do NOT apply.

Both expose the same record schema so downstream code is layout-agnostic:
  {mention, context_left, context_right, label_id, label_title, type, metadata}

This subsumes the old nel_models.NEL_dataset: `Dataset(data_dir)` behaves exactly
like the former `NEL_dataset(data_dir)` (same 4-file read + same SPLITS/split()),
so nel_models aliases `NEL_dataset = Dataset` and run_all is unchanged.

Examples:
    Dataset(config.DS_RELINKED_YR_V1)              # yakusya 4-file (splittable)
    Dataset.load("daime_eval_full")                # by config name
    Dataset.load("hipe2020", lang="fr", split="test")
    ds.split("pair_disjoint")                      # -> (train, valid, test)
    ds.clean(drop_no_gold=True, dedup=True); ds.save(out_dir)
"""

import html
import json
import os
import random
import re
import urllib.parse
from collections import Counter, defaultdict

# trailing "[年:..|書:..]" token injected right before the mention; kept verbatim
# when the context window is truncated (mirrors recut_daime_window.py).
_YEAR_TOKEN_RE = re.compile(r"\s*\[年:[^\]]*\]\s*$")

import config
from entity_canon import build_canon_map
from evidence_agent import Evidence

SPLIT_SEED = config.SPLIT_SEED


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def _write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_entities(data_dir):
    """The richest entities file present.

    HipeBuilder.add_wikidata_evidence writes a benchmark's enriched KB to its own
    file rather than overwriting entities.jsonl, so reading the base name drops the
    Wikidata types and life spans — i.e. most of what the HIPE corpora have to offer
    beyond the text. Prefer the enriched file wherever one was built."""
    for name in ("entities_evidence.jsonl", "entities_aliased.jsonl", "entities.jsonl"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return _read_jsonl(p)
    return []


def _detect_layout(path):
    """Infer a dataset dir's on-disk layout, so any dataset under data/dataset/
    is loadable without hardcoding:
      - 'yakusya' : has train/valid/test.jsonl at top (4-file or eval-only)
      - 'hipe'    : per-language subdirs each holding <split>.jsonl
      - 'unknown' : neither (e.g. a raw repo / CSV dir — not a loadable NEL set)"""
    if not os.path.isdir(path):
        return "unknown"
    if any(os.path.exists(os.path.join(path, f"{s}.jsonl")) for s in ("train", "valid", "test")):
        return "yakusya"
    for sub in os.listdir(path):
        subp = os.path.join(path, sub)
        if os.path.isdir(subp) and any(f.endswith(".jsonl") for f in os.listdir(subp)):
            return "hipe"
    return "unknown"


def _resolve_path(name):
    """Dataset name -> dir. config.DATASETS first, then data/dataset_<name>, then
    data/dataset/<name>, then treat `name` as a path."""
    p = config.DATASETS.get(name)
    if p and os.path.isdir(p):
        return p
    for cand in (config.dataset_data_dir(name), os.path.join(config._DDIR, name), name):
        if os.path.isdir(cand):
            return cand
    return p or name


# --- Label Studio export ------------------------------------------------------
# Only meaningful for the daime corpora: the evidence panel is built out of ARC-DB
# kaimeihyou (name-change tables) and 代目 generations. See Dataset.to_label_studio.

_LS_CONFIG = """<View>
  <Header value="役者評判記 — actor mention: pick the correct generation (代目)"/>
  <Text name="text" value="$text"/>
  <View style="margin-top:8px">
    <Text name="meta" value="year=$year  book=$book  mention=$mention"/>
  </View>

  <!-- 1) mark the actor mention span (coreference / NER layer) -->
  <Labels name="label" toName="text">
    <Label value="役者" background="#e8590c"/>
    <Label value="演目名" background="#1c7ed6"/>
    <Label value="役名" background="#2f9e44"/>
    <Label value="その他" background="#868e96"/>
  </Labels>

  <!-- 2) entity linking: for the marked span, pick the KB generation -->
  <Choices name="link" toName="text" perRegion="true" value="$candidates"
           layout="vertical" showInLine="false"
           whenTagName="label" whenLabelValue="役者"/>

  <!-- 2b) ARC DB evidence panel: kaimeihyou / birth-death / bio / source link
           per candidate, to make the generation choice easy.
           clickableLinks=true: HyperText intercepts clicks by default (for span
           annotation), which swallows the ARC DB <a> links; selectionEnabled=false
           since this panel is read-only reference, not an annotation target. -->
  <HyperText name="arcdb" value="$evidence_html" inline="true"
             clickableLinks="true" selectionEnabled="false"/>

  <!-- 3) coreference relations between spans (as in the LS template) -->
  <Relations>
    <Relation value="coreference"/>
    <Relation value="same-actor"/>
  </Relations>
</View>
"""

_ARC_BASE = "https://www.dh-jac.net/db/shumei/results-big.php"

# NIL option appended to each task: 「正解が候補一覧にない」NIL 選択肢 (out-of-KB / 該当なし)
_LS_NIL = {"value": "NIL｜該当なし（正解が候補一覧にない）"}

# inline provenance tag baked into the context, e.g. "[年:1775|書:役者酸辛甘(江)]";
# collapsed to a single space in the Label Studio display text.
_YEAR_TAG = re.compile(r"\s*\[年:[^\]]*\]\s*")


def _ls_base_name(s):
    return re.sub(r"\s+", "", re.sub(r"〈[^〉]*〉|\([^)]*\)|（[^）]*）", "", s or "")).replace(
        "　", ""
    )


def _ls_norm(s):
    return (s or "").replace("　", "").replace(" ", "")


def _ls_daime_num(e):
    m = re.search(r"〈(\d+)", e.get("title", "") or "")
    return int(m.group(1)) if m else 999


def _ls_yspan(a, b):
    return f"{a or '?'}–{b or '?'}" if (a or b) else "?"


_KAIME_MARGIN = 10  # years of slack around birth/death for a valid 名乗 period


def _ls_clean_kaimeihyou(e):
    """The 改名表 rows that actually belong to this person.

    The ARC scraper (`RitsumeiParser._extract_kaimeihyou`) collected every 〈代目〉
    row from *all* tables on a name's page, so a 名跡-line record glues later
    generations onto the earliest person — e.g. 初代坂東三津五郎 (1745–1782) ends up
    carrying 坂東八十助 1962–2000 and 坂東三津五郎〈10〉 2001–2015. When a birth/death
    year is known, drop any row whose whole span lies outside the lifespan (± a
    margin): a name a person bore must overlap the years they were alive."""
    m = e.get("metadata") or {}
    km = m.get("kaimeihyou") or []
    b, d = m.get("birth_year"), m.get("death_year")
    if b is None and d is None:
        return km
    lo = (b if b is not None else -9999) - _KAIME_MARGIN
    hi = (d if d is not None else 9999) + _KAIME_MARGIN
    out = []
    for p in km:
        ys = [y for y in (p.get("start_year"), p.get("end_year")) if y]
        if ys and (min(ys) > hi or max(ys) < lo):
            continue  # a foreign generation's row scraped onto this record
        out.append(p)
    return out


def _ls_active_window(e):
    m = e.get("metadata") or {}
    ys = []
    for p in _ls_clean_kaimeihyou(e):
        ys += [y for y in (p.get("start_year"), p.get("end_year")) if y]
    for k in ("birth_year", "death_year"):
        if m.get(k):
            ys.append(m[k])
    return (min(ys), max(ys)) if ys else None


def _ls_headline(e):
    """The person's 見出 (headline) name AND its 代目, from the kaimeihyou.

    ARC files each actor's 改名表 under their 見出 name+代目; an entity is often
    *titled* by an early alias instead (「（）千吉〈0〉」/「中村勝十郎〈1〉」 for people
    whose 見出 is 「市川門之助〈2〉」/「中村伝九郎〈2〉」). Returns (name, daime) or
    (None, None).

    When the row is not explicitly typed 見出, fall back to the kaimeihyou row whose
    (name, 代目) equals the entity's own title — the ARC link still needs a
    space-separated name + 代目 (f2/f3), which a bare title cannot supply."""
    km = (e.get("metadata") or {}).get("kaimeihyou") or []
    for p in km:
        if (p.get("name_type") or "") == "見出" and (p.get("name") or "").strip():
            return p["name"], p.get("daime")
    base, dn = _ls_base_name(e.get("title") or ""), _ls_daime_num(e)
    for p in km:
        if _ls_base_name(p.get("name") or "") == base and str(p.get("daime")) == str(dn):
            return p["name"], p.get("daime")
    return None, None


def _ls_arc_url(detail_id, headline_name=None, headline_daime=None, title=None):
    """ARC (dh-jac / Ritsumeikan) actor-DB URL landing on the actor's detail page
    (基本情報表示 / 改名表 / 襲名グラフ).

    Selecting the *right* person is the whole problem: `detail_id` is a scrape-session
    tmpid the server ignores, and a plain `f43[]=<name>` name search matches every
    record whose career passed through that name (中村伝九郎 → 22 hits) and results-2p
    then shows only the FIRST — a different actor whose 改名表 won't match the card.
    The reliable selector is the pagination form `f2[]=<見出名>&f3=<見出代目>`, which
    picks the one person by headline name + 代目 (中村　伝九郎 / 2 → 勝十郎→伝九郎→勘三郎,
    matching the card). Falls back to the f43 name search when no 見出 is known."""
    if not detail_id:
        return None
    if headline_name and headline_daime not in (None, ""):
        nm = re.sub(r"〈[^〉]*〉|\([^)]*\)|（[^）]*）", "", headline_name).strip()
        params = [
            f"f2[]={urllib.parse.quote(nm)}",
            f"f3={urllib.parse.quote(str(headline_daime))}",
            "-format=results-2p.htm",
            "-max=1",
            "enter=default",
        ]
        return f"{_ARC_BASE}?{'&'.join(params)}"
    name = headline_name or title
    if not name:
        return None
    nm = (
        re.sub(r"〈[^〉]*〉|\([^)]*\)|（[^）]*）", "", name)
        .replace("　", "")
        .replace(" ", "")
        .strip()
    )
    params = [
        f"f43[]={urllib.parse.quote(nm)}",
        "-format=results-2p.htm",
        "-max=50",
        "enter=default",
    ]
    return f"{_ARC_BASE}?{'&'.join(params)}"


def _ls_cand_label(e):
    """The dropdown line for one candidate: id｜title｜reading｜birth-death."""
    m = e.get("metadata") or {}
    b, d = m.get("birth_year"), m.get("death_year")
    yrs = f"｜{b or '?'}-{d or '?'}" if (b or d) else ""
    rd = _ls_norm(m.get("reading"))
    return f"{int(e['numeric_id'])}｜{e['title']}{f'｜{rd}' if rd else ''}{yrs}"


_ID_MARK = re.compile(r"[（(]id\d+[)）]")
_CAT_MARK = re.compile(r"[（(]歌舞伎")


def _ls_dedup_candidates(cand_ents, gold_id):
    """Collapse the "one person, two options" duplicates the catalog leaves in a
    candidate list. Within a (base-name, 代目) slot the KB sometimes carries both a
    real person record `X(歌舞伎 役者)〈d〉` (birth/death, dated 名乗 table) AND a
    bare aggregate stub `X〈d〉` (no era). To a labeler these read as the same name
    twice, so the stub is dropped whenever a categorised record shares its slot.

    Deliberately-distinct records tagged `(idNNNN)` are NOT stubs and are kept —
    those mark genuinely different actors the catalog chose to separate. If the
    gold pointed at a dropped stub it is remapped to the kept categorised record,
    so the dropdown never loses its answer. Returns (kept_ents, new_gold_id)."""
    by_slot = defaultdict(list)
    for e in cand_ents:
        by_slot[(_ls_base_name(e["title"]), _ls_daime_num(e))].append(e)

    kept, remap = [], {}
    for slot, es in by_slot.items():
        has_cat = any(_CAT_MARK.search(e["title"]) for e in es)
        keeper = next((e for e in es if _CAT_MARK.search(e["title"])), None)
        for e in es:
            bare = not _CAT_MARK.search(e["title"]) and not _ID_MARK.search(e["title"])
            if has_cat and bare and keeper is not None:
                remap[int(e["numeric_id"])] = int(keeper["numeric_id"])  # drop stub
            else:
                kept.append(e)
    kept.sort(key=_ls_daime_num)
    return kept, remap.get(int(gold_id), int(gold_id))


def _ls_evidence_html(cand_ents, year):
    """One HTML card per candidate for the ARC-DB side panel: its name history,
    life span, a bio snippet, and a badge on whoever was active at `year`."""
    parts = [
        '<div style="font:13px sans-serif">',
        f'<div style="margin-bottom:6px;color:#555">mention year: '
        f"<b>{year if year not in (None, '') else '?'}</b> — "
        f"each card shows the ARC-DB name history (改名表); "
        f"pick the generation active at that year.</div>",
    ]
    for e in cand_ents:
        m = e.get("metadata") or {}
        w = _ls_active_window(e)
        active = bool(w and year not in (None, "") and w[0] - 5 <= year <= w[1] + 5)
        hn, hd = _ls_headline(e)
        url = _ls_arc_url(m.get("detail_id"), hn, hd, title=e.get("title"))
        link = f' <a href="{html.escape(url)}" target="_blank">ARC DB ↗</a>' if url else ""
        head = (
            f"<b>{html.escape(e['title'])}</b> "
            f"({_ls_yspan(m.get('birth_year'), m.get('death_year'))}) "
            f'<span style="color:#888">{html.escape((m.get("reading") or "").strip())}</span>'
            f' <code style="color:#888">#{int(e["numeric_id"])}</code>{link}'
        )
        badge = (
            '<span style="background:#2f9e44;color:#fff;padding:0 5px;border-radius:3px">'
            f"active at {year}</span> "
            if active
            else ""
        )
        rows = "".join(
            f"<tr><td>{html.escape((p.get('name') or '').strip())}"
            f"〈{html.escape(str(p.get('daime') or ''))}〉</td>"
            f'<td style="color:#666">{_ls_yspan(p.get("start_year"), p.get("end_year"))}</td></tr>'
            for p in _ls_clean_kaimeihyou(e)
        )
        table = (
            (
                f'<table style="border-collapse:collapse;margin:3px 0 3px 12px">'
                f'<tr style="color:#999"><th align="left">改名 (name)</th>'
                f'<th align="left">期間 (years)</th></tr>{rows}</table>'
            )
            if rows
            else ""
        )
        bio = html.escape((e.get("text") or "").strip()[:140]).replace("\n", " ")
        parts.append(
            f'<div style="border-left:3px solid '
            f'{"#2f9e44" if active else "#ddd"};padding:2px 0 6px 8px;margin:6px 0">'
            f"{badge}{head}{table}"
            f'<div style="color:#777;font-size:12px">{bio}…</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


class Dataset:
    """A NEL dataset (train/valid/test/entities) with read / clean / save / split.

    Construct from a yakusya 4-file dir (positional `data_dir`, back-compatible
    with the former NEL_dataset), from explicit lists, or via the classmethods
    load()/from_dir()/from_hipe()."""

    SPLITS = ("pair_disjoint", "entity_disjoint", "temporal", "book_disjoint")

    def __init__(
        self,
        data_dir=None,
        *,
        load_entities=True,
        train=None,
        valid=None,
        test=None,
        entities=None,
        name=None,
        layout="yakusya",
    ):
        self.name = name
        self.layout = layout
        self.data_dir = data_dir
        if data_dir is not None:  # 4-file read (NEL_dataset behavior)
            self.train = _read_jsonl(os.path.join(data_dir, "train.jsonl"))
            self.valid = _read_jsonl(os.path.join(data_dir, "valid.jsonl"))
            self.test = _read_jsonl(os.path.join(data_dir, "test.jsonl"))
            self.entities = _read_entities(data_dir) if load_entities else []
        else:
            self.train = train or []
            self.valid = valid or []
            self.test = test or []
            self.entities = entities or []
        # numeric_id -> Evidence: the candidate-side non-text annotation, leak-filtered.
        # Query-independent, so it is correct to build once here rather than per call;
        # a mention's own keys come from Evidence.from_mention(sample) at query time,
        # since pairing the two is what produces the decisive fields.
        self.evidence = Evidence.index(self.entities)

    # ---- constructors ----
    @classmethod
    def from_dir(cls, data_dir, *, load_entities=True, name=None):
        """yakusya 4-file directory."""
        return cls(data_dir, load_entities=load_entities, name=name, layout="yakusya")

    @classmethod
    def from_hipe(cls, name, lang=None, split="test", path=None):
        """HIPE per-language layout: <dir>/<lang>/<split>.jsonl (QID gold, no
        entities). lang=None or "all" concatenates every language (each row keeps
        metadata.language); a specific lang loads just that one. Lands in `.test`."""
        base = path or _resolve_path(name)
        langs = cls.langs(name, path=base) if lang in (None, "all") else [lang]
        rows = []
        for lg in langs:
            rows += _read_jsonl(os.path.join(base, lg, f"{split}.jsonl"))
        return cls(test=rows, name=f"{name}/{'+'.join(langs) or '?'}/{split}", layout="hipe")

    @classmethod
    def load(cls, name, *, lang=None, split="test", load_entities=True, window=None):
        """Load ANY dataset under data/dataset_* or data/dataset/ by name,
        auto-detecting the layout (yakusya 4-file / eval-only, or HIPE per-lang).
        For a per-language dataset, `lang` picks one (default: all concatenated).
        `window=W` truncates context to W chars each side (the w100/w200/w300
        ablation from a single wide-stored dataset)."""
        path = _resolve_path(name)
        layout = _detect_layout(path)
        if layout == "hipe":
            ds = cls.from_hipe(name, lang=lang, split=split, path=path)
        elif layout == "yakusya":
            ds = cls.from_dir(path, load_entities=load_entities, name=name)
        else:
            raise ValueError(
                f"{name!r} ({path}) is not a loadable NEL dataset "
                f"(no train/valid/test.jsonl and no per-language subdirs)."
            )
        return ds.window(window) if window else ds

    # ---- discovery ----
    @classmethod
    def langs(cls, name, path=None):
        """Language subdirs of a per-language dataset (empty for 4-file datasets)."""
        base = path or _resolve_path(name)
        if not os.path.isdir(base):
            return []
        return sorted(
            d
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
            and any(f.endswith(".jsonl") for f in os.listdir(os.path.join(base, d)))
        )

    @classmethod
    def splits(cls, name, lang=None, path=None):
        """Available split names (train/dev/test/...) for a dataset (or one lang)."""
        base = path or _resolve_path(name)
        d = os.path.join(base, lang) if lang else base
        if not os.path.isdir(d):
            return []
        return sorted(f[:-6] for f in os.listdir(d) if f.endswith(".jsonl"))

    @classmethod
    def available(cls):
        """{name: (layout, path)} for every registered dataset that exists on disk
        and is a loadable NEL set."""
        out = {}
        for n, p in config.DATASETS.items():
            lay = _detect_layout(p)
            if lay in ("yakusya", "hipe"):
                out[n] = (lay, p)
        return out

    # ---- unified access ----
    @property
    def all(self):
        """train + valid + test pooled (what the split methods re-partition)."""
        return self.train + self.valid + self.test

    # `samples` is the layout-agnostic view: for HIPE this is the loaded split,
    # for yakusya it is the whole pool.
    @property
    def samples(self):
        return self.all

    def __len__(self):
        return len(self.train) + len(self.valid) + len(self.test)

    def __repr__(self):
        return (
            f"Dataset(name={self.name!r}, layout={self.layout!r}, "
            f"train={len(self.train)}, valid={len(self.valid)}, "
            f"test={len(self.test)}, entities={len(self.entities)})"
        )

    # ---- cleaning (generic, safe ops; domain cleaning stays in build_dataset.py) ----
    def clean(self, *, drop_no_gold=True, dedup=True, lower_type=False):
        """In-place generic cleanup across all splits. Returns self.
        - drop_no_gold: drop rows with no/empty label_id
        - dedup: drop exact (mention, label_id, last-80-context) duplicates
        - lower_type: lowercase the `type` field (harmonizes HIPE LOC vs loc)"""

        def _key(r):
            return (
                r.get("mention", ""),
                r.get("label_id"),
                (r.get("context_left", "") or "")[-80:],
            )

        for attr in ("train", "valid", "test"):
            rows = getattr(self, attr)
            out, seen = [], set()
            for r in rows:
                if drop_no_gold and (r.get("label_id") in (None, "", -1)):
                    continue
                if lower_type and isinstance(r.get("type"), str):
                    r = {**r, "type": r["type"].lower()}
                if dedup:
                    k = _key(r)
                    if k in seen:
                        continue
                    seen.add(k)
                out.append(r)
            setattr(self, attr, out)
        return self

    def filter(self, pred):
        """Keep rows where pred(row) is truthy, across all splits. Returns self."""
        for attr in ("train", "valid", "test"):
            setattr(self, attr, [r for r in getattr(self, attr) if pred(r)])
        return self

    def tagged(self, tag):
        """A NEW sub-Dataset of the rows carrying `tag` in metadata.tags (across
        all splits, split membership preserved; shares the same entities). Lets a
        benchmark be MERGED into a full dataset as a filterable label — e.g.
        Dataset.load('yakusya_nel').tagged('daime') is the daime_eval subset."""

        def has(r):
            return tag in ((r.get("metadata") or {}).get("tags") or [])

        return Dataset(
            train=[r for r in self.train if has(r)],
            valid=[r for r in self.valid if has(r)],
            test=[r for r in self.test if has(r)],
            entities=self.entities,
            name=f"{self.name or '?'}:{tag}",
            layout=self.layout,
        )

    def window(self, w):
        """A NEW Dataset with the context window truncated to `w` chars each side
        (context_left's last w chars + the trailing [年:..] token; context_right's
        first w). Collapses the w100/w200/w300 ablation into ONE wide-stored
        dataset + this load-time param: slicing a w300-stored context to w reads
        exactly like recut_daime_window.py's page[s-w:s]/page[e:e+w] (they share
        the same page-derived body), so no re-cut from source is needed here.
        Only truncates DOWN — stored context must be >= w (store at w300)."""

        def cut(r):
            cl = r.get("context_left", "") or ""
            m = _YEAR_TOKEN_RE.search(cl)
            token = m.group(0) if m else ""
            body = cl[: m.start()] if m else cl
            return {
                **r,
                "context_left": body[-w:] + token,
                "context_right": (r.get("context_right", "") or "")[:w],
            }

        return Dataset(
            train=[cut(r) for r in self.train],
            valid=[cut(r) for r in self.valid],
            test=[cut(r) for r in self.test],
            entities=self.entities,
            name=f"{self.name or '?'}:w{w}",
            layout=self.layout,
        )

    def tags(self):
        """All dataset tags present in the data (metadata.tags), with counts."""
        c = Counter()
        for r in self.all:
            for t in (r.get("metadata") or {}).get("tags") or []:
                c[t] += 1
        return dict(c)

    # ---- saving (replaces the duplicated write_jsonl free functions) ----
    def save(self, out_dir, *, splits=("train", "valid", "test"), entities=True):
        """Write the yakusya 4-file layout into out_dir (only non-empty splits)."""
        os.makedirs(out_dir, exist_ok=True)
        for sp in splits:
            rows = getattr(self, sp)
            if rows:
                _write_jsonl(rows, os.path.join(out_dir, f"{sp}.jsonl"))
        if entities and self.entities:
            _write_jsonl(self.entities, os.path.join(out_dir, "entities.jsonl"))
        return out_dir

    def save_jsonl(self, path, which="test"):
        """Write a single split to `path`."""
        _write_jsonl(getattr(self, which), path)
        return path

    def to_label_studio(self, out_dir, *, which="test", limit=None):
        """Export a split as Label Studio import JSON + its labeling config.

        daime-only: the task it sets up is "which 代目 does this mention name?", and
        the candidate set is every ritsumei generation sharing the mention's base
        name — the true choice set, not a retrieval guess. Each task carries:

          data.text        context_left + mention + context_right
          data.candidates  the dropdown, one line per generation
          data.evidence_html  a per-candidate ARC-DB card (改名表 + life span + bio),
                           badging whoever was active in the mention's year
          predictions[0]   the gold as a pre-annotation (span + generation), so an
                           annotator starts from it rather than a blank task

        -> the tasks JSON path. Prints how often the gold made it into the dropdown;
        anything below 100% means the candidate builder missed a generation."""
        os.makedirs(out_dir, exist_ok=True)
        byid = {int(e["numeric_id"]): e for e in self.entities}
        canon, _ = build_canon_map(self.entities, mode="full")
        cid = lambda i: canon.get(int(i), int(i))

        # base name -> one representative per canonical ritsumei generation
        rit_by_base = defaultdict(dict)
        for e in self.entities:
            if e["idx"].startswith("ritsumei"):
                c = cid(e["numeric_id"])
                rit_by_base[_ls_base_name(e["title"])].setdefault(c, byid[c])

        rows = getattr(self, which)[:limit] if limit else getattr(self, which)
        tasks, n_gold = [], 0
        for i, r in enumerate(rows):
            # drop the inline [年:YYYY|書:...] tag from the display text (the year is
            # already shown in the meta line and used by the evidence panel); strip
            # before measuring so the mention span offsets stay correct.
            left = _YEAR_TAG.sub(" ", r.get("context_left", "") or "")
            men = r.get("mention", "") or ""
            right = _YEAR_TAG.sub(" ", r.get("context_right", "") or "")
            text = left + men + right
            start, end = len(left), len(left) + len(men)

            g = cid(r["label_id"])
            cands = dict(rit_by_base.get(_ls_base_name(men), {}))
            cands.setdefault(g, byid.get(g))  # the gold is always offered
            cand_ents = sorted((e for e in cands.values() if e), key=_ls_daime_num)
            # drop the "same person, two options" catalog stubs; remap gold if hit
            cand_ents, g = _ls_dedup_candidates(cand_ents, g)
            candidates = [{"value": _ls_cand_label(e)} for e in cand_ents]
            candidates.append(_LS_NIL)  # 「正解が候補一覧にない」NIL 選択肢
            gold_val = _ls_cand_label(byid[g]) if g in byid else None
            n_gold += any(c["value"] == gold_val for c in candidates)

            md = r.get("metadata") or {}
            year = md.get("year")
            year = year if isinstance(year, int) else None
            rid = f"m{i}"
            result = [
                {
                    "id": rid,
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": start,
                        "end": end,
                        "text": men,
                        "labels": [r.get("type", "役者")],
                    },
                }
            ]
            if gold_val:
                result.append(
                    {
                        "id": rid,
                        "from_name": "link",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"start": start, "end": end, "text": men, "choices": [gold_val]},
                    }
                )
            tasks.append(
                {
                    "data": {
                        "text": text,
                        "candidates": candidates,
                        "evidence_html": _ls_evidence_html(cand_ents, year),
                        "mention": men,
                        "year": md.get("year", ""),
                        "book": md.get("book_name", ""),
                        "gold_id": g,
                        "sample_idx": i,
                    },
                    "predictions": [{"model_version": "gold", "result": result}],
                }
            )

        out_json = os.path.join(out_dir, f"{self.name or 'dataset'}_label_studio.json")
        with open(out_json, "w") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, "labeling_config.xml"), "w") as f:
            f.write(_LS_CONFIG)
        avg = sum(len(t["data"]["candidates"]) for t in tasks) / max(len(tasks), 1)
        print(f"[label-studio] {len(tasks)} tasks -> {out_json}")
        print(
            f"  gold in dropdown: {n_gold}/{len(tasks)} "
            f"({100 * n_gold / max(len(tasks), 1):.1f}%)  avg candidates/task {avg:.2f}"
        )
        return out_json

    def export_candidates(self, out, *, method="gather", topk=20, which="test"):
        """Write a chatel candidate JSON (the schema gen_blink_cands uses) so the
        reranker-style methods consume it unchanged via --ice/--chatel
        (In_Context_EL / DeepEL / SumMC). The pool is chosen by `method`:

          gather   (default) surface + alias-at-year (kaimeihyou) + family + year
                   rerank via EntityDB.candidates — the agentic pool, ~99% recall on
                   the 2214 daime benchmark. The pool mad/maps build internally.
          surface|name|fuzzy|blink|union   the other EntityDB.candidates strategies.
          basename standalone base-name pool: every entity whose title base name
                   (stripped of 〈代目〉, reading and parentheticals) equals the
                   mention's base name — BLINK-free, ~100% recall on daime but blind
                   to stage-name changes, so ~57% on the full 2214 set.

        Folded in from the former methods/candidates/gen_cands.py; EntityDB lives
        under src/methods and is imported lazily (config puts those subfolders on
        sys.path at import)."""
        rows = getattr(self, which)
        ents = self.entities
        byid = {int(e["numeric_id"]): e for e in ents}

        # basename pool is self-contained (no EntityDB needed)
        if method == "basename":
            by_base = defaultdict(list)
            for e in ents:
                by_base[_ls_base_name(e["title"])].append(e)
            db = None
        else:
            os.environ.setdefault("OPENAI_API_KEY", "x")  # EntityDB import only; no API call
            from llm_tool_disambiguator import EntityDB

            db = EntityDB(ents)
            db.entities = ents

        result, n_recall = {}, 0
        for i, s in enumerate(rows):
            men = s["mention"]
            if method == "basename":
                cands = by_base.get(_ls_base_name(men), [])[:topk]
            else:
                s["_idx"] = i  # for the blink route
                cids = db.candidates(s, method=method, topk=topk)
                cands = [byid[c] for c in cids if c in byid]
            titles = [e["title"] for e in cands]
            descs = [(e.get("text", "") or e.get("entity", "") or "")[:1000] for e in cands]
            if s.get("label_id") in {int(e["numeric_id"]) for e in cands}:
                n_recall += 1
            left = s.get("context_left", "")
            result[f"yakusya_test_{i}"] = {
                "doc_name": f"yakusya_test_{i}",
                "sentence": left + men + s.get("context_right", ""),
                "entities": {
                    "starts": [len(left)],
                    "ends": [len(left) + len(men)],
                    "entity_mentions": [men],
                    "entity_names": [s.get("label_title", "")],
                    "entity_candidates": [titles],
                    "entity_candidates_descriptions": [descs],
                },
            }
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
        n = len(rows)
        print(f"[export_candidates:{method}] {n} mentions -> {out}")
        if n:
            print(
                f"[export_candidates:{method}] pool recall (gold in candidates): "
                f"{n_recall}/{n} = {n_recall / n * 100:.1f}%"
            )
        return out

    # ---- the 4 disjoint re-splits (yakusya-only) ----
    def split(self, scheme):
        """Dispatch by scheme name; returns (train, valid, test)."""
        if self.layout != "yakusya":
            raise ValueError(
                f"split({scheme!r}) is only defined for the yakusya 4-file layout; "
                f"this dataset is layout={self.layout!r} (fixed splits, QID gold). "
                f"Use its existing train/valid/test instead."
            )
        if scheme not in self.SPLITS:
            raise ValueError(f"unknown split scheme {scheme!r}; choose from {self.SPLITS}")
        return getattr(self, f"split_{scheme}")()

    def split_pair_disjoint(self):
        """Hold out whole (mention, label_id) pairs — no test pair seen in train."""
        groups = defaultdict(list)
        for s in self.all:
            groups[(s["mention"], s["label_id"])].append(s)
        keys = list(groups.keys())
        random.Random(SPLIT_SEED).shuffle(keys)
        n_test = int(len(keys) * 0.10)
        n_valid = int(len(keys) * 0.10)
        test_keys = set(keys[:n_test])
        valid_keys = set(keys[n_test : n_test + n_valid])
        train, valid, test = [], [], []
        for k, samples in groups.items():
            (test if k in test_keys else valid if k in valid_keys else train).extend(samples)
        return train, valid, test

    def split_entity_disjoint(self):
        """Hold out whole entities — test entities never appear in train."""
        ent_ids = sorted({s["label_id"] for s in self.all})
        random.Random(SPLIT_SEED).shuffle(ent_ids)
        n_test = max(1, int(len(ent_ids) * 0.10))
        n_valid = max(1, int(len(ent_ids) * 0.10))
        test_ents = set(ent_ids[:n_test])
        valid_ents = set(ent_ids[n_test : n_test + n_valid])
        train, valid, test = [], [], []
        for s in self.all:
            if s["label_id"] in test_ents:
                test.append(s)
            elif s["label_id"] in valid_ents:
                valid.append(s)
            else:
                train.append(s)
        return train, valid, test

    def split_temporal(self):
        """Time split by metadata.year: >=1775 test, ==1774 valid, else/none train."""
        train, valid, test = [], [], []
        for s in self.all:
            y = (s.get("metadata") or {}).get("year")
            if y is None:
                train.append(s)
            elif y >= 1775:
                test.append(s)
            elif y == 1774:
                valid.append(s)
            else:
                train.append(s)
        return train, valid, test

    def split_book_disjoint(self):
        """Hold out the two largest books (largest -> test, 2nd -> valid)."""
        by_book = defaultdict(list)
        for s in self.all:
            by_book[(s.get("metadata") or {}).get("book_name", "?")].append(s)
        books = sorted(by_book.keys(), key=lambda b: -len(by_book[b]))
        test_book = books[0]
        valid_book = books[1] if len(books) > 1 else None
        train, valid, test = [], [], []
        for b, samples in by_book.items():
            if b == test_book:
                test.extend(samples)
            elif b == valid_book:
                valid.extend(samples)
            else:
                train.extend(samples)
        return train, valid, test


# Backward-compatible alias: the former nel_models.NEL_dataset is this class.
NEL_dataset = Dataset


# --- folded CLI tools ---------------------------------------------------------
# Previously the standalone scripts tools/build_daime_ft.py and
# tools/gen_label_studio.py; folded in here so the logic lives with the Dataset
# it operates on. Exposed via the `if __name__ == "__main__"` sub-command CLI.


def build_daime_ft_splits(src=None, dst=None, entities=None):
    """Build a trainable train/valid/test dataset from the eval-only
    daime_eval_full (2214) by its metadata.origin_split, so run_all.py can
    train + evaluate on it.

      origin_split=train (1756) -> train.jsonl   (encoders learn on these)
      origin_split=valid (228)  -> valid.jsonl
      origin_split=test  (230)  -> test.jsonl    (held-out, fair for trained models)

    Then recut to an exact 100-char window (recut_daime_window.py is invoked
    separately). entities.jsonl is symlinked from dataset_daime_eval (5000 ents)."""
    src = src or os.path.join(config.DS_DAIME_EVAL_FULL, "test.jsonl")
    dst = dst or config.DS_DAIME_FT
    entities = entities or os.path.join(config.DS_DAIME_EVAL, "entities.jsonl")

    os.makedirs(dst, exist_ok=True)
    rows = [json.loads(l) for l in open(src)]
    for split, key in (("train", "train"), ("valid", "valid"), ("test", "test")):
        sub = [r for r in rows if (r.get("metadata") or {}).get("origin_split") == key]
        with open(os.path.join(dst, f"{split}.jsonl"), "w") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(sub)}")

    link = os.path.join(dst, "entities.jsonl")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(entities), link)
    print("entities.jsonl -> symlink (5000 entities)")
    return dst


def export_label_studio(
    data_dir=None, *, name="daime_eval_full", out_dir=None, which="test", limit=None
):
    """Generate a Label Studio import file via the existing Dataset.to_label_studio.

      data/label_studio/daime_eval_full_label_studio.json   all 2214 tasks (single file)"""
    data_dir = data_dir or config.DS_DAIME_EVAL_FULL
    out_dir = out_dir or os.path.join(config.PROJ_DATA, "label_studio")
    os.makedirs(out_dir, exist_ok=True)
    ds = Dataset(data_dir, name=name)
    out = ds.to_label_studio(out_dir, which=which, limit=limit)
    print(f"{name}_label_studio.json  <- {len(getattr(ds, which))} tasks")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset build / export tools.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ft = sub.add_parser("build-ft", help="build the trainable daime_ft train/valid/test split")
    p_ft.add_argument("--src", default=None, help="source test.jsonl (default: daime_eval_full)")
    p_ft.add_argument("--dst", default=None, help="output dir (default: dataset_daime_ft)")
    p_ft.add_argument(
        "--entities", default=None, help="entities.jsonl to symlink (default: daime_eval)"
    )

    p_ls = sub.add_parser("label-studio", help="export a Label Studio import file")
    p_ls.add_argument("--data-dir", default=None, help="dataset dir (default: daime_eval_full)")
    p_ls.add_argument("--name", default="daime_eval_full", help="dataset name (output file prefix)")
    p_ls.add_argument("--out", default=None, help="output dir (default: data/label_studio)")
    p_ls.add_argument("--which", default="test", help="split to export (default: test)")
    p_ls.add_argument("--limit", type=int, default=None, help="cap number of tasks")

    p_c = sub.add_parser(
        "candidates", help="write a chatel candidate JSON for the reranker methods"
    )
    p_c.add_argument("--test", required=True, help="test.jsonl (mentions)")
    p_c.add_argument("--entities", required=True, help="entities.jsonl (KB)")
    p_c.add_argument("--out", required=True, help="output chatel candidate JSON")
    p_c.add_argument("--topk", type=int, default=20)
    p_c.add_argument(
        "--method",
        default="gather",
        help="candidate pool: gather|surface|name|fuzzy|blink|union|basename",
    )

    args = parser.parse_args()
    if args.cmd == "build-ft":
        build_daime_ft_splits(src=args.src, dst=args.dst, entities=args.entities)
    elif args.cmd == "label-studio":
        export_label_studio(
            args.data_dir, name=args.name, out_dir=args.out, which=args.which, limit=args.limit
        )
    elif args.cmd == "candidates":
        ds = Dataset(
            test=_read_jsonl(args.test),
            entities=[json.loads(l) for l in open(args.entities)],
            name="candidates",
        )
        ds.export_candidates(args.out, method=args.method, topk=args.topk)
