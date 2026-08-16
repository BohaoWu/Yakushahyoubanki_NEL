#!/usr/bin/env python3
"""How each dataset is built, from raw material to a `Dataset`.

Four corpus families with nothing in common at the source: 役者評判記 (kabuki-actor
critiques) starts from scanned CSV plus an online person DB, HIPE from a TSV repo,
Mahānāma from CoNLL-U. They agree only at the far end: all land in the same on-disk
format, and `build()` hands back a `Dataset`. This file is split along that seam
into three layers:

  PASSES    Each pass is one deterministic transform: dataset dir in, dataset dir out.
            Except PART 0 (which reaches an external DB), every pass is verified
            byte-for-byte against its on-disk artifact.
  BUILDERS  One builder per family: declares which passes the corpus needs, and in
            what order. No reimplementation.
  CLI       A thin wrapper: parse args, call a builder or a pass.

Usage:

  from build_dataset import get_builder
  ds = get_builder("daime_eval_full").build()     # -> Dataset(test=2214, entities=4885)
  ds = get_builder("hipe2020_de").build()         # read straight from disk if present, no rebuild

  python src/build_dataset.py list                # the dataset registry
  python src/build_dataset.py daime_eval          # base=DS_RELINKED_YR_V2 -> DS_DAIME_EVAL_FULL
  python src/build_dataset.py hipe --split all

Paths always come from config.DATASETS, never hardcoded in this file.

⚠ Reproducibility note: the officially frozen daime_eval(1479) cannot be reproduced
bit-for-bit — what this file produces is the build method itself (current data →
2214). PART 0 (raw text → base) depends on the external Ritsumeikan DB, whose result
cannot be reproduced offline, so it is kept as-is rather than rewritten; that is
exactly why `build()` does not rebuild an existing dataset by default.
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- locate project root / import config + entity_canon (file now lives in src/) ---
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)  # entity_canon.py
import config  # noqa: E402  DS_* path registry
from dataset import Dataset  # noqa: E402  build() returns one
from entity_canon import build_canon_map  # noqa: E402

# RitsumeiSourceBuilder reaches the live Ritsumeikan DB through these. Kept as an
# import, not inlined: ritsumei_client is the I/O boundary — HTTP, a 3s rate limit,
# retries, a disk cache and ~500 lines of HTML parsing — and none of that belongs in
# a file about which passes a corpus needs.
from ritsumei_client import (  # noqa: E402
    DEFAULT_MAX_RESULTS,
    fetch_actor_detail,
    query_ritsumei_db,
)

NONPERSON = {
    "yakusya_演目名",
    "yakusya_書名",
    "yakusya_役名",
    "yakusya_音曲",
    "theater",
    "yakusya_事項",
}

# ============================================================================
# shared helpers
# ============================================================================


def base_name(s):
    return re.sub(r"\s+", "", re.sub(r"〈[^〉]*〉|\([^)]*\)|（[^）]*）", "", s or "")).replace(
        "　", ""
    )


def norm(s):
    return (s or "").replace("　", "").replace(" ", "")


def _load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def _write_jsonl(rows, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================================
# BUILDERS
#
# One class per corpus family. Each owns its passes: a pass is one deterministic
# step, dataset dir in, dataset dir out, and every one below is verified
# byte-for-byte against the artifact it produces — except RitsumeiSourceBuilder,
# which reaches a live DB and says so.
# ============================================================================


# ============================================================================
# PART 0: 役者評判記 raw CSV + Ritsumeikan DB -> base corpus
# ============================================================================


class RitsumeiSourceBuilder:
    """CSV annotations + the Ritsumeikan person DB -> base corpus (entities.jsonl + train/valid/test).

    Not the same kind of thing as the other builders, so it stands on its own: they
    do dataset transforms (dir in, dir out), whereas this does collection and store
    construction — parsing scanned CSV, querying the online person DB, scoring
    candidates, splitting. YakusyaBuilder's chain starts from its output.

    The whole block is kept byte-for-byte from the former
    build_yakushahyoubanki_dataset.py, not rewritten. Its build_entities_and_queries
    queries the online DB, whose result cannot be reproduced offline (an incomplete
    cache, 3s per request, O(actors x candidates) detail lookups that time out even
    fully cached), so unlike every downstream pass it has no frozen artifact to diff
    against. The other 15 methods are pure functions, each verified by AST to have an
    unchanged body.
    """

    THEATERS = [
        {
            "title": "中村座",
            "reading": "なかむらざ",
            "text": """中村座は江戸の歌舞伎劇場で、江戸三座の一つ。寛永元年（1624）初世猿若（中村）勘三郎が江戸中橋に創立しました。初期は猿若座として知られていましたが、やがて座元の本姓である中村を取って改称されました。

    初代中村（猿若）勘三郎によって1624年（寛永元年）に猿若座という名前で作られたのが始まりとされており、江戸で最も権威のある劇場として認識されていました。1842年（天保13年）、堺町から猿若町に移転した後、明治時代に入ってからも場所や名前を変えて存続していました。

    劇場は禰宜町・堺町・猿若町と移転し、明治9年（1876）休座。のち何度か再興したが長続きせず、同26年（1893年）に焼失し、再興されることはありませんでした。江戸時代を通じて、中村座は鶴屋南北の代表作《東海道四谷怪談》（1825）など時代を代表する作品を上演し、江戸歌舞伎の筆頭として認識されていた由緒ある劇場でした。""",
            "category": "劇場",
            "zamoto_family": "中村勘三郎",
            "founded": 1624,
            "original_location": "江戸中橋",
            "later_location": "浅草猿若町",
            "crest": "三升紋",
            "wikipedia_ref": "https://ja.wikipedia.org/wiki/中村座",
            "old_entity_titles": ["中村"],
        },
        {
            "title": "市村座",
            "reading": "いちむらざ",
            "text": """市村座は江戸時代から続いた芝居小屋です。寛永十一年（1634）江戸堺町に建てられた村山座が始まりで、承応元年（1652）に市村宇左衛門が興行権を買って市村座と改称しました。

    「大和守日記」では寛文4～7年（1664～1667年）市村宇左衛門・竹之丞の興行が確認されています。また、1667（寛文7）年頃に「市村座」となりました。元禄期までに所在地も葺屋町と定まり、座元名を市村宇左衛門としていました。当初、日本橋葺屋町（現在の日本橋人形町3丁目）にありました。

    天保十三年には当局の命令で浅草猿若町へ、その後明治二十五年には下谷二長町へ移転しています。下谷二長町時代には初代中村吉右衛門や6代目尾上菊五郎が活躍し、明治末期には著名な歌舞伎役者を起用して若手劇場として栄えましたが、1932年（昭和7年）に焼失し、再興されることはありませんでした。""",
            "category": "劇場",
            "zamoto_family": "市村宇左衛門/市村羽左衛門",
            "founded": 1634,
            "original_location": "日本橋葺屋町",
            "later_location": "浅草猿若町",
            "closed": 1932,
            "wikipedia_ref": "https://ja.wikipedia.org/wiki/市村座",
            "old_entity_titles": ["市村", "市村羽左衛門"],
        },
        {
            "title": "守田座",
            "reading": "もりたざ",
            "text": """守田座は江戸にあった歌舞伎の芝居小屋で、江戸三座のひとつ。1660年（万治3年）、初代森田勘彌が木挽町五丁目（現在の銀座六丁目）に森田座を開場しました。座元は森田（守田）勘彌代々です。

    守田座の経営は不安定で、しばしば破綻・休座して、控櫓であった河原崎座に興行権を譲っていました。1843年（天保14年）、天保の改革により、木挽町から猿若町（現在の台東区浅草6丁目）へ移転させられました。

    安政3年（1856年）5月に河原崎座が失火で全焼すると十一代目森田勘彌によって森田座が再興され、その2年後の1858年（安政5年）に名称を森田座から守田座に改称し、姓の森田も守田に改めました。明治5年（1872年）に新富町に移転、明治8年（1875年）にこれを新富座と改称した。新富座の座元十二代目守田勘彌は明治11年（1878年）6月に西洋式の大劇場新富座を開場し、新富座は当時最大の興行施設で、ガス灯による照明器具を備えてそれまでできなかった夜間上演を可能した、画期的な近代劇場だった。""",
            "category": "劇場",
            "zamoto_family": "森田勘彌/守田勘彌",
            "founded": 1660,
            "original_location": "木挽町五丁目",
            "later_location": "猿若町→新富町",
            "later_name": "新富座",
            "wikipedia_ref": "https://ja.wikipedia.org/wiki/守田座",
            "old_entity_titles": ["森田", "守田"],
        },
    ]

    # match-flag bit definitions
    MATCH_FLAG_EXACT_NAME = 1  # 0000001: exact name match

    MATCH_FLAG_KANA_MATCH = 2  # 0000010: exact kana match

    MATCH_FLAG_DAIME_MATCH = 4  # 0000100: generation-number (代数) match

    MATCH_FLAG_PERIOD_CLOSE = 8  # 0001000: 名乗期間 (naming period) close (year is near but not inside the period)

    MATCH_FLAG_YEAR_MATCH = 16  # 0010000: year / period match

    @staticmethod
    def merge_csvs(
        csv_dir: str, output_file: str, start_row: int = 0, end_row: int | None = None
    ) -> list[dict]:
        """
        Parse and merge all CSV files.

        Args:
            csv_dir: directory of CSV files
            output_file: output file path
            start_row: starting row number
            end_row: ending row number

        Returns:
            list of records
        """
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory {csv_dir}")

        print(f"📂 Found {len(csv_files)} CSV files")
        print(f"📋 Supported entity types: {len(config.SUPPORTED_ENTITY_CATEGORIES)}")
        print(f"   {', '.join(config.SUPPORTED_ENTITY_CATEGORIES)}")

        all_records = []
        total_records = 0

        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)

            try:
                # try several Japanese encodings
                encodings = ["shift-jis", "cp932", "iso-2022-jp", "euc-jp", "utf-8"]
                records = None
                last_error = None

                for encoding in encodings:
                    try:
                        with open(csv_path, encoding=encoding) as f:
                            reader = csv.DictReader(f)
                            records = list(reader)
                        break  # read succeeded
                    except (UnicodeDecodeError, LookupError) as e:
                        last_error = e
                        continue

                if records is None:
                    raise last_error if last_error else Exception("Could not read file with any encoding")

                # apply the row-range filter
                if start_row > 0 or end_row is not None:
                    records = records[start_row:end_row]

                # add source_file and year to each record
                book_name = csv_file.replace(".csv", "")
                year_info = config.get_book_year_info(book_name)
                year = year_info.get("year") if year_info else None
                year_nengo = year_info.get("year_str") if year_info else ""

                for record in records:
                    # rename keyword to mention, parent_keyword to parent_name
                    keyword = record.pop("keyword", "").strip()
                    parent_keyword = record.pop("parent_keyword", "").strip()

                    record["mention"] = keyword
                    # parent_name logic: use parent_keyword if non-empty, else mention
                    record["parent_name"] = parent_keyword if parent_keyword else keyword

                    # strip leading/trailing whitespace from all string fields
                    for field in ["kana", "comment"]:
                        if field in record and isinstance(record[field], str):
                            record[field] = record[field].strip()

                    # convert start and end to integers
                    if "start" in record and record["start"]:
                        try:
                            record["start"] = int(record["start"])
                        except (ValueError, TypeError):
                            record["start"] = 0
                    if "end" in record and record["end"]:
                        try:
                            record["end"] = int(record["end"])
                        except (ValueError, TypeError):
                            record["end"] = 0

                    # add file and year info
                    record["source_file"] = csv_file
                    record["book_name"] = book_name
                    if year:
                        record["year_ad"] = year
                    if year_nengo:
                        record["year_nengo"] = year_nengo

                    # remove unneeded fields
                    for field in ["sub_code", "content", "question"]:
                        record.pop(field, None)

                all_records.extend(records)
                total_records += len(records)
                print(f"  ✓ {csv_file}: {len(records)} records")

            except Exception as e:
                print(f"  ✗ {csv_file}: error - {e}")
                continue

        # save the merged result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Total: {total_records} records")
        print(f"✓ Saved to: {output_file}")

        return all_records

    @staticmethod
    def _decode_flags(flags: int) -> str:
        """Decode the match flags into a binary string."""
        return format(flags, "07b")

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalise a name."""
        name = re.sub(r"\s+", "", name)  # remove all whitespace
        name = re.sub(r"[☆★○●◎◇◆□■▽▼△▲]", "", name)  # remove special markers
        name = re.sub(r"\([^)]*\)", "", name)  # remove half-width parentheses
        name = re.sub(r"（[^）]*）", "", name)  # remove full-width parentheses
        name = re.sub(r"〈[^〉]*〉", "", name)  # remove angle brackets
        return name.strip()

    @staticmethod
    def _extract_daime(text: str) -> str | None:
        """Extract the generation number (代数) from the text."""
        match = re.search(r"〈(\d+)〉", text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def match_score(
        query_name: str, year: int | None, entity_info: dict, entity_detail: dict
    ) -> tuple[int, int, str]:
        """
        Compute the match score between a query and an entity.

        Returns:
            (match_score, match_flags, match_reasons)
        """
        flags = 0
        reasons = []

        query_normalized = RitsumeiSourceBuilder.normalize_name(query_name)
        entity_name = entity_info.get("name", "")
        entity_normalized = RitsumeiSourceBuilder.normalize_name(entity_name)

        # 1. exact name match - commented out to stay compatible with the original dataset
        # the original dataset lacks this flag; not set here, to keep the values consistent
        # if query_normalized == entity_normalized:
        #     flags |= RitsumeiSourceBuilder.MATCH_FLAG_EXACT_NAME
        #     reasons.append("exact name match")

        # 2. kana match - compare entity_info's reading against the readings in kaimeihyou
        entity_reading = entity_info.get("reading", "")
        if entity_reading and entity_detail:
            entity_reading_normalized = RitsumeiSourceBuilder.normalize_name(entity_reading)
            kaimeihyou = entity_detail.get("kaimeihyou", [])
            for period in kaimeihyou:
                period_reading = period.get("reading", "")
                if (
                    period_reading
                    and RitsumeiSourceBuilder.normalize_name(period_reading)
                    == entity_reading_normalized
                ):
                    flags |= RitsumeiSourceBuilder.MATCH_FLAG_KANA_MATCH
                    reasons.append("假名完全匹配")
                    break

        # 3. generation-number (代数) match
        query_daime = RitsumeiSourceBuilder._extract_daime(query_name)
        entity_daime = entity_info.get("daime")
        if query_daime and entity_daime and query_daime == entity_daime:
            flags |= RitsumeiSourceBuilder.MATCH_FLAG_DAIME_MATCH
            reasons.append(f"代数匹配({query_daime})")

        # 4. year / period match - three rounds: exact -> lenient -> close
        if year and entity_detail:
            kaimeihyou = entity_detail.get("kaimeihyou", [])
            birth_year = entity_detail.get("birth_year")
            death_year = entity_detail.get("death_year")

            # round 1: check for an exact 名乗期間 (naming period) match
            for period in kaimeihyou:
                start_year = period.get("start_year")
                end_year = period.get("end_year")
                if start_year and end_year and start_year <= year <= end_year:
                    flags |= RitsumeiSourceBuilder.MATCH_FLAG_YEAR_MATCH
                    reasons.append(f"名乗期間匹配({start_year}-{end_year})")
                    break

            # round 2: if no exact match, check a 50-year tolerance
            if not (flags & RitsumeiSourceBuilder.MATCH_FLAG_YEAR_MATCH):
                for period in kaimeihyou:
                    start_year = period.get("start_year")
                    end_year = period.get("end_year")
                    if start_year and end_year and start_year <= year <= end_year + 50:
                        flags |= RitsumeiSourceBuilder.MATCH_FLAG_YEAR_MATCH
                        reasons.append(f"名乗期間结束前50年内({end_year})")
                        break

            # round 3: if still no match, check for a close match (±10 years)
            if not (flags & RitsumeiSourceBuilder.MATCH_FLAG_YEAR_MATCH):
                for period in kaimeihyou:
                    start_year = period.get("start_year")
                    end_year = period.get("end_year")
                    if start_year and end_year:
                        if abs(year - start_year) <= 10 or abs(year - end_year) <= 10:
                            flags |= RitsumeiSourceBuilder.MATCH_FLAG_PERIOD_CLOSE
                            reasons.append(f"名乗期間接近({start_year}-{end_year})")
                            break

            # round 4: if nothing matched, check the birth/death years
            if not (flags & RitsumeiSourceBuilder.MATCH_FLAG_YEAR_MATCH) and not (
                flags & RitsumeiSourceBuilder.MATCH_FLAG_PERIOD_CLOSE
            ):
                if birth_year and death_year and birth_year <= year <= death_year:
                    flags |= RitsumeiSourceBuilder.MATCH_FLAG_YEAR_MATCH
                    reasons.append(f"生卒年匹配({birth_year}-{death_year})")

        score = flags
        reason_str = ", ".join(reasons) if reasons else "无匹配"

        return score, flags, reason_str

    @staticmethod
    def build_entities_and_queries(
        csv_file: str, entities_file: str, queries_file: str, cache_file: str | None = None
    ) -> dict:
        """
        Build the entity store and the queries, with the full matching logic and entity linking.
        """
        # read the CSV data
        with open(csv_file, encoding="utf-8") as f:
            records = json.load(f)

        print(f"📊 Processing {len(records)} records")

        # step 1: collect all entity mentions
        entity_mentions = defaultdict(lambda: defaultdict(list))

        for idx, record in enumerate(records):
            entity_type = record.get("category", "").strip()
            entity_name = record.get("mention", "").strip()

            if entity_type and entity_name:
                normalized_name = (
                    RitsumeiSourceBuilder.normalize_name(entity_name)
                    if entity_type == "役者"
                    else entity_name
                )

                entity_mentions[entity_type][normalized_name].append(
                    {
                        "record_idx": idx,
                        "original_name": entity_name,
                        "year": record.get("year_ad"),
                        "book_name": record.get("book_name", ""),
                        "context_left": record.get("context_left", ""),
                        "context_right": record.get("context_right", ""),
                        "start": record.get("start"),
                        "end": record.get("end"),
                    }
                )

        print("\n🏷️  Discovered entity type stats:")
        for entity_type, entities in entity_mentions.items():
            print(f"   {entity_type}: {len(entities)} unique entities")

        # step 2: build the entity store
        entities_list = []
        entity_id_counter = 0
        entity_name_to_id = {}  # {(type, normalized_name): [numeric_ids]}

        # process 役者 (actor) entities
        print("\n🎭 Querying Ritsumeikan database...")
        matched_actors = 0
        total_actors = len(entity_mentions.get("役者", {}))
        yakusya_actor_counter = 0

        for actor_name, mentions in entity_mentions.get("役者", {}).items():
            results, _ = query_ritsumei_db(
                actor_name, cache_file=cache_file, max_results=DEFAULT_MAX_RESULTS
            )

            if results and len(results) > 0:
                matched_actors += 1

                for result in results:
                    detail_id = result.get("detail_id")
                    if detail_id:
                        detail, _ = fetch_actor_detail(detail_id, cache_file=cache_file)

                        if detail:
                            first_year = mentions[0].get("year")
                            match_score, match_flags, match_reasons = (
                                RitsumeiSourceBuilder.match_score(
                                    actor_name, first_year, result, detail
                                )
                            )

                            entity = {
                                "text": detail.get("text", ""),
                                "idx": f"ritsumei_PD:{detail_id}",
                                "title": result.get("name", actor_name),
                                "entity": result.get("name", actor_name),
                                "numeric_id": entity_id_counter,
                                "metadata": {
                                    "source": "ritsumei_PD",
                                    "category": "役者",
                                    "query_name": actor_name,
                                    "daime": result.get("daime", ""),
                                    "reading": result.get("reading", ""),
                                    "detail_id": detail_id,
                                    "person_record_id": result.get("person_record_id"),
                                    "birth_year": detail.get("birth_year"),
                                    "death_year": detail.get("death_year"),
                                    "mentions_count": len(mentions),
                                    "match_candidates_count": len(results),
                                    "match_score": match_score,
                                    "match_flags": match_flags,
                                    "match_flags_binary": RitsumeiSourceBuilder._decode_flags(
                                        match_flags
                                    ),
                                    "match_reasons": match_reasons,
                                    "match_method": "binary_flags",
                                    "details": {},
                                    "kaimeihyou": detail.get("kaimeihyou", []),
                                },
                            }

                            entities_list.append(entity)
                            key = ("役者", actor_name)
                            if key not in entity_name_to_id:
                                entity_name_to_id[key] = []
                            entity_name_to_id[key].append(entity_id_counter)
                            entity_id_counter += 1
            else:
                # create a local entity
                entity = {
                    "text": f"{actor_name}（役者）。读音：{actor_name}。{mentions[0].get('book_name', '')}中有记载。文献中共{len(mentions)}次记载。",
                    "idx": f"yakusya:{yakusya_actor_counter}",
                    "title": f"{actor_name}〈1〉",
                    "entity": f"{actor_name}〈1〉",
                    "numeric_id": entity_id_counter,
                    "metadata": {
                        "source": "yakusya_local",
                        "category": "役者",
                        "query_name": actor_name,
                        "daime": "1",
                        "reading": actor_name,
                        "mentions_count": len(mentions),
                        "year": mentions[0].get("year"),
                        "book_name": mentions[0].get("book_name", ""),
                        "comment": "",
                    },
                }

                entities_list.append(entity)
                key = ("役者", actor_name)
                if key not in entity_name_to_id:
                    entity_name_to_id[key] = []
                entity_name_to_id[key].append(entity_id_counter)
                entity_id_counter += 1
                yakusya_actor_counter += 1

        # process the other entity types
        print("\n📚 Creating local entities...")
        type_counters = defaultdict(int)

        for entity_type, entities in entity_mentions.items():
            if entity_type == "役者":
                continue

            for entity_name, mentions in entities.items():
                idx_suffix = type_counters[entity_type]

                entity = {
                    "text": entity_name,
                    "idx": f"yakusya_{entity_type}:{idx_suffix}",
                    "title": entity_name,
                    "entity": entity_name,
                    "numeric_id": entity_id_counter,
                    "metadata": {
                        "source": "yakusya_local",
                        "category": entity_type,
                        "mentions_count": len(mentions),
                    },
                }

                entities_list.append(entity)
                key = (entity_type, entity_name)
                if key not in entity_name_to_id:
                    entity_name_to_id[key] = []
                entity_name_to_id[key].append(entity_id_counter)
                entity_id_counter += 1
                type_counters[entity_type] += 1

        # save entities
        with open(entities_file, "w", encoding="utf-8") as f:
            for entity in entities_list:
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")

        print(f"\n✓ Created {len(entities_list)} entities")

        # step 3: generate queries and link them to entities
        print("\n🔗 Generating queries and linking entities...")
        queries = []
        linked_count = 0

        for idx, record in enumerate(records):
            entity_type = record.get("category", "").strip()
            entity_name = record.get("mention", "").strip()

            if entity_type and entity_name:
                normalized_name = (
                    RitsumeiSourceBuilder.normalize_name(entity_name)
                    if entity_type == "役者"
                    else entity_name
                )
                key = (entity_type, normalized_name)

                label_id = -1
                if key in entity_name_to_id and len(entity_name_to_id[key]) > 0:
                    label_id = entity_name_to_id[key][0]
                    linked_count += 1

                query = {
                    "mention": entity_name,
                    "context_left": record.get("context_left", ""),
                    "context_right": record.get("context_right", ""),
                    "label_id": label_id,
                    "label": entity_name,
                    "label_title": entity_name,
                    "type": entity_type,
                    "world": "yakushahyoubanki",
                    "metadata": {
                        "source_file": record.get("source_file", ""),
                        "year": record.get("year_ad"),
                        "book_name": record.get("book_name", ""),
                        "start": record.get("start"),
                        "end": record.get("end"),
                    },
                }

                queries.append(query)

        # save queries
        with open(queries_file, "w", encoding="utf-8") as f:
            for query in queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")

        print(f"✓ Created {len(queries)} queries")
        print(f"✓ Successfully linked {linked_count} queries to entities")

        match_rate = matched_actors / total_actors if total_actors > 0 else 0
        link_rate = linked_count / len(queries) if queries else 0

        return {
            "entities_count": len(entities_list),
            "queries_count": len(queries),
            "matched_actors": matched_actors,
            "total_actors": total_actors,
            "match_rate": match_rate,
            "linked_queries": linked_count,
            "link_rate": link_rate,
        }

    @staticmethod
    def _add_theaters(entities_file: str) -> int:
        """Add the Wikipedia theater entities to the entity store."""
        print("\n" + "=" * 80)
        print("Adding Wikipedia theater entities")
        print("=" * 80 + "\n")

        # read existing entities, find the max ID
        max_numeric_id = 0
        with open(entities_file, encoding="utf-8") as f:
            for line in f:
                entity = json.loads(line.strip())
                if "numeric_id" in entity:
                    max_numeric_id = max(max_numeric_id, entity["numeric_id"])

        # create theater entities
        theater_entities = []
        for idx, theater in enumerate(RitsumeiSourceBuilder.THEATERS, start=1):
            entity = {
                "idx": f"theater:{idx}",
                "title": theater["title"],
                "text": theater["text"],
                "numeric_id": max_numeric_id + idx,
                "metadata": {
                    "source": "wikipedia",
                    "category": theater["category"],
                    "reading": theater["reading"],
                    "zamoto_family": theater.get("zamoto_family", ""),
                    "founded": theater.get("founded", ""),
                    "location_type": theater.get("location_type", "江戸"),
                    "wikipedia_url": theater.get("wikipedia_ref", ""),
                    "old_entity_mapping": theater.get("old_entity_titles", []),
                },
            }

            # add optional fields
            for key in ["original_location", "later_location", "later_name", "crest", "closed"]:
                if key in theater:
                    entity["metadata"][key] = theater[key]

            theater_entities.append(entity)

            print(f"Adding theater entity: {theater['title']}")
            print(f"  Reading: {theater['reading']}")
            print(f"  Zamoto family: {theater.get('zamoto_family', 'N/A')}")
            print(f"  Founded: {theater.get('founded', 'N/A')}")
            if theater.get("old_entity_titles"):
                print(f"  Alternate entities: {', '.join(theater['old_entity_titles'])}")

        # append to the entity file
        with open(entities_file, "a", encoding="utf-8") as f:
            for entity in theater_entities:
                f.write(json.dumps(entity, ensure_ascii=False) + "\n")

        print(f"\n✓ Successfully added {len(theater_entities)} Wikipedia theater entities")
        print(f"  Saved to: {entities_file}\n")

        return len(theater_entities)

    @staticmethod
    def split(
        queries_file: str,
        entities_file: str,
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """Split into train / valid / test sets."""
        random.seed(seed)

        # load queries
        with open(queries_file, encoding="utf-8") as f:
            queries = [json.loads(line.strip()) for line in f if line.strip()]

        # load entities
        with open(entities_file, encoding="utf-8") as f:
            entities = [json.loads(line.strip()) for line in f if line.strip()]

        print(f"📊 Total queries: {len(queries)}")
        print(f"📊 Total entities: {len(entities)}")

        # group queries by entity
        entity_queries = defaultdict(list)
        for query in queries:
            label_id = query.get("label_id", -1)
            entity_queries[label_id].append(query)

        # split entity IDs
        entity_ids = list(entity_queries.keys())
        random.shuffle(entity_ids)

        n_total = len(entity_ids)
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)

        train_entity_ids = set(entity_ids[:n_train])
        dev_entity_ids = set(entity_ids[n_train : n_train + n_dev])
        test_entity_ids = set(entity_ids[n_train + n_dev :])

        # assign queries
        train_queries = []
        dev_queries = []
        test_queries = []

        for entity_id, queries_list in entity_queries.items():
            if entity_id in train_entity_ids:
                train_queries.extend(queries_list)
            elif entity_id in dev_entity_ids:
                dev_queries.extend(queries_list)
            else:
                test_queries.extend(queries_list)

        # create the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # save the datasets
        with open(Path(output_dir) / "train.jsonl", "w", encoding="utf-8") as f:
            for query in train_queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")

        with open(Path(output_dir) / "valid.jsonl", "w", encoding="utf-8") as f:
            for query in dev_queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")

        with open(Path(output_dir) / "test.jsonl", "w", encoding="utf-8") as f:
            for query in test_queries:
                f.write(json.dumps(query, ensure_ascii=False) + "\n")

        print(f"\n✓ Train set: {len(train_queries)} ({len(train_entity_ids)} unique entities)")
        print(f"✓ Validation set: {len(dev_queries)} ({len(dev_entity_ids)} unique entities)")
        print(f"✓ Test set: {len(test_queries)} ({len(test_entity_ids)} unique entities)")

        return {
            "train": len(train_queries),
            "dev": len(dev_queries),
            "test": len(test_queries),
            "train_entities": len(train_entity_ids),
            "dev_entities": len(dev_entity_ids),
            "test_entities": len(test_entity_ids),
        }

    @staticmethod
    def _step1(
        csv_dir: str, output_file: str, start_row: int = 0, end_row: int | None = None
    ) -> list[dict]:
        """Step 1: parse and merge CSV files."""
        print("\n" + "=" * 80)
        print("Step 1: Parse and merge CSV files")
        print("=" * 80 + "\n")

        records = RitsumeiSourceBuilder.merge_csvs(csv_dir, output_file, start_row, end_row)
        return records

    @staticmethod
    def _step2(
        merged_file: str,
        entities_file: str,
        queries_file: str,
        cache_file: str | None = None,
        add_wiki_theaters: bool = False,
    ) -> dict:
        """Step 2: build the entity store and queries."""
        print("\n" + "=" * 80)
        print("Step 2: Build entity store and queries")
        print("=" * 80 + "\n")

        stats = RitsumeiSourceBuilder.build_entities_and_queries(
            merged_file, entities_file, queries_file, cache_file
        )

        if add_wiki_theaters:
            wiki_count = RitsumeiSourceBuilder._add_theaters(entities_file)
            stats["entities_count"] += wiki_count
            stats["wiki_theaters_added"] = wiki_count

        return stats

    @staticmethod
    def _step3(
        queries_file: str,
        entities_file: str,
        output_dir: str,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """Step 3: split into train / valid / test sets."""
        print("\n" + "=" * 80)
        print("Step 3: Split into train/validation/test sets")
        print("=" * 80 + "\n")

        stats = RitsumeiSourceBuilder.split(
            queries_file, entities_file, output_dir, train_ratio, dev_ratio, test_ratio, seed
        )
        return stats

    @staticmethod
    def run(
        csv_dir: str,
        output_dir: str,
        cache_file: str | None = None,
        add_wiki_theaters: bool = False,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict:
        """Full pipeline: from CSV to the final dataset."""
        print("\n" + "=" * 80)
        print("役者評判記 dataset build - full pipeline")
        if add_wiki_theaters:
            print("(including Wikipedia theater entities)")
        print("=" * 80 + "\n")

        # create the output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # intermediate file paths
        merged_file = output_path / "step1_records.json"
        entities_file = output_path / "entities.jsonl"
        queries_file = output_path / "queries.jsonl"

        # Step 1: merge CSVs
        print(f"📂 Input: {csv_dir}")
        print(f"📂 Output: {output_dir}\n")

        records = RitsumeiSourceBuilder._step1(csv_dir, str(merged_file))

        # Step 2: build entities
        stats_step2 = RitsumeiSourceBuilder._step2(
            str(merged_file), str(entities_file), str(queries_file), cache_file, add_wiki_theaters
        )

        # Step 3: split the dataset
        stats_step3 = RitsumeiSourceBuilder._step3(
            str(queries_file),
            str(entities_file),
            str(output_dir),
            train_ratio,
            dev_ratio,
            test_ratio,
            seed,
        )

        # summary statistics
        print("\n" + "=" * 80)
        print("✓ Full pipeline complete!")
        print("=" * 80)
        print("\n📊 Final stats:")
        print(f"  Total entities: {stats_step2['entities_count']}")
        if add_wiki_theaters and "wiki_theaters_added" in stats_step2:
            print(f"    of which Wikipedia theaters: {stats_step2['wiki_theaters_added']}")
        print(f"  Total queries: {stats_step2['queries_count']}")
        print(f"  Match rate: {stats_step2['match_rate'] * 100:.1f}%")
        print(f"\n  Train set: {stats_step3['train']} ({stats_step3['train_entities']} unique entities)")
        print(f"  Validation set: {stats_step3['dev']} ({stats_step3['dev_entities']} unique entities)")
        print(f"  Test set: {stats_step3['test']} ({stats_step3['test_entities']} unique entities)")
        print(f"\n📁 Output directory: {output_dir}")
        print("  ├── entities.jsonl      - entity store")
        print("  ├── train.jsonl         - train set")
        print("  ├── valid.jsonl         - validation set")
        print("  └── test.jsonl          - test set")
        print()

        return {"step1": {"records": len(records)}, "step2": stats_step2, "step3": stats_step3}


#
# Verbatim from the former build_yakushahyoubanki_dataset.py (now merged in). NOT
# rewritten: its build_entities_and_queries queries a live DB whose result cannot be
# reproduced offline (an incomplete cache, 3s/request, O(actors x candidates) detail
# lookups that time out even fully cached), so unlike every pass below it has no
# frozen artifact to diff against. Left byte-for-byte behind its own DB client;
# `YakusyaBuilder.from_csv` calls run_full_pipeline here.
# ============================================================================


# --- context re-cut (page-level, deterministic) ------------------------------
# The daime records carry exact metadata.start/end offsets into their source page,
# so widening the window is a slice, not a fuzzy relocation: page[s-W:s] / page[e:e+W].
# Everything else — label, candidates, split membership, the injected [年:..|書:..]
# token — survives byte-for-byte.

_JSON_DIR = os.path.join(_ROOT, "data", "dataset", "yakusya_annotated_data", "json")
# The trailing " [年:..|書:..] " token that inject_year put before the mention. Kept
# verbatim rather than rebuilt: its book field is sometimes "?" and cannot be redone.
_TOKEN_RE = re.compile(r"\s*\[年:[^\]]*\]\s*$")
_page_cache = {}


def _get_page(book):
    """Source page text for a book, or None. Exact filename only — the half- and
    full-width paren variants are DIFFERENT books and must not be merged."""
    if book not in _page_cache:
        try:
            with open(os.path.join(_JSON_DIR, f"{book}.json"), encoding="utf-8") as f:
                _page_cache[book] = json.load(f)[0]["data"]["text"]
        except Exception:
            _page_cache[book] = None
    return _page_cache[book]


def _sp(t):
    """The whitespace normalisation the original build applied."""
    return t.replace("\n", " ").replace("\r", " ").replace("　", " ")


def recut_record(r, win):
    """-> (record, 'aligned' | 'passthrough').

    A record whose offsets do not align (page[s:e] != mention — book file missing,
    mojibake source, or a ruby-marked page version) passes through UNCHANGED, and
    identically at every W, so it can never bias a w100/w200/w300 comparison."""
    md = r.get("metadata") or {}
    s, e, book = md.get("start"), md.get("end"), md.get("book_name")
    men = r.get("mention", "")
    t = _get_page(book) if book else None
    if t is None or s is None or e is None or t[s:e] != men:
        return r, "passthrough"
    m = _TOKEN_RE.search(r.get("context_left", "") or "")
    return dict(
        r,
        context_left=_sp(t[max(0, s - win) : s]) + (m.group(0) if m else ""),
        context_right=_sp(t[e : e + win]),
    ), "aligned"


# ============================================================================
# shared: tag multi-candidate mentions (same surface -> >=2 distinct gold)
# ============================================================================


def _ambiguous_surfaces(rows):
    """Surfaces that link to >=2 distinct gold ids within `rows`."""
    s = collections.defaultdict(set)
    for r in rows:
        s[r.get("mention", "")].add(r.get("label_id"))
    return {k for k, v in s.items() if len(v) >= 2}


def _apply_tag(path, surfaces, tag):
    """Clean recompute of `tag` on a jsonl file: drop the tag everywhere, re-add
    it to rows whose surface is ambiguous. Other tags (e.g. daime) preserved."""
    rows = [json.loads(l) for l in open(path)]
    n = 0
    for r in rows:
        md = r.get("metadata") or {}
        tags = set(md.get("tags") or [])
        tags.discard(tag)
        if r.get("mention", "") in surfaces:
            tags.add(tag)
            n += 1
        if tags:
            md["tags"] = sorted(tags)
        elif "tags" in md:
            del md["tags"]
        r["metadata"] = md
    _write_jsonl(rows, path)
    return n


# ============================================================================
# DatasetBuilder: raw sources -> a materialized dir -> Dataset
#
# One builder per corpus family, because the families share nothing: yakusya starts
# from scanned CSV plus a live person DB, HIPE from a git repo of TSV, mahanama from
# CoNLL-U. What they DO share is the far end — every one lands in the same on-disk
# schema, and `build()` hands back a `Dataset`. That is the seam this class draws.
#
# The passes themselves are the module-level build_* functions above, unchanged and
# individually verified byte-for-byte against the artifacts on disk. This layer only
# names which passes a corpus needs and in what order; it does not reimplement them.
# ============================================================================


class DatasetBuilder:
    """Base: how one corpus goes from its raw sources to a `Dataset`."""

    @property
    def path(self):
        return config.DATASETS.get(self.name) or config.dataset_data_dir(self.name)

    def exists(self):
        return os.path.exists(os.path.join(self.path, "test.jsonl"))

    def materialize(self, **kw):
        """Run the passes; return the built directory. Implemented per corpus."""
        raise NotImplementedError

    def build(self, *, force=False, **kw):
        """-> Dataset. Builds only if absent, unless `force`.

        Building is expensive and often needs the network (the Ritsumeikan DB, the
        HIPE repo, Wikidata), while the artifacts are stable — so the default is to
        return what is already on disk."""
        if force or not self.exists():
            self.materialize(**kw)
        return Dataset.from_dir(self.path, name=self.name)

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} {'built' if self.exists() else 'absent'}>"


class YakusyaBuilder(DatasetBuilder):
    """役者評判記: scanned CSV + the Ritsumeikan person DB -> a linked corpus.

    Each pass below was one script, chained by hand: every one hardcoded its own
    SRC/DST, so re-running any of them could only write back over what it read —
    which also made them untestable without clobbering the corpus. They take paths
    as arguments here instead.

    A chain, not a step, and each link is its own registered variant:

        raw CSV --(DB)--> base --clean--> --fix--> --relink--> --year--> relinked_yr

    `stage` picks where to stop. `arc` branches off `clean` rather than continuing
    the chain — it is the Ritsumeikan-only slice, not a further repair.

    WHERE relinked_yr_v1 AND _v2 COME FROM — the chain above stops at relinked_yr
    (1085 test mentions), but the corpus everything downstream is built on is
    relinked_yr_v2, and neither v1 nor v2 is reachable from here:

      relinked_yr --(recovery, NOT in this repo)--> v1 (1254) --dedup--> v2 (4885 ents)

    v1 is relinked_yr plus 169 test mentions that the source build had dropped (gold
    de-duplication and cleaning discard records) and that a set of recovery passes
    later won back, each from a different cue — a surface name with exactly one KB
    candidate (556), a multi-generation name pinned by year x kaimeihyou period to a
    single 代目 (468+149), a bare given name resolved to its one possible full name
    then dated (45), and 266 hand-reviewed from a worksheet. Only CONFIDENT hits were
    taken; ambiguous ones went to a human. Their outputs survive as the
    `recovered_*.jsonl` files inside v1/v2 and account for all 169, exactly.

    What is missing is the step that MERGED those files back into the splits — it was
    never in the codebase, so v1 cannot be rebuilt from raw. The recovery scripts
    themselves were deleted (2026-07-17) once that was established: with the merge
    step gone they could only regenerate their own side files, never the dataset.
    v1/v2 are frozen artifacts on disk; `DaimeEvalBuilder` reads v2 and reproduces
    the 2214 exactly, so nothing downstream depends on rebuilding them.
    See src_backup_20260716/build/recover/ for the deleted passes.

    KAIMEIHYOU ON A 〈1〉 IS THE WHOLE LINEAGE, NOT THE PERSON — a latent trap that
    only stays latent while the corpus ends in 1776. The Ritsumeikan/Wikipedia
    record for a first generation tends to list the entire 名跡 succession chain,
    so the table lands on 〈1〉 carrying names its later generations held, modern
    ones included: 松本幸四郎〈1〉 (d. 1730) holds 松本金太郎(1979-1981),
    市川染五郎(1981-2017), 松本幸四郎(2018-) — the tenth generation's career, who
    is himself a separate entity here. Same shape on 市川団十郎〈1〉 (d. 1704 ->
    2023), 中村勘三郎〈1〉 (d. 1658 -> 2012), 岩井半四郎〈1〉 (d. 1699 -> 2011).
    Measured 2026-07-17: 108 of the 1949 entities with both a kaimeihyou and a
    lifespan contradict it (5.5%); of the 26 off by >100 years, 23 are a 〈1〉.

    It costs nothing TODAY because the collapse only reads a period that brackets
    the mention's year (`_collapse_daime`), test years run 1688-1776, and the
    contamination sits at 1900+. Across all three splits it fires 12 times total
    (7 of 2214 on daime_eval_full, 4 of 1254 on relinked_yr_v2, 1 of 230 on
    daime_ft_w100) and misleads on none: to actually break a prediction the bad
    period would have to resolve to the mention's own surface form, and instead it
    resolves to some other generation's name, which excludes the candidate anyway.
    Left unfixed deliberately — the repair is in PART 0's parse (split the
    succession chain per generation), which is kept verbatim, and the payoff is
    0/2214. EXTENDING THE CORPUS PAST ~1900 REMOVES THE ONLY THING MAKING THIS
    HARMLESS: the modern rows would start bracketing real mention years, and a
    〈1〉 would answer to a name only its 〈10〉 ever held."""

    # The 5 theaters whose mentions the original build truncated.

    # OCR noise the scan left in mentions and titles: a marker glued to a
    # parenthesised id (☆(1), ？(12a)), a bare (0a), or a stray placeholder char.

    _KANA, _DAIME, _CLOSE, _YEAR = 2, 4, 8, 16  # match flags, as the original build
    _OCR_PAT = re.compile(
        r"[☆★＄＆◯●＝＋〇○]\(\d+[a-z]?\)"
        r"|[？?]\(\d+[a-z]?\)"
        r"|\(\d+[a-z]?\)"
        r"|[☆★＄＆＝]"
    )
    _THEATERS = {
        10001: "中村座",
        10002: "市村座",
        10003: "森田座",
        10004: "小川座",
        10005: "三升座",
    }

    CHAIN = ("clean", "fix", "relinked", "relinked_yr")

    def __init__(self, stage="relinked_yr"):
        if stage not in self.CHAIN + ("arc",):
            raise ValueError(f"stage must be one of {self.CHAIN + ('arc',)}")
        self.name = stage

    def materialize(self, csv_dir=None, cache_file=None, **_):
        csv_dir = csv_dir or config.DATASETS["raw_csv"] + "/csv"
        base = config.DATASETS["ja"]
        if not os.path.exists(os.path.join(base, "entities.jsonl")):
            # PART 0: the only pass that reaches outside — and the reason `build()`
            # prefers what is on disk. Kept verbatim, see build_from_csv.
            YakusyaBuilder.from_csv(csv_dir, base, cache_file=cache_file)
        clean = config.DATASETS["clean"]
        YakusyaBuilder.clean(base, clean)
        if self.name == "clean":
            return clean
        if self.name == "arc":
            YakusyaBuilder.arc(clean, self.path)
            return self.path
        fix = config.DATASETS["fix"]
        YakusyaBuilder.fix(clean, fix)
        if self.name == "fix":
            return fix
        relinked = config.DATASETS["relinked"]
        YakusyaBuilder.relinked(fix, relinked)
        if self.name == "relinked":
            return relinked
        YakusyaBuilder.year_injected(relinked, self.path)
        return self.path

    @staticmethod
    def _fix_theater(rec):
        """-> (rec, fixed?). The original build cut the 座 off a theater mention and
        left it heading the right context: mention="中村" + context_right="座」…" for
        gold 中村座. Move it back.

        Handles both record shapes: the dataset one (context_right / label_id) and the
        nel_dataset one (right_context / ans_id)."""
        is_nel = "left_context" in rec
        ck = "right_context" if is_nel else "context_right"
        gk = "ans_id" if is_nel else "label_id"
        gid = rec.get(gk)
        if gid not in YakusyaBuilder._THEATERS:
            return rec, False
        cr, mention = rec.get(ck, "") or "", rec.get("mention", "") or ""
        if "座" in mention or not cr.startswith("座"):
            return rec, False  # already whole, or not the truncation
        rec = dict(rec, mention=mention + "座")
        rec[ck] = cr[1:]
        if not is_nel and rec.get("label_title", "") == mention:
            rec["label_title"] = YakusyaBuilder._THEATERS[gid]
        md = rec.get("metadata", {}) or {}
        if md and isinstance(md.get("end"), int):
            rec["metadata"] = dict(md, end=md["end"] + 1)  # mention grew a char
        return rec, True

    @staticmethod
    def clean(src, dst):
        """yakusya_with_theaters -> clean: undo the theater-mention truncation.

        entities.jsonl is copied verbatim — only mentions were ever truncated."""
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, "entities.jsonl"), os.path.join(dst, "entities.jsonl"))
        for split in ("train.jsonl", "valid.jsonl", "test.jsonl", "nel_dataset.jsonl"):
            sp = os.path.join(src, split)
            if not os.path.exists(sp):
                continue
            n = fixed = 0
            with open(sp) as fin, open(os.path.join(dst, split), "w") as fout:
                for line in fin:
                    rec, ok = YakusyaBuilder._fix_theater(json.loads(line))
                    n, fixed = n + 1, fixed + ok
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [clean] {split}: {n} rows, {fixed} theater mentions repaired")
        print(f"  [clean] -> {dst}")

    @staticmethod
    def arc(src, dst):
        """clean -> arc: keep only what the Ritsumeikan DB covers.

        Entities narrow to metadata.source == 'ritsumei_PD' (~3244 of 5000) and mentions
        to those linking into that set — an ARC-only slice for actor disambiguation
        proper, without the corpus name-stubs."""
        os.makedirs(dst, exist_ok=True)
        ents = _load_jsonl(os.path.join(src, "entities.jsonl"))
        arc = [e for e in ents if (e.get("metadata") or {}).get("source") == "ritsumei_PD"]
        ids = {int(e["numeric_id"]) for e in arc}
        _write_jsonl(arc, os.path.join(dst, "entities.jsonl"))
        print(f"  [arc] entities: {len(ents)} -> {len(arc)}")
        for split, key in (
            ("train.jsonl", "label_id"),
            ("valid.jsonl", "label_id"),
            ("test.jsonl", "label_id"),
            ("nel_dataset.jsonl", "ans_id"),
        ):
            sp = os.path.join(src, split)
            if not os.path.exists(sp):
                continue
            n = kept = 0
            with open(sp) as fin, open(os.path.join(dst, split), "w") as fout:
                for line in fin:
                    n += 1
                    if json.loads(line).get(key) in ids:
                        kept += 1
                        fout.write(line)
            print(f"  [arc] {split}: {n} -> {kept}")
        print(f"  [arc] -> {dst}")

    @staticmethod
    def _clean_ocr(s):
        return YakusyaBuilder._OCR_PAT.sub("", s).strip() if s else s

    @staticmethod
    def _clean_title(s):
        """_clean_ocr plus the surname/given-name separator. Dropping U+3000 collapses
        817 spurious "mention not in title" mismatches — the corpus writes 山下八百蔵
        while the DB title is 山下　八百蔵."""
        return YakusyaBuilder._clean_ocr(s).replace("　", "")

    @staticmethod
    def _cue_for(entity):
        """A short parenthetical that tells same-titled entities apart.

        37 titles cover 80 ids: real, distinct people of one lineage, indistinguishable
        from the title alone. Prefer the most discriminating cue the description offers
        — sub-lineage (大坂系), then 俳名, then 屋号, then 通名 — and fall back to the id."""
        text = entity.get("text", "") or ""
        pats = (
            r"\(([^()（）\n]{0,3}系)\)",
            r"（([^()（）\n]{0,3}系)）",  # lineage
            r"俳名・狂歌名[:：]\s*([^\s（(屋号家系\n]+)",  # haiku name
            r"屋号[:：]\s*([^\s（(\n]+)",  # house name
            r"本名[:：]\s*【通】\s*([^\s\n]+)",
        )  # common name
        for i, p in enumerate(pats):
            m = re.search(p, text)
            if not m:
                continue
            cue = m.group(1).strip()
            if i == 2 and cue in ("—", "無", "？"):
                continue
            if cue and len(cue) <= 8:
                return cue
        return f"id{entity.get('numeric_id')}"

    @staticmethod
    def fix(src, dst):
        """clean -> fix: make the labels usable.

        Four repairs, in order:
          titles   strip OCR noise and U+3000 (mentions can then match titles)
          titles   append a cue to same-titled entities so they are distinguishable
          labels   rewrite label/label_title from entities.jsonl — some rows carried the
                   truncated mention text instead of the entity's title
          mentions strip OCR noise, drop rows that were nothing but noise, dedupe"""
        os.makedirs(dst, exist_ok=True)
        ents = _load_jsonl(os.path.join(src, "entities.jsonl"))
        n_ocr = n_sp = 0
        for e in ents:
            orig = e["title"]
            cleaned = YakusyaBuilder._clean_title(orig)
            if cleaned != orig:
                n_ocr += bool(YakusyaBuilder._OCR_PAT.search(orig))
                n_sp += "　" in orig and cleaned == orig.replace("　", "")
                e["title"] = cleaned
        print(f"  [fix] titles: {n_ocr} OCR-stripped, {n_sp} U+3000-stripped")

        by_title = collections.defaultdict(list)
        for e in ents:
            by_title[e["title"]].append(e)
        renamed = {}
        for title, group in by_title.items():
            if len(group) == 1:
                continue
            seen = set()
            for e in group:
                cue = YakusyaBuilder._cue_for(e)
                if cue in seen:  # two cues collided within the group
                    cue = f"{cue}#{e['numeric_id']}"
                seen.add(cue)
                renamed[int(e["numeric_id"])] = f"{title}({cue})"
        print(f"  [fix] {len(renamed)} same-titled entities given a distinguishing cue")

        out = [
            dict(e, title=renamed[int(e["numeric_id"])]) if int(e["numeric_id"]) in renamed else e
            for e in ents
        ]
        _write_jsonl(out, os.path.join(dst, "entities.jsonl"))
        canon_title = {int(e["numeric_id"]): e["title"] for e in out}

        for fn in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            rows, n_lab = [], 0
            for r in _load_jsonl(os.path.join(src, fn)):
                t = canon_title.get(r.get("label_id"))
                if t is not None and (r.get("label_title") != t or r.get("label") != t):
                    n_lab += 1
                    r["label_title"] = r["label"] = t
                rows.append(r)
            kept, seen, n_m, n_drop = [], set(), 0, 0
            for r in rows:
                old = r.get("mention", "")
                new = YakusyaBuilder._clean_ocr(old)
                if not new:  # the mention was pure OCR noise
                    n_drop += 1
                    continue
                if new != old:
                    n_m += 1
                    r["mention"] = new
                key = (
                    r.get("mention"),
                    r.get("label_id"),
                    (r.get("context_left", "") or "")[-80:],
                    (r.get("context_right", "") or "")[:80],
                )
                if key in seen:
                    n_drop += 1
                    continue
                seen.add(key)
                kept.append(r)
            _write_jsonl(kept, os.path.join(dst, fn))
            print(
                f"  [fix] {fn}: {n_lab} labels canonicalised, {n_m} mentions cleaned, "
                f"{n_drop} dropped -> {len(kept)}"
            )

        nel = os.path.join(src, "nel_dataset.jsonl")
        if os.path.exists(nel):
            rows, n = [], 0
            for r in _load_jsonl(nel):
                t = canon_title.get(r.get("ans_id"))
                if t is not None and r.get("output") != t:
                    n += 1
                    r["output"] = t
                rows.append(r)
            _write_jsonl(rows, os.path.join(dst, "nel_dataset.jsonl"))
            print(f"  [fix] nel_dataset.jsonl: {n} canonicalised")
        print(f"  [fix] -> {dst}")

    @staticmethod
    def year_injected(src, dst):
        """relinked -> relinked_yr: put the playbill's own year and book in front of
        the mention, as a "[年:1767|書:役者位下上(坂)] " token on the left context.

        This is what gives a model the query key the whole task turns on: the same
        stage name spans generations, and only the year says which one is speaking.

        The original script offered v0-v3 and va — type-aware injection, book dropout,
        family-candidate tokens, synthesized training rows. Only v0 (inject for every
        row) is kept: it is what `dataset_relinked_yr` on disk actually holds, and the
        others wrote to `dataset_relinked_yr_v1` / `_v2` — names that now mean something
        else entirely (the re-link iteration and its dedup), so running them would
        quietly overwrite the real thing."""
        os.makedirs(dst, exist_ok=True)
        ent_dst = os.path.join(dst, "entities.jsonl")
        if os.path.lexists(ent_dst):
            os.remove(ent_dst)
        os.symlink(os.path.abspath(os.path.join(src, "entities.jsonl")), ent_dst)
        for fn in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            p = os.path.join(src, fn)
            if not os.path.exists(p):
                continue
            rows = _load_jsonl(p)
            for r in rows:
                md = r.get("metadata") or {}
                year = md.get("year", "?")
                book = (md.get("book_name") or "").replace("|", "/")
                r["context_left"] = (
                    r.get("context_left") or ""
                ) + f" [年:{year if year else '?'}|書:{book}] "
            _write_jsonl(rows, os.path.join(dst, fn))
            print(f"  [year] {fn}: {len(rows)} rows tokened")
        print(f"  [year] -> {dst}")

    @staticmethod
    def _norm_name(name):
        """Strip whitespace, OCR marks, parentheticals and the 〈N〉 generation mark.

        NOT the same normaliser as the corpus builder's: that one also folds 代表名.
        The two lived under one name in separate files and would have collided on a
        merge — this is the year-relink flavour, used by relink and inject alike."""
        name = re.sub(r"\s+", "", name or "")
        name = re.sub(r"[☆★○●◎◇◆□■▽▼△▲]", "", name)
        name = re.sub(r"\([^)]*\)", "", name)
        name = re.sub(r"（[^）]*）", "", name)
        return re.sub(r"〈[^〉]*〉", "", name).strip()

    @staticmethod
    def _score_candidate(mention_text, year, ent):
        """-> (flags, distance-to-period-midpoint). Mirrors the corpus builder's
        calculate_match_score, so a re-link agrees with how the KB was matched."""
        md = ent.get("metadata", {})
        flags, best = 0, 10**9
        reading = md.get("reading", "")
        if reading:
            rn = YakusyaBuilder._norm_name(reading)
            if any(
                p.get("reading") and YakusyaBuilder._norm_name(p["reading"]) == rn
                for p in md.get("kaimeihyou") or []
            ):
                flags |= YakusyaBuilder._KANA
        m = re.search(r"〈(\d+)〉", mention_text)
        if m and md.get("daime") == m.group(1):
            flags |= YakusyaBuilder._DAIME
        if year:
            periods = md.get("kaimeihyou") or []
            birth, death = md.get("birth_year"), md.get("death_year")
            for p in periods:  # 1: strictly inside a period
                s, e = p.get("start_year"), p.get("end_year")
                if s and e and s <= year <= e:
                    flags |= YakusyaBuilder._YEAR
                    best = min(best, abs(year - (s + e) / 2))
                    break
            if not flags & YakusyaBuilder._YEAR:  # 2: within 50y of a period's end
                for p in periods:
                    s, e = p.get("start_year"), p.get("end_year")
                    if s and e and s <= year <= e + 50:
                        flags |= YakusyaBuilder._YEAR
                        best = min(best, abs(year - e))
                        break
            if not flags & YakusyaBuilder._YEAR:  # 3: within 10y of a boundary
                for p in periods:
                    s, e = p.get("start_year"), p.get("end_year")
                    if s and e and (abs(year - s) <= 10 or abs(year - e) <= 10):
                        flags |= YakusyaBuilder._CLOSE
                        best = min(best, min(abs(year - s), abs(year - e)))
                        break
            if not flags & (YakusyaBuilder._YEAR | YakusyaBuilder._CLOSE):  # 4: merely alive
                if birth and death and birth <= year <= death:
                    flags |= YakusyaBuilder._YEAR
                    best = min(best, abs(year - (birth + death) / 2))
        return flags, best

    @staticmethod
    def relinked(src, dst):
        """fix -> relinked: re-link every mention on year x kaimeihyou.

        The original build linked a mention to whichever candidate the DB returned
        first — a silent bug, since one stage name spans generations and only the
        playbill year says which held it. Only label_id / label / label_title
        change; entities are copied through."""
        os.makedirs(dst, exist_ok=True)
        ents = _load_jsonl(os.path.join(src, "entities.jsonl"))
        id2e = {int(e["numeric_id"]): e for e in ents}
        _write_jsonl(ents, os.path.join(dst, "entities.jsonl"))  # unchanged

        # (category, query_name) -> ids. query_name is the surface the build queried
        # the DB with, so this bucket can hold cross-family hits; they get filtered by
        # name_matches below.
        cand = collections.defaultdict(list)
        for e in ents:
            md = e.get("metadata", {})
            cand[
                (
                    md.get("category", ""),
                    md.get("query_name") or YakusyaBuilder._norm_name(e["title"]),
                )
            ].append(int(e["numeric_id"]))

        def name_matches(mnorm, ent):
            """The mention names this entity — as its title, or as one of the names it
            historically held (kaimeihyou). Anything else in the bucket is a stale
            DB-query artifact."""
            return YakusyaBuilder._norm_name(ent["title"]) == mnorm or any(
                YakusyaBuilder._norm_name(p.get("name", "")) == mnorm
                for p in (ent.get("metadata", {}).get("kaimeihyou") or [])
            )

        def best_for(mention, year, old_id):
            """-> chosen id. `None` year or a lone candidate means keep what we had."""
            old = id2e.get(old_id)
            if not old:
                return None
            md = old.get("metadata", {})
            ids = [
                c
                for c in cand.get(
                    (
                        md.get("category", ""),
                        md.get("query_name") or YakusyaBuilder._norm_name(old["title"]),
                    ),
                    [],
                )
                if name_matches(YakusyaBuilder._norm_name(mention), id2e[c])
            ]
            if old_id not in ids:
                ids.append(old_id)  # always keep the incumbent
            if len(ids) <= 1 or not year:
                return old_id
            scored = [(*YakusyaBuilder._score_candidate(mention, year, id2e[c]), c) for c in ids]
            # highest flags; then nearest the period midpoint; then lowest id
            scored.sort(key=lambda t: (-t[0], t[1], t[2]))
            return scored[0][2]

        gen_pat = re.compile(r"〈(\d+)〉")
        base_of = lambda i: (
            re.sub(r"\([^)]*\)$", "", gen_pat.sub("", id2e[i]["title"])).strip()
            if i in id2e
            else None
        )
        n = collections.Counter()
        for fn in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            rows = _load_jsonl(os.path.join(src, fn))
            for r in rows:
                old = int(r["label_id"])
                new = best_for(r.get("mention", ""), (r.get("metadata") or {}).get("year"), old)
                if new is None:
                    n["orphan_no_cand"] += 1
                    continue
                if new == old:
                    n["unchanged"] += 1
                    continue
                n["changed"] += 1
                n["gen_changed" if base_of(new) == base_of(old) else "base_changed"] += 1
                r["label_id"] = new
                r["label_title"] = r["label"] = id2e[new]["title"]
            _write_jsonl(rows, os.path.join(dst, fn))

        nel = os.path.join(src, "nel_dataset.jsonl")
        if os.path.exists(nel):
            rows = _load_jsonl(nel)
            for r in rows:
                old = int(r.get("ans_id", -1))
                yr = (r.get("metadata") or {}).get("year")
                if old not in id2e or not yr:
                    continue
                new = best_for(r.get("mention", "") or r.get("input", ""), yr, old)
                if new is not None and new != old:
                    r["ans_id"] = new
                    r["output"] = id2e[new]["title"]
            _write_jsonl(rows, os.path.join(dst, "nel_dataset.jsonl"))
        print(f"  [relink] {dict(n)}")
        print(f"  [relink] -> {dst}")

    @staticmethod
    def dedup(src, dst, mode="source"):
        """Apply entity_canon PHYSICALLY: drop duplicate person records and remap every
        gold to the canonical (min-id) one, so a trained model never sees the same
        person twice in its candidate list. v1 -> v2 is this at mode='source'.

        Pick the mode deliberately.
          source (default, safe to train on): merge only A = records sharing a source
            `idx`, i.e. literal copies. Verified to change 0 gold surface names and 0
            daime — what the model must produce is untouched.
          full (scoring only): merge A + B = cross-name same-person. Right for `eval
            --canon full`, wrong as training data: it rewrites ~986 gold names and ~587
            daime, collapsing each actor's distinct stage names into one — which is the
            very thing this task asks a model to tell apart."""
        os.makedirs(dst, exist_ok=True)
        ents = _load_jsonl(os.path.join(src, "entities.jsonl"))
        canon, stats = build_canon_map(ents, mode=mode)
        cid = lambda i: canon.get(int(i), int(i))
        print(
            f"  [dedup] entities {stats['total']} -> {stats['groups']} "
            f"(merged {stats['merged']}, mode={mode})"
        )

        kept = [e for e in ents if int(e["numeric_id"]) == cid(e["numeric_id"])]
        canon_ent = {int(e["numeric_id"]): e for e in kept}
        _write_jsonl(kept, os.path.join(dst, "entities.jsonl"))

        # nel_dataset candidates carry names, not ids, so they need a name -> id map.
        # Only names resolving to ONE canonical entity go in: strings like "中村　仲蔵〈1〉"
        # are shared by several different people, and collapsing candidates by such a
        # name would merge them. Those fall back to exact (name, text) below.
        groups = collections.defaultdict(set)
        for e in ents:
            for k in (e.get("title"), e.get("entity")):
                if k:
                    groups[k].add(cid(e["numeric_id"]))
        name2cid = {k: next(iter(v)) for k, v in groups.items() if len(v) == 1}

        for sp in ("train", "valid", "test"):
            p = os.path.join(src, f"{sp}.jsonl")
            if not os.path.exists(p):
                continue
            rows = _load_jsonl(p)
            ch = 0
            for r in rows:
                old = int(r["label_id"])
                if cid(old) != old:
                    r["label_id"] = cid(old)
                    ch += 1
            _write_jsonl(rows, os.path.join(dst, f"{sp}.jsonl"))
            print(f"  [dedup] {sp}.jsonl: {len(rows)} rows, {ch} golds remapped")

        nd = os.path.join(src, "nel_dataset.jsonl")
        if os.path.exists(nd):
            rows = _load_jsonl(nd)
            demoted = 0
            for r in rows:
                r["ans_id"] = cid(r.get("ans_id"))
                cands = r.get("candidates", []) or []
                seen, out = set(), []
                for c in cands:
                    nm = c.get("name")
                    ci = name2cid.get(nm)  # None if absent/ambiguous
                    key = ci if ci is not None else (nm, c.get("text"))
                    if key in seen:
                        continue
                    seen.add(key)
                    ce = canon_ent.get(ci) if ci is not None else None
                    if ce:  # a candidate may name a dropped duplicate; retitle it
                        c = dict(c, name=ce.get("title", nm))
                        if "text" in c:
                            c["text"] = ce.get("text", c["text"])
                    out.append(c)
                if len(cands) >= 2 and len(out) < 2:
                    demoted += 1
                # ans_idx must follow the dedup, or --multi_candidate_only desyncs
                r["ans_idx"] = next(
                    (i for i, c in enumerate(out) if name2cid.get(c.get("name")) == r["ans_id"]),
                    min(r.get("ans_idx", 0), max(0, len(out) - 1)),
                )
                r["candidates"] = out
            _write_jsonl(rows, os.path.join(dst, "nel_dataset.jsonl"))
            print(
                f"  [dedup] nel_dataset.jsonl: {len(rows)} rows, "
                f"{demoted} demoted multi->single by candidate dedup"
            )

        for f in sorted(os.listdir(src)):  # intermediate build artifacts
            if f.startswith("recovered_") and f.endswith(".jsonl"):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        print(f"  [dedup] -> {dst}")

    @staticmethod
    def from_csv(csv_dir, out_dir, cache_file=None):
        """PART 1 source: raw CSV + Ritsumeikan person DB -> the base corpus.

        Thin wrapper over run_full_pipeline (PART 0 above), which is kept verbatim
        rather than rewritten: that stage queries a live DB (3s/request, an incomplete cache, non-reproducible
        results), so unlike every other pass in this file it cannot be verified against a
        frozen artifact by md5 — and a silent slip in its 4-round year-match scoring would
        change which entity a mention links to with nothing to catch it. It stays behind
        its own module and its own tested DB client; this is only the entry point.

        Everything downstream of it — clean / fix / relink / year — IS inlined and
        checked, so `raw_to_base` chains straight into them."""
        return RitsumeiSourceBuilder.run(csv_dir, out_dir, cache_file=cache_file)

    @staticmethod
    def raw_to_base(csv_dir, raw_out, cache_file=None):
        """raw CSV -> a relinked_yr-style base, end to end.

        Part 1's DB-backed builder produces the raw corpus; the verified in-file passes
        then carry it clean -> fix -> relinked -> year.

        This stops at relinked_yr, NOT at relinked_yr_v1: the recovery pass that wins
        back the 169 dropped mentions never existed as code here. See the class
        docstring."""
        print("[stage 0] raw CSV -> base entities  (⚠ needs Ritsumeikan DB via ritsumei_client)")
        try:
            YakusyaBuilder.from_csv(csv_dir, raw_out, cache_file=cache_file)
        except Exception as e:
            print(
                f"  [stage 0] builder failed ({type(e).__name__}: {e}) — likely no DB "
                f"access. Aborting."
            )
            return False
        print("[stage 0] clean -> fix -> relinked -> year (verified in-file passes)")
        YakusyaBuilder.clean(raw_out, raw_out + "_clean")
        YakusyaBuilder.fix(raw_out + "_clean", raw_out + "_fix")
        YakusyaBuilder.relinked(raw_out + "_fix", raw_out + "_relinked")
        YakusyaBuilder.year_injected(raw_out + "_relinked", raw_out + "_yr")
        print(
            f"  [stage 0] -> {raw_out}_yr  (this is NOT relinked_yr_v1: the recovery\n          pass that wins back 169 dropped mentions is not in this repo —\n          see YakusyaBuilder's docstring)"
        )
        return raw_out + "_yr"


class DaimeEvalBuilder(DatasetBuilder):
    """代目 disambiguation: the 2214-mention eval set, the corpus this work is about.

    Starts from relinked_yr_v2 (v1 with entity_canon's same-source duplicates merged)
    and keeps only mentions whose gold resolves to one specific generation — so every
    item is a real, answerable generation question rather than an in/out-of-DB one."""

    name = "daime_eval_full"

    def materialize(self, base=None, work=None, **_):
        base = base or config.DS_RELINKED_YR_V2
        work = work or (self.path.rstrip("/") + "_multidaime_clean")
        DaimeEvalBuilder.multidaime_clean(base, work)
        DaimeEvalBuilder.daime_eval(base, DaimeEvalBuilder.probe(work), self.path)
        return self.path

    @staticmethod
    def _year_mdc(r):
        """multidaime_clean's year_of: metadata.year first, else context regex."""
        y = (r.get("metadata") or {}).get("year")
        if isinstance(y, int):
            return y
        m = re.search(
            r"年[:：](\d{3,4})",
            (r.get("context_left", "") or "") + (r.get("context_right", "") or ""),
        )
        return int(m.group(1)) if m else None

    @staticmethod
    def _consistent(e, yr):
        md = e["metadata"]
        by, dy = md.get("birth_year"), md.get("death_year")
        for p in md.get("kaimeihyou") or []:
            s, en = p.get("start_year"), p.get("end_year")
            if s and en and s <= yr <= en:
                return True
        if by and dy and by <= yr <= dy:
            return True
        if by and not dy and by <= yr <= by + 90:
            return True
        return False

    @staticmethod
    def multidaime_clean(base_dir, dst_dir):
        """Clean, expanded multi-daime probe from all splits of `base_dir`
        (test > valid > train priority, no re-shuffle, leak-drop vs base train)."""
        os.makedirs(dst_dir, exist_ok=True)
        ents = _load_jsonl(os.path.join(base_dir, "entities.jsonl"))
        by_id = {int(e["numeric_id"]): e for e in ents}
        base2ids = collections.defaultdict(list)
        for e in ents:
            base2ids[base_name(e["title"])].append(int(e["numeric_id"]))

        def key(r):
            return (
                r.get("mention", ""),
                int(r["label_id"]),
                (r.get("context_left", "") or "")[-80:],
            )

        train_keys = {key(r) for r in _load_jsonl(os.path.join(base_dir, "train.jsonl"))}

        def clean_split(split, taken):
            rows = _load_jsonl(os.path.join(base_dir, f"{split}.jsonl"))
            multi = [
                r
                for r in rows
                if len(
                    {
                        by_id[i]["metadata"].get("daime")
                        for i in base2ids.get(base_name(r.get("mention", "")), [])
                    }
                )
                >= 2
            ]
            kept, n_leak, n_fix, n_noise, n_drop = [], 0, 0, 0, 0
            for r in multi:
                if key(r) in taken:
                    n_leak += 1
                    continue
                if split != "train" and key(r) in train_keys:
                    n_leak += 1
                    continue
                g = int(r["label_id"])
                e = by_id.get(g)
                yr = DaimeEvalBuilder._year_mdc(r)
                b = base_name(r.get("mention", ""))
                by = e["metadata"].get("birth_year") if e else None
                stub = bool(e) and "カラ名跡" in e.get("text", "")
                if not (stub or (e and by and yr and by > yr)):
                    kept.append(r)
                    continue
                if not stub and by and yr and 0 < by - yr <= 2:
                    n_noise += 1
                    kept.append(r)
                    continue
                others = [
                    i
                    for i in base2ids.get(b, [])
                    if i != g and yr and DaimeEvalBuilder._consistent(by_id[i], yr)
                ]
                if len(others) == 1:
                    r = dict(
                        r,
                        label_id=others[0],
                        label_title=by_id[others[0]]["title"],
                        metadata={**(r.get("metadata") or {}), "gold_relinked_from": g},
                    )
                    n_fix += 1
                    kept.append(r)
                else:
                    n_drop += 1
            taken.update(key(r) for r in kept)
            _write_jsonl(kept, os.path.join(dst_dir, f"{split}.jsonl"))
            print(
                f"  [mdc:{split}] pool={len(multi)} leak/dup-drop={n_leak} "
                f"unanswerable-drop={n_drop} relinked={n_fix} noise-kept={n_noise} -> kept={len(kept)}"
            )

        taken = set()
        for split in ("test", "valid", "train"):  # test > valid > train priority
            clean_split(split, taken)
        _write_jsonl(ents, os.path.join(dst_dir, "entities.jsonl"))
        print(f"  [mdc] -> {dst_dir}/  (entities: {len(ents)})")

    @staticmethod
    def _year_de(r):
        """daime_eval's year_of: context_left regex only (kept distinct on purpose)."""
        m = re.search(r"年[:：](\d{3,4})", (r.get("context_left", "") or ""))
        return int(m.group(1)) if m else None

    @staticmethod
    def _active_window(e):
        m = e["metadata"]
        ys = []
        for p in m.get("kaimeihyou") or []:
            if p.get("start_year"):
                ys.append(p["start_year"])
            if p.get("end_year"):
                ys.append(p["end_year"])
        if m.get("birth_year"):
            ys.append(m["birth_year"])
        if m.get("death_year"):
            ys.append(m["death_year"])
        return (min(ys), max(ys)) if ys else None

    @staticmethod
    def _name_at_year(e, Y):
        """The base stage-name (kaimeihyou 名) this entity actually held in year Y."""
        for p in e["metadata"].get("kaimeihyou") or []:
            s, en = p.get("start_year"), p.get("end_year")
            if s and en and s <= Y <= en:
                return base_name(p.get("name") or "")
        return None

    @staticmethod
    def daime_eval(base_dir, probe_rows, dst_dir):
        """Resolve every actor mention's gold to a specific ritsumei generation.
        Stamps gold_resolved_from + resolve_by (incl. keep_person) and carries
        origin_split/heldout, matching the official daime_eval schema."""
        os.makedirs(dst_dir, exist_ok=True)
        ents = _load_jsonl(os.path.join(base_dir, "entities.jsonl"))
        byid = {int(e["numeric_id"]): e for e in ents}
        canon, _ = build_canon_map(ents, mode="full")
        cid = lambda i: canon.get(int(i), int(i))

        rit_base = collections.defaultdict(set)  # base name -> canonical ritsumei ids
        rit_reading = collections.defaultdict(set)  # reading   -> canonical ritsumei ids
        for e in ents:
            if e["idx"].startswith("ritsumei"):
                rit_base[base_name(e["title"])].add(cid(e["numeric_id"]))
                rit_reading[norm(e["metadata"].get("reading"))].add(cid(e["numeric_id"]))

        def is_person_stub(e):
            return (
                not e["idx"].startswith("ritsumei")
                and e["idx"].split(":")[0] not in NONPERSON
                and cid(e["numeric_id"]) == int(e["numeric_id"])
            )

        kept = []
        n = collections.Counter()
        for r in probe_rows:
            if r.get("type") != "役者":
                n["drop_non_actor"] += 1
                continue
            g = byid.get(int(r["label_id"]))
            if g is None:
                n["drop_missing"] += 1
                continue
            if g["idx"].startswith("ritsumei") or cid(g["numeric_id"]) != int(g["numeric_id"]):
                r = dict(
                    r,
                    label_id=cid(g["numeric_id"]),
                    metadata={
                        **(r.get("metadata") or {}),
                        "gold_resolved_from": int(g["numeric_id"]),
                        "resolve_by": "person",
                    },
                )
                n["keep_person"] += 1
                kept.append(r)
                continue
            if not is_person_stub(g):
                n["drop_nonperson_gold"] += 1
                continue
            b = base_name(g["title"])
            gens = rit_base.get(b)
            if not gens:
                n["drop_out_of_db"] += 1
                continue
            rd = norm(g["metadata"].get("reading"))
            new = why = None
            if rd and len(rit_reading.get(rd, set())) == 1:
                new = next(iter(rit_reading[rd]))
                why = "reading"
            else:
                Y = DaimeEvalBuilder._year_de(r)
                active = [
                    x
                    for x in gens
                    if (w := DaimeEvalBuilder._active_window(byid[x]))
                    and w[0] - 5 <= (Y or -9999) <= w[1] + 5
                ]
                if Y and len(active) == 1:
                    new = active[0]
                    why = "year"
                elif Y and len(active) >= 2:
                    # stricter: exact stage-name held at Y unique to one generation
                    mb = base_name(r.get("mention", ""))
                    hit = [x for x in active if DaimeEvalBuilder._name_at_year(byid[x], Y) == mb]
                    if len(hit) == 1:
                        new = hit[0]
                        why = "name_year"
            if new is None:
                n["drop_unresolved"] += 1
                continue
            r = dict(
                r,
                label_id=new,
                label_title=byid[new]["title"],
                metadata={
                    **(r.get("metadata") or {}),
                    "gold_resolved_from": int(g["numeric_id"]),
                    "resolve_by": why,
                },
            )
            n["relinked_" + why] += 1
            kept.append(r)

        _write_jsonl(kept, os.path.join(dst_dir, "test.jsonl"))
        _write_jsonl(ents, os.path.join(dst_dir, "entities.jsonl"))
        print(f"  [daime_eval] source={len(probe_rows)} probe samples")
        for k, v in sorted(n.items()):
            print(f"    {k}: {v}")
        print(f"  [daime_eval] kept {len(kept)} -> {dst_dir}/test.jsonl")
        return kept

    @staticmethod
    def probe(mdc_dir):
        """Concatenate multidaime_clean train/valid/test into one probe, tagging
        metadata.origin_split + heldout (=valid/test)."""
        probe = []
        for sp in ("train", "valid", "test"):
            for r in _load_jsonl(os.path.join(mdc_dir, f"{sp}.jsonl")):
                md = {
                    **(r.get("metadata") or {}),
                    "origin_split": sp,
                    "heldout": sp in ("valid", "test"),
                }
                probe.append({**r, "metadata": md})
        return probe


class HipeBuilder(DatasetBuilder):
    """HIPE-2022: one of the 12 (benchmark, language) EL corpora.

    Five passes, because the shipped TSV is a long way from a linked dataset:
        TSV -> NEL jsonl -> KB (QID -> numeric_id + entities.jsonl)
            -> train/valid splits -> fine_type -> Wikidata evidence

    The last two are what the evidence work needs: `fine_type` carries the corpus's
    own NE-FINE-LIT tag (a mention typed `loc.adm.town`, not merely `LOC`), and the
    evidence pass fetches each entity's P31 type and life span. Both were absent from
    the original conversion, which cost hipe2020_de about 8 points."""

    REPO = "https://github.com/hipe-eval/HIPE-2022-data.git"
    #: letemps is NER-only — no entity links, so it is not an EL corpus.
    EL_DATASETS = ("hipe2020", "newseye", "ajmc", "sonar", "topres19th")
    QID_RE = re.compile(r"^Q\d+$")

    #: the 12 (benchmark, language) pairs this project builds
    DS = {
        "ajmc_en": ("ajmc", "en"),
        "ajmc_de": ("ajmc", "de"),
        "ajmc_fr": ("ajmc", "fr"),
        "hipe2020_en": ("hipe2020", "en"),
        "hipe2020_de": ("hipe2020", "de"),
        "hipe2020_fr": ("hipe2020", "fr"),
        "newseye_de": ("newseye", "de"),
        "newseye_fi": ("newseye", "fi"),
        "newseye_fr": ("newseye", "fr"),
        "newseye_sv": ("newseye", "sv"),
        "sonar_de": ("sonar", "de"),
        "topres19th_en": ("topres19th", "en"),
    }

    NEL_SRC = os.path.join(_ROOT, "data", "dataset")
    HIPE_DATA = os.path.join(NEL_SRC, "HIPE-2022-data", "data")
    #: v2.1 ships the labelled TEST release; v1.0 has train/dev only, and its
    #: documents are disjoint from test — joining fine types against it matches 0%.
    RAW_V21 = os.path.join(HIPE_DATA, "v2.1")
    SPLIT_WIN = 400  # context chars each side, matching the existing test.jsonl
    DEV_AS_TRAIN = False  # for corpora that ship no train split

    # --- Wikidata ---
    WD_API = "https://www.wikidata.org/w/api.php"
    UA = (
        "YakushaNEL-research/1.0 (academic; contact "
        f"{os.environ.get('YAKUSYA_CONTACT', 'https://github.com/BohaoWu/Yakushahyoubanki_NEL')})"
    )
    QID_CACHE = os.path.join(NEL_SRC, ".wikidata_cache.json")
    CLAIM_CACHE = os.path.join(NEL_SRC, ".wikidata_claims_cache.json")
    LABEL_CACHE = os.path.join(NEL_SRC, ".wikidata_typelabels_cache.json")
    DATE_PROPS = {
        "P569": "birth_year",
        "P570": "death_year",
        "P571": "inception_year",
        "P577": "publication_year",
    }
    _qid_cache = {}

    def __init__(self, bench, lang):
        self.bench, self.lang = bench, lang
        self.name = f"{bench}_{lang}"

    def materialize(self, window=400, download=True, **_):
        raw = config.DS_HIPE_RAW
        if download and not os.path.isdir(os.path.join(raw, "data", "v2.1")):
            HipeBuilder.download(raw)
        self.to_nel(self.bench, window=window, lang=self.lang)  # TSV -> NEL jsonl
        self.to_kb(self.bench, self.lang)  # QID -> numeric_id + KB
        self.add_splits(self.bench, self.lang)  # + train/valid
        self.add_fine_types(self.name)  # + metadata.fine_type
        self.add_wikidata_evidence(self.name)  # + P31 type, life span
        return self.path

    @staticmethod
    def download(dst, shallow=True):
        """git clone HIPE-2022-data into `dst` (skip if already present)."""
        if os.path.isdir(os.path.join(dst, "data")):
            print(f"[hipe] already present: {dst}")
            return True
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        cmd = ["git", "clone"] + (["--depth", "1"] if shallow else []) + [HipeBuilder.REPO, dst]
        print(f"  $ {' '.join(cmd)}")
        return subprocess.run(cmd).returncode == 0

    @staticmethod
    def _parse_tsv(path):
        """Yield HIPE documents {meta, tokens:[{tok,ner,nel,nospace}]}."""
        header = None
        doc = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("TOKEN\t") and header is None:
                    header = {name: i for i, name in enumerate(line.split("\t"))}
                    continue
                if line.startswith("#"):
                    m = re.match(r"#\s*hipe2022:(\S+)\s*=\s*(.*)", line)
                    if m and m.group(1) == "document_id":
                        if doc and doc["tokens"]:
                            yield doc
                        doc = {"meta": {"document_id": m.group(2).strip()}, "tokens": []}
                    elif m and doc is not None:
                        doc["meta"][m.group(1)] = m.group(2).strip()
                    continue
                if doc is None or header is None:
                    continue
                cols = line.split("\t")
                if len(cols) <= header.get("NEL-LIT", 7):
                    continue
                misc = cols[header["MISC"]] if header.get("MISC") is not None else "_"
                doc["tokens"].append(
                    {
                        "tok": cols[header["TOKEN"]],
                        "ner": cols[header["NE-COARSE-LIT"]],
                        "nel": cols[header["NEL-LIT"]],
                        "nospace": "NoSpaceAfter" in misc,
                    }
                )
        if doc and doc["tokens"]:
            yield doc

    @staticmethod
    def _reconstruct(tokens):
        parts, spans, pos = [], [], 0
        for i, t in enumerate(tokens):
            tok = t["tok"]
            spans.append((pos, pos + len(tok)))
            parts.append(tok)
            pos += len(tok)
            if not t["nospace"] and i != len(tokens) - 1:
                parts.append(" ")
                pos += 1
        return "".join(parts), spans

    @staticmethod
    def _mentions(doc, window):
        """Linked mentions (NEL-LIT is a Wikidata QID) as unified samples."""
        tokens = doc["tokens"]
        text, spans = HipeBuilder._reconstruct(tokens)
        out = []
        i, n = 0, len(tokens)
        while i < n:
            ner = tokens[i]["ner"]
            if not ner.startswith("B-"):
                i += 1
                continue
            etype = ner[2:]
            j = i + 1
            while j < n and tokens[j]["ner"] == f"I-{etype}":
                j += 1
            qids = [
                tokens[k]["nel"] for k in range(i, j) if HipeBuilder.QID_RE.match(tokens[k]["nel"])
            ]
            if qids:
                m_start, m_end = spans[i][0], spans[j - 1][1]
                left, right = text[:m_start], text[m_end:]
                if window and window > 0:
                    left, right = left[-window:], right[:window]
                md = doc["meta"]
                out.append(
                    {
                        "mention": text[m_start:m_end],
                        "context_left": left,
                        "context_right": right,
                        "label_id": qids[0],  # Wikidata QID (string)
                        "label_title": "",
                        "type": etype,
                        "metadata": {
                            "qid": qids[0],
                            "doc_id": md.get("document_id"),
                            "date": md.get("date"),
                            "language": md.get("language"),
                            "dataset": md.get("dataset"),
                        },
                    }
                )
            i = j
        return out

    @staticmethod
    def to_nel(bench, splits=("train", "dev", "test"), window=400, lang=None):
        """HIPE TSV -> unified NEL JSONL, at data/dataset/<bench>/<lang>/<split>.jsonl.

        The first of HIPE's five passes. `lang=None` does every language the benchmark
        ships. -> (mentions written, type counts)."""
        root = os.path.join(config.DS_HIPE_RAW, "data", "v2.1")
        if not os.path.isdir(root):
            print(f"[hipe] no {root}; is the repo present?")
            return 0, collections.Counter()
        ds_dir = os.path.join(root, bench)
        if not os.path.isdir(ds_dir):
            print(f"[skip] {bench}")
            return 0, collections.Counter()
        # *masked* variants blank the columns we need — they are the shared-task inputs.
        tsvs = [
            os.path.join(dp, fn)
            for dp, _, fns in os.walk(ds_dir)
            for fn in fns
            if fn.endswith(".tsv") and "masked" not in fn
        ]
        total, types = 0, collections.Counter()
        for path in sorted(tsvs):
            m = re.search(r"-(train|dev|dev2|test)-([a-z]{2})\.tsv$", os.path.basename(path))
            if not m:
                continue
            split, lg = m.group(1), m.group(2)
            if lang and lg != lang:
                continue
            if split.replace("dev2", "dev") not in splits and split not in splits:
                continue
            samples = []
            for doc in HipeBuilder._parse_tsv(path):
                samples.extend(HipeBuilder._mentions(doc, window))
            out = config.HIPE_NEL_DATASETS.get(bench, os.path.join(config._DDIR, bench))
            _write_jsonl(samples, os.path.join(out, lg, f"{split}.jsonl"))
            t = collections.Counter(s["type"] for s in samples)
            total += len(samples)
            types.update(t)
            print(
                f"  [{bench}/{lg}/{split}] {len(samples)} linked mentions "
                f"({', '.join(f'{k}={v}' for k, v in t.most_common())})"
            )
        return total, types

    @staticmethod
    def _load_qid_cache():
        if os.path.exists(HipeBuilder.QID_CACHE):
            HipeBuilder._qid_cache = json.load(open(HipeBuilder.QID_CACHE))

    @staticmethod
    def _save_qid_cache():
        json.dump(HipeBuilder._qid_cache, open(HipeBuilder.QID_CACHE, "w"), ensure_ascii=False)

    @staticmethod
    def _fetch_qid_labels(qids, lang):
        """QID -> {'label','description','aliases':[...]} in `lang`, cached per (qid,lang)."""
        need = [q for q in qids if f"{q}@{lang}" not in HipeBuilder._qid_cache]
        for i in range(0, len(need), 50):
            batch = need[i : i + 50]
            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|descriptions|aliases",
                "languages": lang,
                "format": "json",
            }
            url = HipeBuilder.WD_API + "?" + urllib.parse.urlencode(params)
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": HipeBuilder.UA})
                    d = json.load(urllib.request.urlopen(req, timeout=40))
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"    [warn] fetch failed for batch {i}: {e}")
                        d = {"entities": {}}
                    else:
                        time.sleep(2 * (attempt + 1))
            for q in batch:
                e = (d.get("entities") or {}).get(q, {})
                lab = (e.get("labels", {}).get(lang) or {}).get("value", "")
                desc = (e.get("descriptions", {}).get(lang) or {}).get("value", "")
                al = [a["value"] for a in e.get("aliases", {}).get(lang, [])]
                HipeBuilder._qid_cache[f"{q}@{lang}"] = {
                    "label": lab,
                    "description": desc,
                    "aliases": al,
                }
            HipeBuilder._save_qid_cache()
            print(f"    fetched {min(i + 50, len(need))}/{len(need)} qids ({lang})")
        return {
            q: HipeBuilder._qid_cache.get(
                f"{q}@{lang}", {"label": "", "description": "", "aliases": []}
            )
            for q in qids
        }

    @staticmethod
    def to_kb(bench, lang):
        src = os.path.join(HipeBuilder.NEL_SRC, bench, lang, "test.jsonl")
        if not os.path.exists(src):
            return None
        rows = [json.loads(l) for l in open(src)]
        # gold qids in first-seen order -> numeric_id
        qids, seen = [], set()
        for r in rows:
            q = str((r.get("metadata") or {}).get("qid") or r.get("label_id") or "")
            if q and q not in seen:
                seen.add(q)
                qids.append(q)
        real = [q for q in qids if q.startswith("Q")]
        info = HipeBuilder._fetch_qid_labels(real, lang)

        qid2nid, ents = {}, []
        for q in qids:
            nid = len(ents)
            qid2nid[q] = nid
            meta = info.get(q, {"label": "", "description": "", "aliases": []})
            is_nil = not q.startswith("Q")
            title = (meta["label"] or q) if not is_nil else q
            ents.append(
                {
                    "numeric_id": nid,
                    "idx": f"wikidata:{q}",
                    "title": title,
                    "entity": meta["description"],
                    "text": (title + "。" + meta["description"]).strip("。"),
                    "metadata": {
                        "qid": q,
                        "aliases": meta["aliases"],
                        "language": lang,
                        "dataset": bench,
                        "nil": is_nil,
                    },
                }
            )

        out_dir = os.path.join(_ROOT, "data", f"dataset_{bench}_{lang}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "entities.jsonl"), "w") as f:
            for e in ents:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        n_gold = 0
        with open(os.path.join(out_dir, "test.jsonl"), "w") as f:
            for r in rows:
                q = str((r.get("metadata") or {}).get("qid") or r.get("label_id") or "")
                nid = qid2nid.get(q, -1)
                if nid >= 0:
                    n_gold += 1
                o = dict(r)
                o["label_id"] = nid
                o["label_title"] = ents[nid]["title"] if nid >= 0 else ""
                o["label"] = o["label_title"]  # label needed by BLINK/DyVo readers
                md = dict(o.get("metadata") or {})
                md["qid"] = q
                o["metadata"] = md
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(
            f"  {bench}_{lang}: {len(rows)} mentions, {len(ents)} entities "
            f"({len(real)} Wikidata QIDs) -> data/dataset_{bench}_{lang}/"
        )
        return f"{bench}_{lang}"

    @staticmethod
    def _find_split_tsv(bench, lang, split):
        """Newest-version train/dev TSV for a corpus (v2.1 > v2.0 > v1.0)."""
        for ver in ("v2.1", "v2.0", "v1.0"):
            hits = glob.glob(
                os.path.join(HipeBuilder.HIPE_DATA, ver, bench, lang, f"*-{split}-{lang}.tsv")
            )
            if hits:
                return sorted(hits)[0]
        return None

    @staticmethod
    def _parse_split_tsv(path):
        """Yield mention records {mention, context_left, context_right, qid, type, meta}."""
        docs, cur, meta = [], [], {}
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("# hipe2022:document_id"):
                if cur:
                    docs.append((cur, meta))
                    cur, meta = [], {}
                meta = {"document_id": line.split("=", 1)[1].strip()}
            elif line.startswith("# hipe2022:"):
                k, _, v = line[len("# hipe2022:") :].partition("=")
                meta[k.strip()] = v.strip()
            elif line and not line.startswith("#") and not line.startswith("TOKEN\t"):
                cur.append(line.split("\t"))
        if cur:
            docs.append((cur, meta))

        for rows, meta in docs:
            # reconstruct text + per-token char offsets
            text, spans = "", []
            for c in rows:
                tok = c[0]
                misc = c[9] if len(c) > 9 else ""
                start = len(text)
                text += tok
                spans.append((start, len(text), c))
                if "NoSpaceAfter" not in misc:
                    text += " "
            # group contiguous B-/I- entity spans sharing a NEL QID
            i = 0
            while i < len(spans):
                s0, e0, c = spans[i]
                coarse = c[1]
                nel = c[7] if len(c) > 7 else "_"
                if coarse.startswith("B-") and nel.startswith("Q"):
                    j = i + 1
                    while j < len(spans) and spans[j][2][1].startswith("I-"):
                        j += 1
                    m_start, m_end = s0, spans[j - 1][1]
                    yield {
                        "mention": text[m_start:m_end],
                        "context_left": text[max(0, m_start - HipeBuilder.SPLIT_WIN) : m_start],
                        "context_right": text[m_end : m_end + HipeBuilder.SPLIT_WIN],
                        "qid": nel.split("|")[0].split(",")[0],
                        "type": coarse[2:],
                        "metadata": {
                            "qid": nel.split("|")[0].split(",")[0],
                            "doc_id": meta.get("document_id", ""),
                            "date": meta.get("date", ""),
                            "language": meta.get("language", ""),
                            "dataset": meta.get("dataset", ""),
                        },
                    }
                    i = j
                else:
                    i += 1

    @staticmethod
    def add_splits(bench, lang):
        out_dir = os.path.join(_ROOT, "data", f"dataset_{bench}_{lang}")
        if not os.path.isdir(out_dir):
            print(f"  [skip] {bench}_{lang}: no converted dir (run to_kb first)")
            return None
        # gather records per split
        split_recs = {}
        test_p = os.path.join(out_dir, "test.jsonl")
        if os.path.exists(test_p):
            # re-read the original (QID-labelled) test to re-key against the new pool
            orig_test = os.path.join(HipeBuilder.NEL_SRC, bench, lang, "test.jsonl")
            split_recs["test"] = [json.loads(l) for l in open(orig_test)]
        for split, name in (("train", "train"), ("dev", "valid")):
            tsv = HipeBuilder._find_split_tsv(bench, lang, split)
            if tsv:
                split_recs[name] = list(HipeBuilder._parse_split_tsv(tsv))
        # corpora with no official train (hipe2020_en, sonar_de): promote dev -> train,
        # carving a small deterministic valid slice so trainers still have an eval set.
        if "train" not in split_recs and HipeBuilder.DEV_AS_TRAIN and "valid" in split_recs:
            import random

            dev = list(split_recs["valid"])
            random.Random(52313).shuffle(dev)
            cut = max(1, int(len(dev) * 0.15))
            split_recs["valid"], split_recs["train"] = dev[:cut], dev[cut:]
            print(
                f"  [dev->train] {bench}_{lang}: dev {len(dev)} -> train {len(dev) - cut} / valid {cut}"
            )
        if "train" not in split_recs:
            print(f"  [skip] {bench}_{lang}: no train TSV")
            return None

        # union of QIDs -> fetch -> entities
        def qid_of(r):
            return str(
                (r.get("metadata") or {}).get("qid") or r.get("qid") or r.get("label_id") or ""
            )

        qids, seen = [], set()
        for recs in split_recs.values():
            for r in recs:
                q = qid_of(r)
                if q and q not in seen:
                    seen.add(q)
                    qids.append(q)
        real = [q for q in qids if q.startswith("Q")]
        HipeBuilder._load_qid_cache()
        info = HipeBuilder._fetch_qid_labels(real, lang)

        qid2nid, ents = {}, []
        for q in qids:
            qid2nid[q] = len(ents)
            m = info.get(q, {"label": "", "description": "", "aliases": []})
            nil = not q.startswith("Q")
            title = q if nil else (m["label"] or q)
            ents.append(
                {
                    "numeric_id": len(ents),
                    "idx": f"wikidata:{q}",
                    "title": title,
                    "entity": m["description"],
                    "text": (title + "。" + m["description"]).strip("。"),
                    "metadata": {
                        "qid": q,
                        "aliases": m["aliases"],
                        "language": lang,
                        "dataset": bench,
                        "nil": nil,
                    },
                }
            )
        with open(os.path.join(out_dir, "entities.jsonl"), "w") as f:
            for e in ents:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        counts = {}
        for name, recs in split_recs.items():
            with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
                for r in recs:
                    q = qid_of(r)
                    nid = qid2nid.get(q, -1)
                    title = ents[nid]["title"] if nid >= 0 else ""
                    o = {
                        "mention": r["mention"],
                        "context_left": r.get("context_left", ""),
                        "context_right": r.get("context_right", ""),
                        "label_id": nid,
                        "label": title,
                        "label_title": title,  # label needed by BLINK/DyVo
                        "type": r.get("type", ""),
                        "world": bench,
                        "metadata": {**(r.get("metadata") or {}), "qid": q},
                    }
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            counts[name] = len(recs)
        print(f"  {bench}_{lang}: {counts} | {len(ents)} entities")
        return f"{bench}_{lang}"

    @staticmethod
    def _fine_docs(path):
        """Yield (token_rows, doc_meta) per document, as _parse_split_tsv parses them."""
        docs, cur, meta = [], [], {}
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.startswith("# hipe2022:document_id"):
                if cur:
                    docs.append((cur, meta))
                    cur, meta = [], {}
                meta = {"document_id": line.split("=", 1)[1].strip()}
            elif line.startswith("# hipe2022:"):
                k, _, v = line[len("# hipe2022:") :].partition("=")
                meta[k.strip()] = v.strip()
            elif line and not line.startswith("#") and not line.startswith("TOKEN\t"):
                cur.append(line.split("\t"))
        if cur:
            docs.append((cur, meta))
        return docs

    @staticmethod
    def _fine_map(bench, lang):
        """(doc_id, surface, qid) -> fine tag ('' when the corpus leaves it unannotated).

        Mirrors _parse_split_tsv's span logic exactly, so the surfaces it
        produces are the same strings that ended up in test.jsonl."""
        out = {}
        # *masked* variants blank the very columns we need — they are the shared-task
        # inputs, not the labelled release.
        for path in (
            p for p in glob.glob(f"{HipeBuilder.RAW_V21}/{bench}/{lang}/*.tsv") if "masked" not in p
        ):
            for rows, meta in HipeBuilder._fine_docs(path):
                text, spans = "", []
                for c in rows:
                    misc = c[9] if len(c) > 9 else ""
                    st = len(text)
                    text += c[0]
                    spans.append((st, len(text), c))
                    if "NoSpaceAfter" not in misc:
                        text += " "
                i = 0
                while i < len(spans):
                    s0, _, c = spans[i]
                    coarse, nel = c[1], (c[7] if len(c) > 7 else "_")
                    if coarse.startswith("B-") and nel.startswith("Q"):
                        j = i + 1
                        while j < len(spans) and spans[j][2][1].startswith("I-"):
                            j += 1
                        f = c[3] if len(c) > 3 else "_"
                        f = f[2:] if f[:2] in ("B-", "I-") else ""
                        key = (
                            meta.get("document_id", ""),
                            text[s0 : spans[j - 1][1]],
                            nel.split("|")[0].split(",")[0],
                        )
                        out[key] = "" if f in ("_", "O") else f
                        i = j
                    else:
                        i += 1
        return out

    @staticmethod
    def add_fine_types(ds, dry_run=False):
        bench, lang = HipeBuilder.DS[ds]
        fm = HipeBuilder._fine_map(bench, lang)
        path = f"{_ROOT}/data/dataset_{ds}/test.jsonl"
        rows = [json.loads(l) for l in open(path)]
        hit = fine = 0
        vals = set()
        for r in rows:
            md = r.setdefault("metadata", {})
            f = fm.get((md.get("doc_id", ""), r["mention"], md.get("qid", "")))
            if f is None:
                continue
            hit += 1
            if f:
                md["fine_type"] = f
                fine += 1
                vals.add(f)
        n = len(rows)
        print(
            f"{ds:<14} joined {100 * hit / n:5.1f}%  fine_type on {100 * fine / n:5.1f}%  "
            f"({len(vals)} tags) {sorted(vals)[:4]}"
        )
        if not dry_run and fine:
            with open(path, "w") as f_:
                for r in rows:
                    f_.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def _wd_year(claim):
        """Signed year (negative = BC; AJMC is classical, ~half its dates are BC)."""
        try:
            t = claim["mainsnak"]["datavalue"]["value"]["time"]
            m = re.match(r"([+-])(\d{1,5})-", t)
            if not m:
                return None
            y = int(m.group(2))
            return -y if m.group(1) == "-" else y
        except Exception:
            return None

    @staticmethod
    def _wd_yr(y):
        return f"{abs(y)} BC" if y < 0 else str(y)

    @staticmethod
    def _wd_get(url):
        req = urllib.request.Request(url, headers={"User-Agent": HipeBuilder.UA})
        return json.load(urllib.request.urlopen(req, timeout=40))

    @staticmethod
    def _batched(items, size=50):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    @staticmethod
    def _fetch_claims(qids):
        data = (
            json.load(open(HipeBuilder.CLAIM_CACHE))
            if os.path.exists(HipeBuilder.CLAIM_CACHE)
            else {}
        )
        todo = [q for q in qids if q not in data]
        for n, batch in enumerate(HipeBuilder._batched(todo)):
            url = (
                "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="
                + "|".join(batch)
                + "&props=claims&format=json"
            )
            for attempt in range(4):
                try:
                    ents = HipeBuilder._wd_get(url)["entities"]
                    for q in batch:
                        cl = (ents.get(q) or {}).get("claims", {}) or {}
                        rec = {"p31": []}
                        for c in cl.get("P31", [])[:3]:
                            try:
                                rec["p31"].append(c["mainsnak"]["datavalue"]["value"]["id"])
                            except Exception:
                                pass
                        for pid, name in HipeBuilder.DATE_PROPS.items():
                            if pid in cl:
                                y = HipeBuilder._wd_year(cl[pid][0])
                                if y is not None:
                                    rec[name] = y
                        data[q] = rec
                    break
                except Exception as e:
                    if attempt == 3:
                        for q in batch:
                            data.setdefault(q, {"p31": []})
                        print(f"  claim batch failed: {repr(e)[:60]}")
                    time.sleep(2**attempt)
            json.dump(data, open(HipeBuilder.CLAIM_CACHE, "w"))
            print(f"  claims {min((n + 1) * 50, len(todo))}/{len(todo)}", end="\r")
        if todo:
            print()
        return data

    @staticmethod
    def _fetch_type_labels(type_qids):
        lab = (
            json.load(open(HipeBuilder.LABEL_CACHE))
            if os.path.exists(HipeBuilder.LABEL_CACHE)
            else {}
        )
        todo = [q for q in type_qids if q not in lab]
        for batch in HipeBuilder._batched(todo):
            url = (
                "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="
                + "|".join(batch)
                + "&props=labels&languages=en&format=json"
            )
            try:
                ents = HipeBuilder._wd_get(url)["entities"]
                for q in batch:
                    lab[q] = ((ents.get(q) or {}).get("labels", {}).get("en", {}) or {}).get(
                        "value", ""
                    )
            except Exception as e:
                print(f"  label batch failed: {repr(e)[:60]}")
                for q in batch:
                    lab.setdefault(q, "")
            json.dump(lab, open(HipeBuilder.LABEL_CACHE, "w"))
        return lab

    @staticmethod
    def add_wikidata_evidence(ds):
        D = f"{_ROOT}/data/dataset_{ds}"
        src = next(
            f"{D}/{f}"
            for f in ("entities_aliased.jsonl", "entities.jsonl")
            if os.path.exists(f"{D}/{f}")
        )
        ents = [json.loads(l) for l in open(src)]
        qids = sorted(
            {
                (e.get("metadata") or {}).get("qid")
                for e in ents
                if str((e.get("metadata") or {}).get("qid", "")).startswith("Q")
            }
        )
        print(f"[{ds}] {len(qids)} QIDs (from {os.path.basename(src)})")
        claims = HipeBuilder._fetch_claims(qids)
        labels = HipeBuilder._fetch_type_labels(
            sorted({v for r in claims.values() for v in r.get("p31", [])})
        )

        n_t = n_d = 0
        for e in ents:
            md = e.setdefault("metadata", {})
            rec = claims.get(md.get("qid") or "", {})
            types = [labels.get(v, "") for v in rec.get("p31", []) if labels.get(v)]
            if types:
                md["wd_type"] = "; ".join(types[:3])
                n_t += 1
            b, d = rec.get("birth_year"), rec.get("death_year")
            inc = rec.get("inception_year") or rec.get("publication_year")
            if b or d:
                md["span"] = (
                    f"{HipeBuilder._wd_yr(b) if b else '?'}-{HipeBuilder._wd_yr(d) if d else '?'}"
                )
                n_d += 1
            elif inc is not None:
                md["span"] = HipeBuilder._wd_yr(inc)
                n_d += 1
        out = f"{D}/entities_evidence.jsonl"
        with open(out, "w") as f:
            for e in ents:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(
            f"  wd_type {n_t}/{len(ents)} ({100 * n_t / len(ents):.0f}%) | "
            f"dates {n_d}/{len(ents)} ({100 * n_d / len(ents):.0f}%) -> {os.path.basename(out)}"
        )


class MahanamaBuilder(DatasetBuilder):
    """Mahānāma: Sanskrit EDL over the Mahābhārata, CoNLL-U + an English KB.

    Kept for its role as this project's negative control — a corpus with abundant
    annotation (266 epithets per entity) but no query key that separates candidates,
    where every form of evidence provision measurably loses."""

    name = "mahanama"

    #: the upstream repo (CoNLL-U corpus + the English-glossed index KB)
    REPO = f"{config.WS}/mahanama"
    #: MISC opening bracket: Entity=(e12207-person--vizRu)  (id-type-head-identity)
    _OPEN = re.compile(r"Entity=\((e\d+)-([^-]*)-[^-]*-([^)|]*)")
    _BASE = re.compile(r"base_name=([^|]+)")

    def materialize(self, sample=1000, full=False, with_train=False, **kw):
        return self.convert(self.path, sample=sample, full=full, with_train=with_train, **kw)

    @staticmethod
    def _scan_types():
        """eid -> entity type (person / location / misc), read off the CoNLL-U layer.

        knowledge_base.json carries no type at all; the annotation lives only on the
        corpus tokens (`Entity=(e3873-location--ekacakrA)`). Reading the type off the
        KB therefore always missed, and defaulting the miss to "person" made all 12,713
        entities look identically typed — which in turn made the type key look useless
        by construction rather than by evidence. Entities never mentioned in the corpus
        (57%) genuinely have no annotated type and are left without one."""
        types = {}
        for fp in sorted(
            glob.glob(
                os.path.join(
                    MahanamaBuilder.REPO, "data", "mahanama_conllu", "volume_*", "*.conllu"
                )
            )
        ):
            for line in open(fp, encoding="utf-8"):
                m = MahanamaBuilder._OPEN.search(line)
                if m and m.group(2):
                    types[m.group(1)] = m.group(2)
        return types

    @staticmethod
    def _build_kb(etypes=None):
        """KB eid -> entity row (mirrors the HIPE KB schema); returns
        (ents, eid2nid, eid2cluster_nid)."""
        etypes = etypes if etypes is not None else MahanamaBuilder._scan_types()
        kb = json.load(
            open(os.path.join(MahanamaBuilder.REPO, "data", "kb", "knowledge_base.json"))
        )
        eids = list(kb.keys())
        eid2nid = {e: i for i, e in enumerate(eids)}

        # cluster map: every head + its alias siblings share one canonical (head) id.
        eid2head = {}
        for e, v in kb.items():
            if v.get("cluster_head"):
                eid2head[e] = e
                for a in v.get("aliases") or []:
                    eid2head[a] = e
        for e in eids:  # uncovered entries are their own cluster
            eid2head.setdefault(e, e)

        # surface names of every cluster member, attached to each member for recall.
        cluster_names = {}
        for e in eids:
            cluster_names.setdefault(eid2head[e], set())
        for e, v in kb.items():
            cluster_names[eid2head[e]].add(v.get("key", ""))

        ents = []
        for e in eids:
            v = kb[e]
            head = eid2head[e]
            title = v.get("key", "") or e
            desc = v.get("cleaned_description") or v.get("description") or ""
            aliases = sorted(n for n in cluster_names[head] if n and n != title)
            md = {
                "qid": e,
                "aliases": aliases,
                "language": "sa",
                "dataset": "mahanama",
                "nil": False,
                "cluster_head_eid": head,
                "cluster_id": eid2nid[head],
            }
            if etypes.get(e):
                md["type"] = etypes[e]
            ents.append(
                {
                    "numeric_id": eid2nid[e],
                    "idx": f"mahanama:{e}",
                    "title": title,
                    "entity": desc,
                    "text": (title + "。" + desc).strip("。"),
                    "metadata": md,
                }
            )
        eid2cluster_nid = {e: eid2nid[eid2head[e]] for e in eids}
        return ents, eid2nid, eid2cluster_nid

    @staticmethod
    def _iter_mentions():
        """Yield (eid, etype, surface, sentence) for each annotated token."""
        files = sorted(
            glob.glob(
                os.path.join(
                    MahanamaBuilder.REPO, "data", "mahanama_conllu", "volume_*", "*.conllu"
                )
            )
        )
        for fp in files:
            sent, toks = "", []
            for line in open(fp, encoding="utf-8"):
                if line.startswith("# text ="):
                    sent = line.split("=", 1)[1].strip()
                elif line.strip() == "":
                    for form, misc in toks:
                        m = MahanamaBuilder._OPEN.search(misc)
                        if not m:
                            continue
                        bm = MahanamaBuilder._BASE.search(misc)
                        surface = (bm.group(1).strip() if bm else form).strip()
                        yield m.group(1), (m.group(2) or "person"), surface, sent
                    sent, toks = "", []
                elif not line.startswith("#"):
                    c = line.rstrip("\n").split("\t")
                    if len(c) >= 10:
                        toks.append((c[1], c[9]))

    @staticmethod
    def _split_ctx(sentence, surface):
        """Left/right split of the verse around the mention surface (best effort)."""
        i = sentence.find(surface)
        if i < 0:
            return "", sentence
        return sentence[:i], sentence[i + len(surface) :]

    @staticmethod
    def convert(
        dst,
        *,
        sample=1000,
        full=False,
        with_train=False,
        train_size=30000,
        valid_size=2000,
        seed=13,
    ):
        """CoNLL-U + KB json -> this project's NEL format, at `dst`.

        `sample` is the test size: rows are shuffled with a fixed seed and the first
        `sample` become test, so shrinking it yields a subset of the previous test
        (first 1000 of the same 13-seeded order), not a disjoint draw."""
        os.makedirs(dst, exist_ok=True)
        ents, eid2nid, eid2cluster = MahanamaBuilder._build_kb()
        _write_jsonl(ents, os.path.join(dst, "entities.jsonl"))
        n_alias = sum(1 for e in ents if e["metadata"]["aliases"])
        print(f"  entities: {len(ents)} ({n_alias} with >=1 alias) -> {dst}/entities.jsonl")

        rows, dropped = [], 0
        for eid, etype, surface, sent in MahanamaBuilder._iter_mentions():
            nid = eid2nid.get(eid, -1)
            if nid < 0 or not surface:
                dropped += 1
                continue
            cl, cr = MahanamaBuilder._split_ctx(sent, surface)
            rows.append(
                {
                    "mention": surface,
                    "context_left": cl,
                    "context_right": cr,
                    "label_id": nid,
                    "label_title": ents[nid]["title"],
                    "label": ents[nid]["title"],
                    "type": etype,
                    "world": "mahanama",
                    "metadata": {
                        "qid": eid,
                        "language": "sa",
                        "dataset": "mahanama",
                        "cluster_id": eid2cluster[eid],
                        "sentence": sent,
                    },
                }
            )
        total = len(rows)

        def write(split, items):
            _write_jsonl(items, os.path.join(dst, f"{split}.jsonl"))
            print(f"  {split}: {len(items)} mentions -> {dst}/{split}.jsonl")

        if full:
            write("test", rows)
            return dst
        # Deterministic shuffle; test = first `sample` (UNCHANGED from prior runs so
        # the 4k predictions still align), then disjoint valid + train for encoders.
        random.Random(seed).shuffle(rows)
        write("test", rows[:sample])
        if with_train:
            v0, v1 = sample, sample + valid_size
            write("valid", rows[v0:v1])
            write("train", rows[v1 : v1 + train_size])
        print(f"  (of {total} total mentions, {dropped} dropped)")
        return dst


#: name -> builder. HIPE's 12 are generated rather than listed: they differ only in
#: which (benchmark, language) pair they read.
BUILDERS = {
    **{s: (lambda s=s: YakusyaBuilder(s)) for s in YakusyaBuilder.CHAIN + ("arc",)},
    "daime_eval_full": DaimeEvalBuilder,
    "mahanama": MahanamaBuilder,
    **{
        f"{b}_{l}": (lambda b=b, l=l: HipeBuilder(b, l))
        for b, langs in (
            ("ajmc", "de en fr"),
            ("hipe2020", "de en fr"),
            ("newseye", "de fi fr sv"),
            ("sonar", "de"),
            ("topres19th", "en"),
        )
        for l in langs.split()
    },
}


def get_builder(name):
    """-> the builder for a registered dataset name, or None if it has none.

    Not every registered dataset is buildable: `daime_eval` is the frozen 1479-row
    subset kept for comparison against published numbers, and `daime_ft_w100` is a
    re-split of daime_eval_full — both are artifacts of a past run, not passes."""
    f = BUILDERS.get(name)
    return f() if f else None


# ============================================================================
# CLI
# Thin wrappers: parse args, call a builder or a pass.
# ============================================================================


def cmd_daime_eval(args):
    if args.frm == "raw":
        if not (args.csv_dir and args.raw_out):
            sys.exit("--from raw requires --csv_dir and --raw_out")
        base_yr = YakusyaBuilder.raw_to_base(args.csv_dir, args.raw_out)
        if not base_yr:
            print("[abort] stage 0 incomplete; fix the external dependencies then resume with --from base.")
            return
        args.base = args.base or base_yr
    # v2, not v1: v2 is v1 with entity_canon's A-only (same-source) duplicates
    # physically merged, 5000 -> 4885 entities. It changes nothing about which
    # mentions survive or what their gold is — build_daime_eval runs the same canon
    # map itself, so both bases yield the identical 2214 with identical golds — but
    # it means the entities.jsonl shipped alongside is the deduplicated one rather
    # than one carrying 115 known same-source duplicates.
    base = args.base or config.DS_RELINKED_YR_V2
    dst = args.dst or config.DS_DAIME_EVAL_FULL
    work = args.work or (dst.rstrip("/") + "_multidaime_clean")
    print(f"[stage 1] base={base} -> multidaime_clean={work}")
    DaimeEvalBuilder.multidaime_clean(base, work)
    print(f"[stage 2] multidaime_clean(all splits) -> daime_eval={dst}")
    DaimeEvalBuilder.daime_eval(base, DaimeEvalBuilder.probe(work), dst)
    print(f"\n[done] {dst}/test.jsonl + entities.jsonl")


def cmd_hipe(args):
    """Download HIPE-2022 + convert its EL datasets to unified NEL JSONL."""
    if not args.no_download and not HipeBuilder.download(config.DS_HIPE_RAW):
        print("[hipe] clone failed (no network?). Aborting.")
        return
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    grand, gtypes = 0, collections.Counter()
    for ds in args.datasets:
        n, t = HipeBuilder.to_nel(ds, splits=splits, window=args.window)
        grand += n
        gtypes.update(t)
    print(f"\n[hipe] {grand} linked mentions -> data/dataset/<dataset>/<lang>/")
    print(f"[hipe] types: {dict(gtypes.most_common())}")


def cmd_tag(args):
    """Tag multi-candidate mentions across datasets. Ambiguity is pooled within a
    yakusya dataset (all splits) and within a HIPE dataset's LANGUAGE (all its
    splits), so a surface ambiguous anywhere in that scope is tagged everywhere."""
    from dataset import Dataset

    tag = args.tag
    names = args.datasets or (["yakusya_nel"] + list(config.HIPE_NEL_DATASETS))
    total = 0
    for name in names:
        path = config.DATASETS.get(name) or os.path.join(config._DDIR, name)
        yfiles = [
            os.path.join(path, f"{s}.jsonl")
            for s in ("train", "valid", "test")
            if os.path.exists(os.path.join(path, f"{s}.jsonl"))
        ]
        if yfiles:  # yakusya 4-file: pool all splits
            surf = _ambiguous_surfaces([json.loads(l) for f in yfiles for l in open(f)])
            n = sum(_apply_tag(f, surf, tag) for f in yfiles)
            print(f"  {name}: {tag} x{n} ({len(surf)} ambiguous surfaces)")
            total += n
            continue
        n = 0  # HIPE per-language: pool a lang's splits
        for lang in Dataset.langs(name, path=path):
            langdir = os.path.join(path, lang)
            files = [os.path.join(langdir, f) for f in os.listdir(langdir) if f.endswith(".jsonl")]
            surf = _ambiguous_surfaces([json.loads(l) for f in files for l in open(f)])
            n += sum(_apply_tag(f, surf, tag) for f in files)
        print(f"  {name}: {tag} x{n}")
        total += n
    print(f"[tag] total {tag}: {total}")


# ============================================================================
# recut context window (store wide; truncate at load via Dataset.load(window=W))
# ============================================================================


def cmd_recut(args):
    """Re-cut a dataset's context window to W chars from the SOURCE pages, so the
    w100/w200/w300 ablation collapses to ONE wide-stored dataset + a load-time
    param: store at w300 here, then Dataset.load(..., window=100) truncates down."""
    path = config.DATASETS.get(args.dataset) or os.path.join(config._DDIR, args.dataset)
    if not os.path.isdir(path):
        print(f"[recut] no dataset dir: {path}")
        return
    for fn in ("train", "valid", "test"):
        p = os.path.join(path, f"{fn}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        out, na = [], 0
        for r in rows:
            rr, st = recut_record(r, args.win)
            out.append(rr)
            na += st == "aligned"
        _write_jsonl(out, p)
        print(
            f"  {args.dataset}/{fn}: {len(rows)} -> w{args.win} (aligned {na}, "
            f"{len(rows) - na} passthrough)"
        )


# ============================================================================
# merge per-model LLM cleaning caches into entities_enriched.jsonl
# ============================================================================


def merge_llm_caches(input, enriched, cache_glob):
    def load_jsonl(path):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    src = load_jsonl(input)
    by_id = {r["numeric_id"]: r for r in src}
    print(f"[load] source: {len(src)} entities", file=sys.stderr)

    if Path(enriched).exists():
        prev = load_jsonl(enriched)
        for r in prev:
            nid = r["numeric_id"]
            if nid in by_id:
                # carry existing extra fields (text_clean_py, etc.)
                for k, v in r.items():
                    if k not in by_id[nid]:
                        by_id[nid][k] = v
        print(f"[load] existing enriched: {len(prev)} entities", file=sys.stderr)

    cache_files = sorted(glob.glob(cache_glob))
    print(
        f"[caches] found {len(cache_files)}: {[os.path.basename(p) for p in cache_files]}",
        file=sys.stderr,
    )

    name_re = re.compile(r"_llm_clean_cache_([^/]+)\.jsonl$")
    summary = []
    for cp in cache_files:
        m = name_re.search(cp)
        if not m:
            print(f"[skip] cannot parse model name: {cp}", file=sys.stderr)
            continue
        short = m.group(1)
        field = f"text_clean_llm_{short}"
        n_set = 0
        n_flagged = 0
        for d in load_jsonl(cp):
            nid = d.get("numeric_id")
            text = d.get("text_clean_llm", "")
            if nid in by_id and text:
                by_id[nid][field] = text
                n_set += 1
                if d.get("added_facts"):
                    n_flagged += 1
        summary.append((short, field, n_set, n_flagged))
        print(f"[merge] {short}: set {n_set} field={field} flagged={n_flagged}", file=sys.stderr)

    out = list(by_id.values())
    out.sort(key=lambda r: r["numeric_id"])
    with open(enriched, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {enriched}", file=sys.stderr)

    # final field coverage
    print("\n=== Field coverage ===", file=sys.stderr)
    fields = set()
    for r in out:
        fields.update(r.keys())
    extra = sorted(f for f in fields if f.startswith("text_clean_"))
    for f_ in extra:
        n = sum(1 for r in out if r.get(f_))
        print(f"  {f_}: {n}/{len(out)}", file=sys.stderr)


def cmd_merge_caches(args):
    merge_llm_caches(args.input, args.enriched, args.cache_glob)


# ============================================================================
# registry listing
# ============================================================================


def cmd_list(_args):
    print(f"{'name':24s} {'exists':6s}  path")
    for name, path in config.DATASETS.items():
        ok = "✓" if os.path.exists(path) else "✗"
        print(f"{name:24s} {ok:6s}  {path}")


# ============================================================================
# driver
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Build each dataset from raw material to a Dataset (paths in config.DATASETS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "daime_eval", help="private 代目 disambiguation set: base -> multidaime_clean -> daime_eval"
    )
    p.add_argument("--from", dest="frm", choices=["raw", "base"], default="base")
    p.add_argument("--csv_dir", help="[--from raw] source CSV directory (default: config.DS_RAW_CSV/csv)")
    p.add_argument("--raw_out", help="[--from raw] stage0 base output directory")
    p.add_argument("--base", help="base dataset directory (default: config.DS_RELINKED_YR_V2)")
    p.add_argument("--dst", help="output directory (default: config.DS_DAIME_EVAL_FULL)")
    p.add_argument("--work", help="multidaime_clean intermediate directory (default: <dst>_multidaime_clean)")
    p.set_defaults(func=cmd_daime_eval)

    p = sub.add_parser("hipe", help="public dataset: download HIPE-2022 + convert to NEL JSONL")
    p.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    p.add_argument("--datasets", nargs="*", default=list(HipeBuilder.EL_DATASETS))
    p.add_argument("--window", type=int, default=400, help="context chars each side (0=full doc)")
    p.add_argument("--no_download", action="store_true", help="skip clone, convert existing repo")
    p.set_defaults(func=cmd_hipe)

    p = sub.add_parser("tag", help="tag multi-candidate (same surface >=2 gold) mentions with metadata.tags")
    p.add_argument("--tag", default="multi_cand", help="tag name (default: multi_cand)")
    p.add_argument("--datasets", nargs="*", help="default: yakusya_nel + each HIPE set")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("recut", help="re-cut the context window (store wide w300, truncate via Dataset.load(window=W))")
    p.add_argument("dataset", help="config.DATASETS name (daime family)")
    p.add_argument("--win", type=int, default=300, help="window width (chars), default 300")
    p.set_defaults(func=cmd_recut)

    p = sub.add_parser("merge-caches", help="merge per-model LLM cleaning caches -> entities_enriched.jsonl")
    p.add_argument(
        "--input", default=f"{config.WS}/BLINK/dataset_ja/yakusya_with_theaters/entities.jsonl"
    )
    p.add_argument(
        "--enriched",
        default=f"{config.WS}/BLINK/dataset_ja/yakusya_with_theaters/entities_enriched.jsonl",
    )
    p.add_argument(
        "--cache-glob",
        default=f"{config.WS}/BLINK/dataset_ja/yakusya_with_theaters/_llm_clean_cache_*.jsonl",
        help="glob matching per-model cache files",
    )
    p.set_defaults(func=cmd_merge_caches)

    p = sub.add_parser("list", help="print the dataset registry (config.DATASETS)")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
