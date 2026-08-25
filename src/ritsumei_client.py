#!/usr/bin/env python3
"""
Ritsumeikan Person Database query client

This library wraps access to Ritsumeikan University's kabuki (歌舞伎) actor
database, providing:
- Actor information search
- Detail page retrieval
- Local cache management
- HTML parsing
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

import config

__version__ = "1.0.0"
__author__ = "Japanese NEL Team"

# ============================================================================
# Configuration constants
# ============================================================================

# API configuration
BASE_URL = "https://www.dh-jac.net/db/shumei"
SEARCH_ENDPOINT = f"{BASE_URL}/results.php"
DETAIL_ENDPOINT = f"{BASE_URL}/results-big.php"

# Default local cache location. config.PROJ_DATA is <project>/data (overridable via
# YAKUSYA_DATA_ROOT), so the ARC-DB search cache is found without callers passing a path.
DEFAULT_CACHE_FILE = os.path.join(config.PROJ_DATA, "dataset_ja", "ritsumei_pd_cache.json")

# Default parameters
DEFAULT_MAX_RESULTS = 200  # Default maximum number of results
REQUEST_DELAY = 3.0  # Request delay (seconds) to avoid overloading the server
DEFAULT_TIMEOUT = 10  # Request timeout (seconds)

# HTTP Headers
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# ============================================================================
# URL utility functions
# ============================================================================


def build_detail_url(detail_id: str, use_cache: bool = True, cache_file: str = None) -> str:
    """
    Build the detail-page URL for the Ritsumeikan database from a detail_id

    Args:
        detail_id: the actor's detail_id (e.g. "80039")
        use_cache: whether to fetch the full URL from cache (default True)
        cache_file: cache file path (optional)

    Returns:
        the full detail-page URL

    Examples:
        >>> build_detail_url("80039")
        'https://www.dh-jac.net/db/shumei/results-big.php?f43[]=...&tmpid=80039'
        >>> build_detail_url("80039", use_cache=False)
        'https://www.dh-jac.net/db/shumei/results-big.php?tmpid=80039'
    """
    if not detail_id:
        return ""

    # Strip any surrounding whitespace
    detail_id = str(detail_id).strip()

    # If using cache, try to fetch the full URL from cache
    if use_cache:
        full_url = get_full_url_from_cache(detail_id, cache_file)
        if full_url:
            return full_url

    # Fall back to the simplified URL (using the tmpid parameter)
    return f"{DETAIL_ENDPOINT}?tmpid={detail_id}"


def get_full_url_from_cache(detail_id: str, cache_file: str = None) -> str | None:
    """
    Fetch the full detail_url from cache

    Args:
        detail_id: the actor's detail_id
        cache_file: cache file path (optional)

    Returns:
        the full URL, or None if not found
    """
    if cache_file is None:
        cache_file = DEFAULT_CACHE_FILE

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)

        # Look up a matching detail_id in the cache
        for query, data in cache.items():
            if isinstance(data, dict) and "results" in data:
                for result in data["results"]:
                    if str(result.get("detail_id", "")) == str(detail_id):
                        detail_url = result.get("detail_url", "")
                        if detail_url:
                            # Convert relative URL to absolute URL
                            if detail_url.startswith("./"):
                                return f"{BASE_URL}/{detail_url[2:]}"
                            elif not detail_url.startswith("http"):
                                return f"{BASE_URL}/{detail_url}"
                            return detail_url

        return None
    except Exception:
        # If reading the cache fails, return None
        return None


def normalize_detail_url(url: str) -> str:
    """
    Normalize a detail_url, ensuring it is a full URL

    Args:
        url: relative or absolute URL

    Returns:
        the full URL
    """
    if not url:
        return ""

    # Already a full URL
    if url.startswith("http"):
        return url

    # Strip a leading ./
    if url.startswith("./"):
        url = url[2:]

    # Prepend the base URL
    return f"{BASE_URL}/{url}"


# ============================================================================
# Cache management
# ============================================================================


class CacheManager:
    """Local cache manager"""

    def __init__(self, cache_file: str | None = None):
        """
        Initialize the cache manager

        Args:
            cache_file: cache file path
        """
        self.cache_file = cache_file
        self.cache = self._load() if cache_file else {}

    def _load(self) -> dict:
        """Load the cache file"""
        if not os.path.exists(self.cache_file):
            return {}

        try:
            with open(self.cache_file, encoding="utf-8") as f:
                cache = json.load(f)

            # Automatically migrate old-format cache
            migrated = self._migrate_old_format(cache)
            if migrated > 0:
                print(f"  ℹ️  migrated {migrated} old-format cache entries")
                self.save()

            return cache

        except Exception as e:
            print(f"⚠️  failed to read cache file: {e}")
            return {}

    def _migrate_old_format(self, cache: dict) -> int:
        """
        Migrate old-format cache
        Old format: {actor_name: [results...]}
        New format: {actor_name: {'max_results': N, 'results': [...]}}
        """
        migrated = 0
        for key, value in list(cache.items()):
            if key.startswith("detail_"):
                continue

            if isinstance(value, list):
                cache[key] = {"max_results": DEFAULT_MAX_RESULTS, "results": value}
                migrated += 1

        return migrated

    def save(self):
        """Save the cache to file"""
        if not self.cache_file:
            return

        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  failed to save cache file: {e}")

    def get_query(self, actor_name: str, max_results: int) -> tuple[list[dict], bool] | None:
        """
        Get a cached query

        Args:
            actor_name: actor name
            max_results: the maximum number of results requested

        Returns:
            (results list, True) if the cache is sufficient
            None if a re-query is needed
        """
        if actor_name not in self.cache:
            return None

        cached_data = self.cache[actor_name]

        # New format
        if isinstance(cached_data, dict) and "results" in cached_data:
            cached_max = cached_data.get("max_results", 0)
            cached_results = cached_data["results"]

            if max_results <= cached_max:
                return cached_results[:max_results], True

        return None

    def set_query(self, actor_name: str, max_results: int, results: list[dict]):
        """Set a query cache entry"""
        self.cache[actor_name] = {"max_results": max_results, "results": results}

    def get_detail(self, detail_id: str) -> dict | None:
        """Get a cached detail entry"""
        cache_key = f"detail_{detail_id}"
        return self.cache.get(cache_key)

    def set_detail(self, detail_id: str, detail_info: dict):
        """Set a detail cache entry"""
        cache_key = f"detail_{detail_id}"
        self.cache[cache_key] = detail_info

    def __len__(self):
        """Return the number of cache entries"""
        return len(self.cache)


# ============================================================================
# HTML parser
# ============================================================================


class RitsumeiParser:
    """Ritsumeikan database HTML parser"""

    @staticmethod
    def parse_search_results(html_content: str) -> list[dict]:
        """
        Parse the search results page

        Returns:
            a list of actor info, each item containing:
            - name: name
            - reading: reading (pronunciation)
            - generation: generation (代数)
            - category: category
            - birth_year, death_year: birth/death years
            - detail_id, detail_url: detail link
            - etc.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        results = []
        rows = soup.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            try:
                actor_info = RitsumeiParser._parse_result_row(cells)
                if actor_info and actor_info.get("name"):
                    results.append(actor_info)
            except Exception:
                continue

        return results

    @staticmethod
    def _parse_result_row(cells: list) -> dict | None:
        """Parse a single search result row"""
        # Column 1: detail button
        detail_id, detail_url = RitsumeiParser._extract_detail_info(cells[0])

        # Column 2: name information
        name_info = RitsumeiParser._extract_name_info(cells[1])

        # Column 3: active period
        active_period = cells[2].get_text(strip=True)
        active_start_year, active_end_year = RitsumeiParser._extract_period_years(active_period)

        # Column 4: birth/death years
        birth_death_text = cells[3].get_text(strip=True)
        birth_info, death_info, age = RitsumeiParser._extract_birth_death(birth_death_text)

        # Column 5: relationships
        relationships = cells[4].get_text(strip=True) if len(cells) > 4 else ""
        later_name, final_daime = RitsumeiParser._extract_later_name(relationships)

        # Build the result
        return {
            "name": name_info["name"],
            "reading": name_info["reading"],
            "generation": name_info["generation"],
            "category": name_info["category"],
            "active_period": active_period,
            "active_start_year": active_start_year,
            "active_end_year": active_end_year,
            "birth_year": birth_info["year"],
            "birth_nengo": birth_info["nengo"],
            "death_year": death_info["year"],
            "death_nengo": death_info["nengo"],
            "age": age,
            "relationships": relationships,
            "later_name": later_name,
            "final_daime": final_daime,
            "daime": final_daime or name_info["generation"],
            "description": f"{name_info['name']}（{name_info['reading']}）は、{name_info['category']}。{birth_death_text}",
            "detail_id": detail_id,
            "detail_url": detail_url,
            "person_record_id": name_info["person_record_id"],
        }

    @staticmethod
    def _extract_detail_info(cell) -> tuple[str | None, str | None]:
        """Extract the detail ID and full URL"""
        detail_link = cell.find("a")
        if not detail_link:
            return None, None

        onclick = detail_link.get("onclick", "")
        url_match = re.search(r"MssgWindow1\('([^']+)'\)", onclick)
        if not url_match:
            return None, None

        detail_url = url_match.group(1)

        # Convert to a full URL
        if detail_url.startswith("./"):
            detail_url = detail_url[2:]
        if not detail_url.startswith("http"):
            detail_url = f"https://www.dh-jac.net/db/shumei/{detail_url}"

        tmpid_match = re.search(r"tmpid=(\d+)", detail_url)
        detail_id = tmpid_match.group(1) if tmpid_match else None

        return detail_id, detail_url

    @staticmethod
    def _extract_name_info(cell) -> dict:
        """Extract name-related information"""
        cell_text = cell.get_text(strip=True)

        # Extract the queried name and generation (代数)
        queried_match = re.search(
            r"〈\s*(\d+)\s*〉\s*([^\s（]+(?:\s+[^\s（]+)*?)\s*(?:（代表名|$)", cell_text
        )
        if queried_match:
            generation = queried_match.group(1)
            queried_name = queried_match.group(2).strip()
        else:
            simple_match = re.search(r"〈\s*(\d+)\s*〉\s*([^\s]+(?:\s+[^\s]+)*)", cell_text)
            if simple_match:
                generation = simple_match.group(1)
                queried_name = simple_match.group(2).strip()
            else:
                generation = "1"
                queried_name = None

        # Extract person_record_id
        person_record_id = None
        name_link = cell.find("a")
        if name_link and name_link.get("href"):
            person_match = re.search(r"-recid=([^&]+)", name_link.get("href"))
            if person_match:
                person_record_id = person_match.group(1)

        # Extract reading, name, category
        reading = ""
        name = queried_name or ""
        category = ""

        for link in cell.find_all("a"):
            href = link.get("href", "")

            if "f1=" in href:  # reading
                reading = (
                    link.find("span").get_text(strip=True)
                    if link.find("span")
                    else link.get_text(strip=True)
                )
            elif "f2=" in href and "f3=" not in href and not queried_name:  # name
                name = (
                    link.find("span").get_text(strip=True)
                    if link.find("span")
                    else link.get_text(strip=True)
                )
            elif "f33=" in href:  # category
                category = (
                    link.find("span").get_text(strip=True)
                    if link.find("span")
                    else link.get_text(strip=True)
                )

        return {
            "name": name,
            "reading": reading,
            "generation": generation,
            "category": category,
            "person_record_id": person_record_id,
        }

    @staticmethod
    def _extract_period_years(period_text: str) -> tuple[int | None, int | None]:
        """Extract the years of the active period"""
        if not period_text:
            return None, None

        parts = re.split(r"[〜～]", period_text)

        start_year = None
        if len(parts) >= 1:
            start_years = re.findall(r"（(\d{4})）", parts[0])
            if start_years:
                start_year = int(start_years[0])

        end_year = None
        if len(parts) >= 2:
            end_years = re.findall(r"（(\d{4})）", parts[1])
            if end_years:
                end_year = int(end_years[0])

        return start_year, end_year

    @staticmethod
    def _extract_birth_death(text: str) -> tuple[dict, dict, int | None]:
        """Extract birth/death year information"""
        birth_info = {"year": None, "nengo": ""}
        death_info = {"year": None, "nengo": ""}
        age = None

        # Birth
        birth_match = re.search(r"（(\d{4})）.*?生", text)
        if birth_match:
            birth_info["year"] = int(birth_match.group(1))
            nengo_match = re.search(r"([^（]+)（" + str(birth_info["year"]) + r"）.*?生", text)
            if nengo_match:
                birth_info["nengo"] = nengo_match.group(1).strip()

        # Death
        death_match = re.search(r"[〜～].*?（(\d{4})）.*?没", text)
        if death_match:
            death_info["year"] = int(death_match.group(1))
            nengo_match = re.search(
                r"[〜～].*?([^（]+)（" + str(death_info["year"]) + r"）.*?没", text
            )
            if nengo_match:
                death_info["nengo"] = nengo_match.group(1).strip()

        # Age
        age_match = re.search(r"享年\s*(\d+)", text)
        if age_match:
            age = int(age_match.group(1))

        return birth_info, death_info, age

    @staticmethod
    def _extract_later_name(relationships: str) -> tuple[str, str]:
        """Extract the final name and generation (代数) from the relationships column"""
        if not relationships:
            return "", ""

        later_match = re.search(r"〈(\d+)〉([^〈〉]+)", relationships)
        if later_match:
            return later_match.group(2).strip(), later_match.group(1)

        return "", ""

    @staticmethod
    def _extract_correct_daime(text: str, target_name: str) -> str | None:
        """
        Extract the correct generation (代数) from the （代数）人名 section

        This is the most accurate source of the generation, because the
        title generation in the Ritsumeikan database is sometimes inaccurate

        Args:
            text: the full text
            target_name: the target name

        Returns:
            the generation string (e.g. "3b"), or None if not found
        """
        if "（代数）人名" not in text:
            return None

        # Extract the （代数）人名 section
        start = text.find("（代数）人名")
        end = text.find("表示できる年表", start)
        if end == -1:
            end = len(text)

        section = text[start:end]

        # Clean up the target name
        clean_target = target_name.replace("　", "").replace(" ", "").replace("(歌舞伎 役者)", "")

        # Try several matching patterns
        patterns = [
            rf"〈([^〉]+)〉{re.escape(target_name)}",  # exact match
            rf"〈([^〉]+)〉{re.escape(clean_target)}",  # whitespace-stripped match
            rf"〈([^〉]+)〉{re.escape(clean_target.split('〈')[0])}",  # partial match
        ]

        for pattern in patterns:
            matches = re.findall(pattern, section)
            if matches:
                return matches[0]

        return None

    @staticmethod
    def parse_detail_page(html_content: str) -> dict:
        """
        Parse the detail page

        Returns:
            a dict of detailed information, containing:
            - queried_name, queried_daime: queried name and generation (extracted from the title)
            - correct_daime: correct generation (extracted from the （代数）人名 section, most accurate)
            - daime_mismatch: whether the title generation and correct generation disagree
            - representative_name, representative_daime: representative name (代表名)
            - birth_year, death_year: birth/death years
            - kaimeihyou: name-change table (改名表)
            - full_text: the full text
            - etc.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        text_content = soup.get_text(separator="\n", strip=True)
        full_text_raw = "\n".join(line.strip() for line in text_content.split("\n") if line.strip())

        result = {
            "queried_name": "",
            "queried_daime": "",
            "queried_reading": "",
            "representative_name": "",
            "representative_daime": "",
            "category": "",
            "birth_year": None,
            "birth_nengo": "",
            "birth_month_day": "",
            "death_year": None,
            "death_nengo": "",
            "death_month_day": "",
            "age": None,
            "real_name": "",
            "haiku_names": "",
            "yago": "",
            "family": "",
            "kaimeihyou": [],
            "full_text": RitsumeiParser._clean_ui_text(full_text_raw),
            "correct_daime": None,  # correct generation extracted from （代数）人名
            "daime_mismatch": False,  # title generation disagrees with text generation
        }

        # Extract information for each section
        RitsumeiParser._extract_header_info(full_text_raw, result)
        RitsumeiParser._extract_birth_death_info(full_text_raw, result)
        RitsumeiParser._extract_detail_fields(full_text_raw, result)
        RitsumeiParser._extract_kaimeihyou(soup, result)

        # Find the reading from the name-change table (改名表)
        if result["queried_name"] and not result["queried_reading"]:
            for entry in result["kaimeihyou"]:
                if (
                    entry["daime"] == result["queried_daime"]
                    and result["queried_name"] in entry["name"]
                ):
                    result["queried_reading"] = entry["reading"]
                    break

        # Extract the correct generation (from the （代数）人名 section)
        if result["queried_name"]:
            base_name = result["queried_name"].replace("(歌舞伎 役者)", "").strip()
            correct_daime = RitsumeiParser._extract_correct_daime(full_text_raw, base_name)

            if correct_daime:
                result["correct_daime"] = correct_daime

                # Check whether it disagrees with the title generation
                if correct_daime != result["queried_daime"]:
                    result["daime_mismatch"] = True

        return result

    @staticmethod
    def _extract_header_info(text: str, result: dict):
        """Extract the queried name and representative name (代表名) from the header"""
        # Format with a representative name
        pattern1 = r"〈\s*(\d+)\s*〉\s*([^\s（]+(?:\s+[^\s（]+)*?)\s*（代表名:\s*〈\s*(\d+)\s*〉\s*([^\s）]+(?:\s+[^\s）]+)*?)\s*）\s*\(([^)]+)\)"
        match1 = re.search(pattern1, text)

        if match1:
            result["queried_daime"] = match1.group(1)
            result["queried_name"] = match1.group(2).strip()
            result["representative_daime"] = match1.group(3)
            result["representative_name"] = match1.group(4).strip()
            result["category"] = match1.group(5).strip()
        else:
            # Format without a representative name
            pattern2 = r"〈\s*(\d+)\s*〉\s*([^\s（]+(?:\s+[^\s（]+)*?)\s*\(([^)]+)\)"
            match2 = re.search(pattern2, text)
            if match2:
                result["queried_daime"] = match2.group(1)
                result["queried_name"] = match2.group(2).strip()
                result["representative_name"] = result["queried_name"]
                result["representative_daime"] = result["queried_daime"]
                result["category"] = match2.group(3).strip()

    @staticmethod
    def _extract_birth_death_info(text: str, result: dict):
        """Extract birth/death year information"""
        pattern = r"([^\s]+)\s*（(\d{4}）)\s*([^生]+)生～\s*([^\s]+)\s*（(\d{4}）)\s*([^没]+)没.*?享年\s*(\d+)"
        match = re.search(pattern, text)

        if match:
            result["birth_nengo"] = match.group(1)
            result["birth_year"] = int(match.group(2).replace("）", ""))
            result["birth_month_day"] = match.group(3).strip()
            result["death_nengo"] = match.group(4)
            result["death_year"] = int(match.group(5).replace("）", ""))
            result["death_month_day"] = match.group(6).strip()
            result["age"] = int(match.group(7))

    @staticmethod
    def _extract_detail_fields(text: str, result: dict):
        """Extract detail fields"""
        # Real name (本名)
        match = re.search(r"本名[：:]\s*([^。\n]+)", text)
        if match:
            result["real_name"] = match.group(1).strip()

        # Haiku name (俳名)
        match = re.search(r"俳名[・·]?狂歌名[：:]\s*([^。\n]+)", text)
        if match:
            result["haiku_names"] = match.group(1).strip()

        # Shop name / yago (屋号)
        match = re.search(r"屋号[：:]\s*([^。\n]+)", text)
        if match:
            result["yago"] = match.group(1).strip()

        # Family lineage (家系)
        match = re.search(r"家系[：:]\s*([^。\n]+)", text)
        if match:
            result["family"] = match.group(1).strip()

    @staticmethod
    def _extract_kaimeihyou(soup, result: dict):
        """Extract the name-change table (改名表), taking only the 名乗期間 table for the current person.

        results-2p.htm shows one person per page, whose 名乗期間 appears in the
        first <table> containing name-change rows; later <table>s on the page are
        either a repeated rendering of the same table, or (when a page renders
        multiple people) belong to a **different** generation — historically it
        was the latter that merged the tenth-generation (十代目) rows into the
        first-generation (初代) record (see the _ls_clean_kaimeihyou comment in
        dataset.py). Therefore: accumulate the first table that yields name-change
        rows, stop as soon as that table has yielded rows, and never merge across
        tables."""
        tables = soup.find_all("table")
        seen_entries = set()

        for table in tables:
            produced = 0
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")

                if len(cells) >= 3:
                    col1_text = cells[0].get_text(strip=True)
                    col2_text = cells[1].get_text(strip=True)
                    col3_text = cells[2].get_text(strip=True)

                    if "〈" in col1_text and "〉" in col1_text and "件" not in col1_text:
                        kaime_match = re.search(
                            r"([^\s〈]+(?:\s+[^\s〈]+)*?)\s*〈(\d+)〉\s*(.+)", col1_text
                        )

                        if kaime_match:
                            reading = kaime_match.group(1).strip()
                            daime = kaime_match.group(2)
                            name = kaime_match.group(3).strip()

                            # Parse the period
                            period_parts = re.split(r"[～〜]", col2_text)
                            period_start = period_parts[0].strip() if len(period_parts) > 0 else ""
                            period_end = period_parts[1].strip() if len(period_parts) > 1 else ""

                            # Extract the years
                            start_year = None
                            start_match = re.search(r"（(\d{4})）", period_start)
                            if start_match:
                                start_year = int(start_match.group(1))

                            end_year = None
                            end_match = re.search(r"（(\d{4})）", period_end)
                            if end_match:
                                end_year = int(end_match.group(1))

                            # Deduplicate
                            entry_key = (name, daime, col3_text)
                            if entry_key not in seen_entries and len(name) < 30:
                                seen_entries.add(entry_key)
                                produced += 1
                                result["kaimeihyou"].append(
                                    {
                                        "reading": reading,
                                        "name": name,
                                        "daime": daime,
                                        "period_start": period_start,
                                        "period_end": period_end,
                                        "start_year": start_year,
                                        "end_year": end_year,
                                        "name_type": col3_text,
                                    }
                                )
            # Keep only the current person's first 名乗期間 table, blocking cross-person merges
            if produced:
                break

    @staticmethod
    def _clean_ui_text(text: str) -> str:
        """Clean UI elements out of the text"""
        ui_patterns = [
            r"﻿",
            r"基本情報表示",
            r"詳細情報表示",
            r"一覧表示に戻る",
            r"ArtWiki",
            r"前の人物",
            r"新規検索",
            r"次の人物",
            r"文化DigiL",
            r"Wikipedia\.jp",
            r"Google",
            r"Permalink[：:][^\n]*",
            r"肖像[：:][^\n]*",
            r"写真DB",
            r"浮世絵DB",
            r"人名典拠DB",
            r"年譜DB",
            r"UsrMemo",
            r"BookText",
            r"UkiyoeTxt",
            r"NIJL著作一覧",
            r"NDL著者",
            r"東文研物故者",
            r"JBDB",
            r"詳細情報",
            r"改名表",
            r"襲名グラフ",
            r"\d+\s*件の内",
            r"\d+\s*件目を表示中",
            r"←",
            r"→",
        ]

        cleaned = text
        for pattern in ui_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Clean up short lines
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        cleaned_lines = [
            line
            for line in lines
            if len(line) > 3 or re.search(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", line)
        ]

        # Reassemble paragraphs
        result_lines = []
        current_paragraph = []

        for line in cleaned_lines:
            current_paragraph.append(line)
            if line and (line[-1] in "。．、，！？）」" or "（" in line or "）" in line):
                result_lines.append("".join(current_paragraph))
                current_paragraph = []

        if current_paragraph:
            result_lines.append("".join(current_paragraph))

        cleaned = "\n".join(result_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()


# ============================================================================
# Main client class
# ============================================================================


class RitsumeiClient:
    """Ritsumeikan Person Database query client"""

    def __init__(self, cache_file: str | None = None, request_delay: float = REQUEST_DELAY):
        """
        Initialize the client

        Args:
            cache_file: cache file path (optional)
            request_delay: request delay in seconds
        """
        self.cache_manager = CacheManager(cache_file)
        self.parser = RitsumeiParser()
        self.request_delay = request_delay

    def search(
        self, actor_name: str, max_results: int = DEFAULT_MAX_RESULTS
    ) -> tuple[list[dict], bool]:
        """
        Search for actor information

        Args:
            actor_name: actor name (e.g. "市川海老蔵")
            max_results: maximum number of results to return

        Returns:
            (actor info list, whether it came from cache)
        """
        print(f"\n🔍 querying actor: {actor_name} (max_results={max_results})")

        # Check the cache
        cached = self.cache_manager.get_query(actor_name, max_results)
        if cached:
            results, from_cache = cached
            print(f"✓ loaded from cache ({len(results)} entries)")
            return results, from_cache

        # API query
        params = {
            "-format": "results-1p.htm",
            "enter": "default",
            "-max": str(max_results),
            "f43": actor_name,
            "-Find": "Search",
        }

        print(f"URL: {SEARCH_ENDPOINT}?f43={actor_name}")

        try:
            response = requests.get(
                SEARCH_ENDPOINT, params=params, headers=DEFAULT_HEADERS, timeout=DEFAULT_TIMEOUT
            )
            response.encoding = "utf-8"
            time.sleep(self.request_delay)

            if response.status_code == 200:
                print("✓ request ok")
                results = self.parser.parse_search_results(response.text)

                # Save the cache
                self.cache_manager.set_query(actor_name, max_results, results)
                self.cache_manager.save()

                return results, False
            else:
                print(f"✗ request failed: {response.status_code}")
                return [], False

        except Exception as e:
            print(f"✗ query error: {e}")
            return [], False

    def get_detail(self, detail_id_or_url: str) -> tuple[dict | None, bool]:
        """
        Get an actor's detailed information

        Args:
            detail_id_or_url: detail record ID (tmpid) or full URL

        Returns:
            (detail info dict, whether it came from cache)
        """
        if not detail_id_or_url:
            return None, False

        # Extract detail_id
        if isinstance(detail_id_or_url, str) and (
            "http" in detail_id_or_url or "/" in detail_id_or_url
        ):
            detail_url = detail_id_or_url
            tmpid_match = re.search(r"tmpid=(\d+)", detail_url)
            detail_id = tmpid_match.group(1) if tmpid_match else detail_url
        else:
            detail_id = detail_id_or_url
            detail_url = None

        # Check the cache
        cached_detail = self.cache_manager.get_detail(detail_id)
        if cached_detail:
            return cached_detail, True

        # API query
        try:
            if detail_url:
                if detail_url.startswith("./"):
                    detail_url = detail_url[2:]
                if not detail_url.startswith("http"):
                    detail_url = f"{BASE_URL}/{detail_url}"

                import html

                detail_url = html.unescape(detail_url)
                response = requests.get(
                    detail_url, headers=DEFAULT_HEADERS, timeout=DEFAULT_TIMEOUT
                )
            else:
                params = {"tmpid": detail_id, "-format": "results-2p.htm", "enter": "default"}
                response = requests.get(
                    DETAIL_ENDPOINT, params=params, headers=DEFAULT_HEADERS, timeout=DEFAULT_TIMEOUT
                )

            time.sleep(self.request_delay)
            response.encoding = "utf-8"

            if response.status_code == 200:
                detail_info = self.parser.parse_detail_page(response.text)

                # Save the cache
                self.cache_manager.set_detail(detail_id, detail_info)
                self.cache_manager.save()

                return detail_info, False
            else:
                print(f"✗ detail-page request failed: {response.status_code}")
                return None, False

        except Exception as e:
            print(f"✗ error fetching detail page: {e}")
            return None, False

    @property
    def cache_size(self) -> int:
        """Return the cache size"""
        return len(self.cache_manager)


# ============================================================================
# Backward-compatible function interfaces
# ============================================================================

# Global default client instance
_default_client = None


def load_cache(cache_file: str) -> dict:
    """Load the cache (backward-compatible interface)"""
    manager = CacheManager(cache_file)
    return manager.cache


def save_cache(cache_file: str, cache_data: dict):
    """Save the cache (backward-compatible interface)"""
    manager = CacheManager(cache_file)
    manager.cache = cache_data
    manager.save()


def query_ritsumei_db(
    actor_name: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    cache: dict | None = None,
    cache_file: str | None = None,
) -> tuple[list[dict], bool]:
    """
    Query the Ritsumeikan database (backward-compatible interface)

    Args:
        actor_name: actor name
        max_results: maximum number of results
        cache: cache dict (deprecated, kept for compatibility)
        cache_file: cache file path

    Returns:
        (actor info list, whether it came from cache)
    """
    global _default_client
    if _default_client is None or _default_client.cache_manager.cache_file != cache_file:
        _default_client = RitsumeiClient(cache_file)

    return _default_client.search(actor_name, max_results)


def fetch_actor_detail(
    detail_id_or_url: str, cache: dict | None = None, cache_file: str | None = None
) -> tuple[dict | None, bool]:
    """
    Get an actor's detailed information (backward-compatible interface)

    Args:
        detail_id_or_url: detail ID or URL
        cache: cache dict (deprecated, kept for compatibility)
        cache_file: cache file path

    Returns:
        (detail info dict, whether it came from cache)
    """
    global _default_client
    if _default_client is None or _default_client.cache_manager.cache_file != cache_file:
        _default_client = RitsumeiClient(cache_file)

    return _default_client.get_detail(detail_id_or_url)


# ============================================================================
# Test code
# ============================================================================


def main():
    """Test the query functionality"""
    print("=" * 70)
    print("Ritsumeikan person database — query client test")
    print("=" * 70)

    cache_file = DEFAULT_CACHE_FILE

    client = RitsumeiClient(cache_file)
    print(f"\n💾 cache file: {cache_file}")
    print(f"   cached: {client.cache_size} entries")

    # Test the search
    test_names = ["市川海老蔵", "市川団十郎", "中村歌右衛門"]

    for name in test_names:
        results, from_cache = client.search(name, max_results=3)

        print(f"\nresults: {len(results)} entries")

        for i, result in enumerate(results[:3], 1):
            print(f"\n  {i}. {result.get('name', 'N/A')}")
            print(f"     reading: {result.get('reading', 'N/A')}")
            print(f"     generation: 〈{result.get('generation', '')}〉")
            print(f"     category: {result.get('category', 'N/A')}")
            if result.get("birth_year"):
                print(
                    f"     birth: {result.get('birth_nengo', '')} ({result.get('birth_year', '')})"
                )
            if result.get("death_year"):
                print(
                    f"     death: {result.get('death_nengo', '')} ({result.get('death_year', '')})"
                )

    print("\n" + "=" * 70)
    print(f"💾 final cache: {client.cache_size} entries")
    print("=" * 70)


if __name__ == "__main__":
    main()
