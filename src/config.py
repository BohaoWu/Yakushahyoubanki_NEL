#!/usr/bin/env python3
"""Central run configuration for the yakusya NEL models.

Single source of truth, loaded by run_all.py (and nel_models.py) at startup:
  - the workspace root + .env loading,
  - per-method repo / venv / script paths,
  - MODELS: each model's default run hyperparameters.

Edit a model's defaults here rather than in the class code. Shell environment
variables always win over values loaded from .env.
"""

import os
import sys

# ---- workspace root + .env ----
# Project root (the repo checkout) and its src/, robust to WS override.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")


def _setup_methods_path():
    """Put ``src/methods`` and all its subpackages on ``sys.path``.

    The candidate / disambiguator modules under ``src/methods`` import each other
    by bare module name, so every subfolder must be importable. Importing ``config`` runs this once, so callers only need
    ``src`` on the path plus ``import config`` instead of hand-listing the
    subfolders. Idempotent.
    """
    methods = os.path.join(SRC_DIR, "methods")
    if not os.path.isdir(methods):
        return
    for d in (methods, *(os.path.join(methods, s) for s in os.listdir(methods))):
        if os.path.isdir(d) and not os.path.basename(d).startswith("__") and d not in sys.path:
            sys.path.insert(0, d)


_setup_methods_path()

_WS_DEFAULT = os.environ.get("YAKUSYA_WORKSPACE", os.path.dirname(PROJECT_DIR))


def _load_dotenv(path):
    """Load KEY=VALUE pairs from `path` into os.environ without overriding vars
    already set (in the shell or a higher-priority .env loaded earlier)."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


# The project's own .env takes priority; BLINK/.env fills any keys it doesn't set.
_load_dotenv(os.path.join(PROJECT_DIR, ".env"))
_load_dotenv(os.path.join(_WS_DEFAULT, "BLINK", ".env"))

WS = os.environ.get("YAKUSYA_WORKSPACE", os.path.dirname(PROJECT_DIR))

# Reproducibility seed for the 4 dataset splits (keep at 52313 to reproduce results).
SPLIT_SEED = int(os.environ.get("YAKUSYA_SPLIT_SEED", "52313"))

# ---- per-method repos, venvs, scripts ----
BLINK_DIR = f"{WS}/BLINK"
BLINK_PY = os.environ.get("BLINK_PY", "/usr/bin/python3")
GENRE_DIR = f"{WS}/GENRE"
MGENRE_PY = os.environ.get("MGENRE_PY", f"{GENRE_DIR}/venv_yakusya/bin/python")
MGENRE_TRAIN_PY = f"{GENRE_DIR}/scripts_yakusya/run_yakusya.py"
MGENRE_BLINK10_PY = f"{GENRE_DIR}/scripts_yakusya/predict_mgenre_blink10_split.py"
REFINED_DIR = f"{WS}/ReFinED"
REFINED_PY = os.environ.get("REFINED_PY", f"{REFINED_DIR}/venv/bin/python")
REFINED_TRAIN_PY = f"{REFINED_DIR}/run_yakusya.py"
REFINED_BLINK10_PY = f"{REFINED_DIR}/predict_yakusya_blink_cands_split.py"
ONENET_PY = os.environ.get("ONENET_PY", f"{WS}/OneNet/.venv/bin/python")
LLMD_PY = os.environ.get("LLMD_PY", f"{WS}/llm_disambiguator/venv/bin/python")

# Other method repos / venvs / scripts (not in the NELModel framework yet).
DYVO_DIR = f"{WS}/DyVo"
DYVO_PY = os.environ.get("DYVO_PY", f"{DYVO_DIR}/venv/bin/python")
DYVO_TRAIN_PY = f"{DYVO_DIR}/lsr/train.py"
DYVO_PREP_PY = f"{DYVO_DIR}/scripts/prepare_yakusya.py"

# ---- project directory layout ----
PROJ_SRC = SRC_DIR
PROJ_DATA = os.path.join(PROJECT_DIR, "data")
PROJ_EXP = os.path.join(PROJECT_DIR, "experiment")
PROJ_LOG = os.path.join(PROJECT_DIR, "log")
PROJ_PIC = os.path.join(PROJECT_DIR, "pic")

# The 5 train/test splits.
SCHEMES = ["original", "pair_disjoint", "entity_disjoint", "temporal", "book_disjoint"]

# ---- dataset variant selection (YAKUSYA_DATASET) ----
# Setting YAKUSYA_DATASET=<v> routes everything to the parallel dataset_<v> /
# experiments_<v> / predictions_<v> / logs_<v> / models_<v> layout (unset = the
# original yakusya_with_theaters). Recognized variants:
#   clean              theater truncation fixed (BLINK/dataset_clean)
#   arc                ARC-only subset (entities -> 3244 Ritsumeikan-DB, ~5% of data)
#   fix                clean + canonicalize label_title + disambiguate dup-title entities
#   relinked           fix + year-based mention->entity re-linking (kaimeihyou periods)
#   relinked_yr        relinked + "[年:N|書:X]" prefix baked into context_left
#   relinked_yr_v1     relinked_yr re-link iteration 1 (the trained-models baseline)
#   relinked_yr_v2     v1 + entity_canon A-only (same-source) dedup applied
#                      physically: 5000->4885 entities, gold remapped in all
#                      splits (build_dataset.py's build_dedup()
#                      --mode source). A-only keeps every gold's surface name +
#                      daime unchanged (safe for training). B (cross-name same-
#                      person) is intentionally NOT baked in -- it would rewrite
#                      986 gold names / 587 daime; use it only at eval scoring
#                      time via `eval --canon full`.
#   relinked_yr_v{3,...}  further iterations over relinked_yr (prefix match)
#   multidaime_clean   multi-generation (代目) in-domain probe: each split is the
#                      multi-daime subset of the matching relinked_yr_v1 split,
#                      cleaned so every gold is answerable (build/
#                      build_multidaime_clean.py). train 2350 / valid 304 / test
#                      312, mutually disjoint, v1 boundaries preserved.
#   daime_eval_full    clean 代目-disambiguation benchmark + the DEFAULT dataset.
#                      EVAL-ONLY (test only, no train): test = 2214 multi-candidate
#                      actor mentions (>=2 ritsumei generations of the name) whose
#                      gold is resolved to a specific generation. It spans the whole
#                      corpus, so it has NO disjoint same-corpus train — it is a
#                      benchmark for NON-training methods (mad/transmad/llm_tool),
#                      which can use all 2214 leak-free. Each row tags metadata.
#                      origin_split + heldout: the 458 heldout=true rows (valid/test
#                      origin, also dumped as test_heldout.jsonl) are the only ones a
#                      TRAINED model (mGENRE/BLINK, which saw the 1756 train-origin
#                      rows) may be fairly scored on. For training / trained-model
#                      runs use YAKUSYA_DATASET=relinked_yr_v1.
#   daime_eval         the official frozen 1479-mention subset of daime_eval_full,
#                      kept as an OPTION (YAKUSYA_DATASET=daime_eval) for comparing
#                      against the published 1479 numbers. Same shape (test-only,
#                      1175 train-origin + 304 heldout). It is NOT bit-reproducible
#                      from the current pipeline — build_dataset.py now emits the
#                      2214; see that file's header.
_DATASET_VARIANTS = (
    "clean",
    "arc",
    "fix",
    "relinked",
    "relinked_yr",
    # the default: full 代目 eval benchmark (2214 mentions, eval-only)
    "daime_eval_full",
    # frozen 1479 subset of daime_eval_full (published numbers)
    "daime_eval",
    # trainable split of daime_eval_full by origin_split, recut to
    # w100 (train 1756 / valid 228 / test 230); test is held-out
    "daime_ft_w100",
    # public historical-NEL benchmarks per language (Wikidata-QID,
    # eval-only) converted to numeric_id nel format by
    # HipeBuilder.to_kb — entities.jsonl = KB of the gold QIDs
    "ajmc_de",
    "ajmc_en",
    "ajmc_fr",
    "hipe2020_de",
    "hipe2020_en",
    "hipe2020_fr",
    "newseye_de",
    "newseye_fi",
    "newseye_fr",
    "newseye_sv",
    "sonar_de",
    "topres19th_en",
    # Mahānāma: Sanskrit (SLP1) EDL over the Mahābhārata, English
    # KB with name-variant clusters; built by MahanamaBuilder.convert
    "mahanama",
)


def is_variant(ds):
    """True if YAKUSYA_DATASET names a recognized dataset variant."""
    return bool(ds) and (ds in _DATASET_VARIANTS or ds.startswith("relinked_yr_v"))


# Default (original) data/experiment layout; a recognized variant or the explicit
# YAKUSYA_DATA_ROOT / YAKUSYA_EXPERIMENTS_ROOT env vars override it.
ORIG_DATA = os.path.join(PROJ_DATA, "dataset_ja/yakusya_with_theaters")
LLMAEL_DATA_DIR = os.path.join(PROJ_DATA, "dataset_ja/yakusya_with_theaters_llmael")
EXPERIMENTS = os.path.join(PROJ_EXP, "experiments")
PRED_DIR = os.path.join(PROJ_EXP, "predictions")
LOG_DIR = os.path.join(PROJ_LOG, "logs")

# Default dataset = the clean 代目-disambiguation eval probe, full 2214 version
# (override with the YAKUSYA_DATASET env var: daime_eval for the frozen 1479 the
# published numbers were measured on, relinked_yr_v1 for training/trained-model
# runs). Used both here and in scheme_paths so the whole layout stays consistent.
_DEFAULT_DATASET = "daime_eval_full"
_DATASET = os.environ.get("YAKUSYA_DATASET", _DEFAULT_DATASET)


# ============================================================================
# Dataset path registry — ONE variable per dataset dir. src/build_dataset.py
# builds/downloads into these; other code should reference the variable, never a
# hardcoded path. Most live at data/dataset_<name>; a few sit under data/dataset/.
# ============================================================================
def _dd(name):
    """data/dataset_<name> — the default location convention."""
    return os.path.join(PROJ_DATA, f"dataset_{name}")


_DDIR = os.path.join(PROJ_DATA, "dataset")  # things physically under data/dataset/

# --- raw source (non-public: 役者評判記 annotated CSV + external Ritsumeikan DB) ---
DS_RAW_CSV = os.path.join(_DDIR, "yakusya_annotated_data")
DS_JA = os.path.join(PROJ_DATA, "dataset_ja", "yakusya_with_theaters")

# --- full yakusya NEL dataset (relinked_yr_v1 clean core, corrected + deduped) ---
# 12168 mentions (all types incl. theaters) / 4885 entities. Built from
# relinked_yr_v1's clean core, then: (1) 1129 daime golds corrected via
# dataset_daime_eval_full (metadata.gold_corrected_from), (2) A-only entity_canon
# dedup applied (5000->4885, same-source duplicates merged, gold remapped).
DS_YAKUSYA_NEL = os.path.join(_DDIR, "yakusya_nel")

# --- yakusya cleanup / relink chain (non-public; built from base via recovery) ---
DS_CLEAN = _dd("clean")
DS_FIX = _dd("fix")
DS_ARC = _dd("arc")
DS_RELINKED = _dd("relinked")
DS_RELINKED_YR = _dd("relinked_yr")
DS_RELINKED_YR_V1 = _dd("relinked_yr_v1")  # trained-models baseline + daime base
DS_RELINKED_YR_V2 = _dd("relinked_yr_v2")  # v1 + A-only dedup (4885 entities)

# --- 代目 (generation) disambiguation eval benchmarks (built by build_dataset.py) ---
# These store a WIDE w300 context; the old w100/w200/w300 ablation is now one
# dataset + a load-time param: Dataset.load(name, window=100) truncates down
# (re-cut a dataset to a width with `build_dataset.py recut <name> --win W`).
DS_MULTI_DAIME = _dd("multi_daime")  # md173 (gold-certain subset), w300
DS_DAIME_EVAL = _dd("daime_eval")  # official frozen 1479 (option), w300
DS_DAIME_EVAL_FULL = os.path.join(_DDIR, "dataset_daime_eval_full")  # 2214 (default, moved), w300
DS_DAIME_FT = _dd("daime_ft")  # trainable daime split (leaderboard), w300
DS_DAIME_FT_W100 = _dd("daime_ft_w100")  # legacy w100 = daime_ft.window(100)

# --- public historical-text NEL datasets (downloaded + converted by build_dataset.py) ---
# HIPE-2022 bundles 5 originally-distinct corpora; we convert each to unified NEL
# JSONL and place them PARALLEL under data/dataset/ (only hipe2020 is HIPE-native;
# newseye/ajmc/sonar/topres19th come from their own projects).
DS_HIPE_RAW = os.path.join(_DDIR, "HIPE-2022-data")  # cloned HIPE-2022 repo (source)
DS_HIPE2020 = os.path.join(_DDIR, "hipe2020")  # CLEF-HIPE-2020 (de/fr/en)
DS_NEWSEYE = os.path.join(_DDIR, "newseye")  # NewsEye (de/fi/fr/sv)
DS_AJMC = os.path.join(_DDIR, "ajmc")  # Ajax MultiCommentary (de/en/fr)
DS_SONAR = os.path.join(_DDIR, "sonar")  # SoNAR-IDH (de)
DS_TOPRES19TH = os.path.join(_DDIR, "topres19th")  # TopRes19th (en)
# name -> converted-NEL dir, for build_dataset.py hipe conversion + iteration.
HIPE_NEL_DATASETS = {
    "hipe2020": DS_HIPE2020,
    "newseye": DS_NEWSEYE,
    "ajmc": DS_AJMC,
    "sonar": DS_SONAR,
    "topres19th": DS_TOPRES19TH,
}

# name -> path, for programmatic iteration (src/build_dataset.py list/registry).
DATASETS = {
    "raw_csv": DS_RAW_CSV,
    "ja": DS_JA,
    "yakusya_nel": DS_YAKUSYA_NEL,
    "clean": DS_CLEAN,
    "fix": DS_FIX,
    "arc": DS_ARC,
    "relinked": DS_RELINKED,
    "relinked_yr": DS_RELINKED_YR,
    "relinked_yr_v1": DS_RELINKED_YR_V1,
    "relinked_yr_v2": DS_RELINKED_YR_V2,
    "multi_daime": DS_MULTI_DAIME,
    "daime_eval": DS_DAIME_EVAL,
    "daime_eval_full": DS_DAIME_EVAL_FULL,
    "daime_ft": DS_DAIME_FT,
    "daime_ft_w100": DS_DAIME_FT_W100,
    "hipe_raw": DS_HIPE_RAW,
    "hipe2020": DS_HIPE2020,
    "newseye": DS_NEWSEYE,
    "ajmc": DS_AJMC,
    "sonar": DS_SONAR,
    "topres19th": DS_TOPRES19TH,
}

# Variants whose data dir differs from the data/dataset_<v> convention (moved).
DATASET_PATH_OVERRIDES = {
    "daime_eval_full": DS_DAIME_EVAL_FULL,
}


def dataset_data_dir(ds):
    """Resolve a dataset variant's data dir (override table first, else the
    data/dataset_<v> convention)."""
    return DATASET_PATH_OVERRIDES.get(ds, os.path.join(PROJ_DATA, f"dataset_{ds}"))


if is_variant(_DATASET):
    ORIG_DATA = dataset_data_dir(_DATASET)
    EXPERIMENTS = os.path.join(PROJ_EXP, f"experiments_{_DATASET}")
    PRED_DIR = os.path.join(PROJ_EXP, f"predictions_{_DATASET}")
    LOG_DIR = os.path.join(PROJ_LOG, f"logs_{_DATASET}")
if os.environ.get("YAKUSYA_DATA_ROOT"):
    ORIG_DATA = os.environ["YAKUSYA_DATA_ROOT"]
if os.environ.get("YAKUSYA_EXPERIMENTS_ROOT"):
    EXPERIMENTS = os.environ["YAKUSYA_EXPERIMENTS_ROOT"]


def scheme_paths(scheme):
    """Single source of every per-scheme path (data/models/predictions/logs/...).
    original points at ORIG_DATA + the repo-root predictions/; a dataset variant
    (YAKUSYA_DATASET) routes models/refined/dyvo to the *_<variant> subdirs."""
    ds = os.environ.get("YAKUSYA_DATASET", _DEFAULT_DATASET)
    if scheme == "original":
        if is_variant(ds):
            subdir = f"models_{ds}"
            return {
                "scheme": scheme,
                "data": ORIG_DATA,
                "test": os.path.join(ORIG_DATA, "test.jsonl"),
                "train": os.path.join(ORIG_DATA, "train.jsonl"),
                "valid": os.path.join(ORIG_DATA, "valid.jsonl"),
                "entities": os.path.join(ORIG_DATA, "entities.jsonl"),
                "models": os.path.join(PROJ_EXP, f"{subdir}/original"),
                "trie": os.path.join(PROJ_EXP, f"{subdir}/original/yakusya_trie.pkl"),
                "predictions": PRED_DIR,
                "logs": LOG_DIR,
                "chatel": os.path.join(EXPERIMENTS, "original_onenet_chatel.json"),
                "refined_data": os.path.join(PROJ_EXP, f"{subdir}/original/refined_data"),
                "dyvo_outputs": os.path.join(PROJ_EXP, f"{subdir}/original/dyvo_outputs"),
                "dyvo_data": os.path.join(PROJ_EXP, f"{subdir}/original/dyvo_data"),
            }
        return {
            "scheme": scheme,
            "data": ORIG_DATA,
            "test": os.path.join(ORIG_DATA, "test.jsonl"),
            "train": os.path.join(ORIG_DATA, "train.jsonl"),
            "valid": os.path.join(ORIG_DATA, "valid.jsonl"),
            "entities": os.path.join(ORIG_DATA, "entities.jsonl"),
            "models": os.path.join(PROJ_EXP, "models_ja"),
            "trie": os.path.join(GENRE_DIR, "data/yakusya_trie.pkl"),
            "predictions": PRED_DIR,
            "logs": LOG_DIR,
            "chatel": f"{WS}/In_Context_EL/data/yakusya/yakusya_test.json",
            "refined_data": os.path.join(REFINED_DIR, "yakusya_data"),
            "dyvo_outputs": os.path.join(DYVO_DIR, "outputs/yakusya/04-28-2026"),
            "dyvo_data": os.path.join(DYVO_DIR, "yakusya_data"),
        }
    base = os.path.join(EXPERIMENTS, scheme)
    return {
        "scheme": scheme,
        "data": os.path.join(base, "data"),
        "test": os.path.join(base, "data/test.jsonl"),
        "train": os.path.join(base, "data/train.jsonl"),
        "valid": os.path.join(base, "data/valid.jsonl"),
        "entities": os.path.join(base, "data/entities.jsonl"),
        "models": os.path.join(base, "models"),
        "trie": os.path.join(base, "models/yakusya_trie.pkl"),
        "predictions": os.path.join(base, "predictions"),
        "logs": os.path.join(base, "logs"),
        "chatel": os.path.join(base, "onenet_chatel.json"),
        "refined_data": os.path.join(base, "refined_data"),
        "dyvo_outputs": os.path.join(base, "dyvo_outputs"),
        "dyvo_data": os.path.join(base, "dyvo_data"),
    }


# ---- per-model run config (default hyperparameters; edit here) ----
# Keys map to each model's NELModel subclass `name`. venv/script paths live on the
# classes (derived from the constants above); these are the tunable run defaults.
MODELS = {
    # chatel LLM rerankers
    "In_Context_EL": {"model": "gpt-4o-mini", "num_workers": 20},
    "DeepEL": {"model": "gpt-4o-mini", "num_workers": 20},
    "SumMC": {"model": "gpt-4o-mini", "num_workers": 20},
    # subprocess predictors
    "onenet": {
        "model": f"{WS}/OneNet/models/Llama-3.1-70B-Instruct",
        "num_cands": 10,
        "gpu": "0,1",
    },
    "llmd": {"model": "gpt-4o-mini", "num_workers": 20},
    "mad": {"model": "gpt-4o-mini", "rounds": 2, "k": 5, "max_steps": 8, "num_workers": 20},
    "transmad": {"model": "gpt-4o-mini", "rounds": 2, "k": 5, "max_steps": 8, "num_workers": 20},
    "reflexion": {
        "model": "gpt-4o-mini",
        "max_trials": 3,
        "k": 5,
        "max_steps": 8,
        "num_workers": 20,
    },
    "maps": {"model": "gpt-4o-mini", "k": 5, "max_steps": 8, "num_workers": 20},
    "rerank": {
        "model": "gpt-4o-mini",
        "n_samples": 3,
        "temperature": 0.7,
        "k": 5,
        "max_steps": 8,
        "num_workers": 20,
    },
    "zscot": {"model": "gpt-4o-mini", "k": 5, "max_steps": 8, "num_workers": 20},
    "selfcon": {
        "model": "gpt-4o-mini",
        "n_samples": 5,
        "temperature": 0.7,
        "k": 5,
        "max_steps": 8,
        "num_workers": 20,
    },
    "plansolve": {"model": "gpt-4o-mini", "k": 5, "max_steps": 8, "num_workers": 20},
    "cove": {"model": "gpt-4o-mini", "k": 5, "max_steps": 8, "num_workers": 20},
    "llm_tool": {"model": "gpt-4o-mini", "max_steps": 8, "num_workers": 20},
    # trainable / candidate-list models
    "blink": {"train_batch_size": 16, "bert_model": "bert-base-multilingual-cased"},
    "mgenre": {"train_batch_size": 16},
    "refined": {"train_batch_size": 1},
}


def load():
    """Importing this module already loads .env and builds the config; this is a
    no-op hook run_all calls so 'load the config' is explicit at startup."""
    return MODELS


# ============================================================================
# 役者評判記 corpus tables
#
# Moved here from the former clients/convert_config.py: these are lookup tables,
# not a client — they reach nothing outside the process. The rest of that file
# (a data-source priority registry and an EntitySourceQuery facade over the
# Ritsumeikan/jinmei lookups) had no callers and was deleted with it.
# ============================================================================

# Map from CSV file name (without the .csv suffix) to its year (年) information
BOOK_YEAR_MAPPING = {
    "三国舞台鏡(上)": {"year": 1786, "year_str": "天明六年 (推断)"},
    "三国舞台鏡(下)": {"year": 1786, "year_str": "天明六年"},
    "三芝居役者評判千舟湊(上)": {"year": 1777, "year_str": "安永六年 (文中)"},
    "三芝居役者評判千舟湊(下)": {"year": 0, "year_str": ""},
    "三芝居役者評判千舟湊(中)": {"year": 0, "year_str": ""},
    "天明四年江戸評判記": {"year": 1784, "year_str": "天明四年 (文中)"},
    "岡見八眼": {"year": 1772, "year_str": "明和九年 (文中)"},
    "役者☆名位(上)": {"year": 1733, "year_str": "享保十八年 (文中)"},
    "役者☆名位(下)": {"year": 1733, "year_str": "享保十八年 (文中)"},
    "役者☆名位(中)": {"year": 1733, "year_str": "享保十八年 (文中)"},
    "役者一陽来(坂)": {"year": 1773, "year_str": "安永二年 (文中)"},
    "役者一陽来(江)": {"year": 1773, "year_str": "安永二年 (巳正月吉日)"},
    "役者三ツ叶(坂)": {"year": 1769, "year_str": "明和六年 (文中)"},
    "役者三ツ叶(江)": {"year": 1785, "year_str": "天明五年 (文中)"},
    "役者三ヶの角文字(京)": {"year": 1781, "year_str": "安永十年"},
    "役者三ヶの角文字(坂)": {"year": 1781, "year_str": "安永十年"},
    "役者三ヶの角文字(江)": {"year": 1781, "year_str": "安永十年"},
    "役者三升華(上)": {"year": 1779, "year_str": "安永八年 (文中)"},
    "役者三升華(下)": {"year": 1779, "year_str": "安永八年"},
    "役者三升華(中)": {"year": 1770, "year_str": "明和七年"},
    "役者三部教(京)": {"year": 1783, "year_str": "天明三年"},
    "役者三部教(坂)": {"year": 1783, "year_str": "天明三年 (推断)"},
    "役者三部教(江)": {"year": 1783, "year_str": "天明三年"},
    "役者三靨面": {"year": 1782, "year_str": "天明二年 (文中)"},
    "役者世鳳凰(京)": {"year": 1774, "year_str": "安永三年"},
    "役者世鳳凰(坂)": {"year": 1777, "year_str": "安永六年"},
    "役者世鳳凰(江)": {"year": 1724, "year_str": "享保九年"},
    "役者互先(京)": {"year": 1779, "year_str": "安永八年"},
    "役者互先(坂)": {"year": 1779, "year_str": "安永八年"},
    "役者互先(江)": {"year": 1779, "year_str": "安永八年"},
    "役者位上下(京)": {"year": 1774, "year_str": "安永三年"},
    "役者位下上(坂)": {"year": 1767, "year_str": "明和四年"},
    "役者位下上(江)": {"year": 1774, "year_str": "安永三年"},
    "役者位下上(京)": {"year": 1774, "year_str": "安永三年 (文中)"},
    "役者位弥満(上)": {"year": 1688, "year_str": "元禄十七年 (推断)"},
    "役者位弥満(下)": {"year": 1688, "year_str": "元禄十七年"},
    "役者位弥満(中)": {"year": 1688, "year_str": "元禄十七年 (推断)"},
    "役者兎手柄(京)": {"year": 1783, "year_str": "天明三年"},
    "役者兎手柄(坂)": {"year": 1783, "year_str": "天明三年"},
    "役者初艶皃(上)": {"year": 1785, "year_str": "天明五年 (推断)"},
    "役者初艶皃(下)": {"year": 1785, "year_str": "天明五年"},
    "役者初艶皃(中)": {"year": 1785, "year_str": "天明五年 (推断)"},
    "役者千両箱(京)": {"year": 1784, "year_str": "天明四年"},
    "役者千両箱(坂)": {"year": 1784, "year_str": "天明四年 (推断)"},
    "役者千両箱(江)": {"year": 1784, "year_str": "天明四年"},
    "役者名物集": {"year": 1711, "year_str": "正徳元年 (文中)"},
    "役者名物集(全)": {"year": 1711, "year_str": "正徳元年 (文中)"},
    "役者商売往来(上)": {"year": 1779, "year_str": "安永八年 (推断)"},
    "役者商売往来(下)": {"year": 1779, "year_str": "安永八年"},
    "役者商売往来(中)": {"year": 1779, "year_str": "安永八年 (推断)"},
    "役者土地万両": {"year": 1784, "year_str": "天明四年"},
    "役者夕円居": {"year": 1783, "year_str": "天明三年"},
    "役者大極図(京)": {"year": 1786, "year_str": "天明六年"},
    "役者大極図(坂)": {"year": 1786, "year_str": "天明六年"},
    "役者大極図(江)": {"year": 1786, "year_str": "天明六年"},
    "役者大矢数(京)": {"year": 1778, "year_str": "安永七年"},
    "役者大矢数(坂)": {"year": 1778, "year_str": "安永七年"},
    "役者大矢数(江)": {"year": 1778, "year_str": "安永七年"},
    "役者大通鑑(京)": {"year": 1776, "year_str": "安永五年"},
    "役者大通鑑(坂)": {"year": 1776, "year_str": "安永五年"},
    "役者大通鑑(江)": {"year": 1776, "year_str": "安永五年"},
    "役者帰り花(上)": {"year": 1778, "year_str": "安永七年 (文中)"},
    "役者帰り花(下)": {"year": 0, "year_str": ""},
    "役者帰り花(中)": {"year": 0, "year_str": ""},
    "役者年始状(京)": {"year": 1785, "year_str": "天明五年"},
    "役者年始状(坂)": {"year": 1785, "year_str": "天明五年"},
    "役者年始状(江)": {"year": 1784, "year_str": "天明四年"},
    "役者晴小袖(京)": {"year": 1780, "year_str": "安永九年"},
    "役者晴小袖(坂)": {"year": 1780, "year_str": "安永九年"},
    "役者晴小袖(江)": {"year": 1780, "year_str": "安永九年"},
    "役者有難(京)": {"year": 1774, "year_str": "安永三年"},
    "役者有難(坂)": {"year": 1774, "year_str": "安永三年"},
    "役者有難(江)": {"year": 1774, "year_str": "安永三年"},
    "役者清濁(京)": {"year": 1773, "year_str": "安永二年 (推断)"},
    "役者清濁(坂)": {"year": 1773, "year_str": "安永二年 (推断)"},
    "役者清濁(江)": {"year": 1773, "year_str": "安永二年"},
    "役者清濁（京）": {"year": 1773, "year_str": "安永二年 (文中)"},
    "役者男紫花(京)": {"year": 1779, "year_str": "安永八年"},
    "役者男紫花(坂)": {"year": 1779, "year_str": "安永八年"},
    "役者男紫花(江)": {"year": 1779, "year_str": "安永八年"},
    "役者男風流(京)": {"year": 1776, "year_str": "安永五年"},
    "役者男風流(坂)": {"year": 1776, "year_str": "安永五年"},
    "役者男風流(江)": {"year": 1776, "year_str": "安永五年"},
    "役者白虎通(京)": {"year": 1782, "year_str": "天明二年 (推断)"},
    "役者白虎通(坂)": {"year": 1782, "year_str": "天明二年 (推断)"},
    "役者白虎通(江)": {"year": 1782, "year_str": "天明二年"},
    "役者百囀(上)": {"year": 1785, "year_str": "天明五年 (推断)"},
    "役者百囀(下)": {"year": 1785, "year_str": "天明五年"},
    "役者百囀(中)": {"year": 1785, "year_str": "天明五年 (推断)"},
    "役者百薬長": {"year": 1778, "year_str": "安永七年"},
    "役者真功記(京)": {"year": 1786, "year_str": "天明六年"},
    "役者真功記(坂)": {"year": 1786, "year_str": "天明六年"},
    "役者真功記(江)": {"year": 1786, "year_str": "天明六年 (推断)"},
    "役者瞽千人(上)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者瞽千人(下)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者競馬香(上)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者競馬香(下)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者競馬香(中)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者紫郎鼠(京)": {"year": 1780, "year_str": "安永九年"},
    "役者紫郎鼠(坂)": {"year": 1780, "year_str": "安永九年 (推断)"},
    "役者紫郎鼠(江)": {"year": 1780, "year_str": "安永九年"},
    "役者花の会(京)": {"year": 1777, "year_str": "安永六年"},
    "役者花の会(坂)": {"year": 1777, "year_str": "安永六年 (推断)"},
    "役者花の会(江)": {"year": 1777, "year_str": "安永六年"},
    "役者花実論(京)": {"year": 1782, "year_str": "天明二年 (文中)"},
    "役者花実論(坂)": {"year": 1782, "year_str": "天明二年 (文中)"},
    "役者花実論(江)": {"year": 1747, "year_str": "延享四年 (文中)"},
    "役者花見車": {"year": 1779, "year_str": "安永八年 (文中)"},
    "役者花酒盛(上)": {"year": 1716, "year_str": "享保十二年"},
    "役者花酒盛(下)": {"year": 1716, "year_str": "享保十二年 (推断)"},
    "役者花酒盛(中)": {"year": 1716, "year_str": "享保十二年 (推断)"},
    "役者芸雛形(京)": {"year": 1775, "year_str": "安永四年"},
    "役者芸雛形(坂)": {"year": 1772, "year_str": "明和九年"},
    "役者芸雛形(江)": {"year": 1775, "year_str": "安永四年"},
    "役者蓼喰虫(京)": {"year": 1784, "year_str": "天明四年"},
    "役者蓼喰虫(坂)": {"year": 1780, "year_str": "安永九年"},
    "役者評判万年醯(上)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者評判万年醯(下)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者評判万年醯(中)": {"year": 1776, "year_str": "安永五年 (文中)"},
    "役者誉舞台(京)": {"year": 1784, "year_str": "天明四年 (推断)"},
    "役者誉舞台(坂)": {"year": 1784, "year_str": "天明四年"},
    "役者誉舞台(江)": {"year": 1784, "year_str": "天明四年"},
    "役者贔屓升(上)": {"year": 1780, "year_str": "安永九年 (推断)"},
    "役者贔屓升(下)": {"year": 1780, "year_str": "安永九年"},
    "役者贔屓升(中)": {"year": 1780, "year_str": "安永九年 (推断)"},
    "役者通利句(上)": {"year": 1776, "year_str": "安永五年 (推断)"},
    "役者通利句(下)": {"year": 1776, "year_str": "安永五年"},
    "役者通利句(中)": {"year": 1776, "year_str": "安永五年 (推断)"},
    "役者酸辛甘(京)": {"year": 1775, "year_str": "安永四年"},
    "役者酸辛甘(坂)": {"year": 1775, "year_str": "安永四年"},
    "役者酸辛甘(江)": {"year": 1775, "year_str": "安永四年"},
    "役者金色(京)": {"year": 1778, "year_str": "安永七年"},
    "役者金色(坂)": {"year": 1778, "year_str": "安永七年"},
    "役者金色(江)": {"year": 1778, "year_str": "安永七年"},
    "役者雛祭り(上)": {"year": 1719, "year_str": "享保四年"},
    "役者雛祭り(下)": {"year": 1719, "year_str": "享保四年 (推断)"},
    "役者雛祭り(中)": {"year": 1719, "year_str": "享保四年 (推断)"},
    "役者風俗通(上)": {"year": 1688, "year_str": "元禄元年"},
    "役者風俗通(下)": {"year": 1688, "year_str": "元禄元年 (推断)"},
    "役者風俗通(中)": {"year": 1688, "year_str": "元禄元年 (推断)"},
    "役者首尾吟": {"year": 1781, "year_str": "天明元年 (文中)"},
    "役者馴染衣": {"year": 1780, "year_str": "安永九年"},
    "ｳ役者江戸桜(上)": {"year": 1781, "year_str": "天明元年 (文中)"},
    "ｳ役者江戸桜(下)": {"year": 0, "year_str": ""},
    "ｳ役者江戸桜(中)": {"year": 1758, "year_str": "宝暦八年 (文中)"},
    "ﾊﾙ役者茶臼芸(上)": {"year": 1781, "year_str": "天明元年 (推断)"},
    "ﾊﾙ役者茶臼芸(下)": {"year": 1781, "year_str": "天明元年"},
    "ﾊﾙ役者茶臼芸(中)": {"year": 1781, "year_str": "天明元年 (推断)"},
}


def get_book_year_info(book_name):
    """
    Get a book's year (年) information.

    Args:
        book_name: book (書) name (without the .csv suffix)

    Returns:
        dict: {'year': int, 'year_str': str} or None
    """
    return BOOK_YEAR_MAPPING.get(book_name)


SUPPORTED_ENTITY_CATEGORIES = [
    "役者",  # actor
    "興行関係者",  # production/performance staff
    "俳名",  # haiku pen name (俳号)
    "演目名",  # play title
    "人名",  # personal name
    "書名",  # book (書) title
    "狂言作者",  # playwright
    "役名",  # role name
    "屋号",  # house name (屋号 / yago)
    "音曲",  # music
    "事項",  # matter/item
]


# ---- method on/off switches (single source of truth for run_all) ----
# run_all's register_methods() consults this: a method whose switch is False is not
# registered as a subcommand (it neither shows in --help nor runs), while its code
# stays in place. Flip a value to control what executes — no code change needed.
# Names not listed here default to enabled.
METHOD_ENABLED = {
    # LLM rerankers
    "In_Context_EL": True,
    "DeepEL": True,
    "SumMC": True,
    "onenet": True,
    # LLM agentic (paper)
    "mad": True,
    "reflexion": True,
    "zscot": True,
    "plansolve": True,
    "selfcon": True,
    "cove": True,
    "maps": True,
    "rerank": True,
    # trained linkers
    "blink": True,
    "mgenre": True,
    "refined": True,
    "dyvo": True,
    # off by default — implemented but not in the paper's comparison, kept in the code
    "llmd": False,  # third-party biomedical disambiguator (Ye 2025), cited only
    "llmael": False,  # BLINK + LLMAEL augmentation, cited only
    "t5ja_ft": False,  # T5-Japanese seq2seq EL, not in the paper
    "transmad": False,  # Translation-augmented MAD (ties/loses to MAD)
    "llm_tool": False,  # dropped as a compared system; module stays as shared lib
}


def method_enabled(name):
    """Whether method ``name`` is enabled (default True if it is not listed)."""
    return METHOD_ENABLED.get(name, True)
