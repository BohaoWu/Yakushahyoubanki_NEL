#!/usr/bin/env python3
"""
Unified CLI driver for Yakusya entity linking: 5 splits × all models, covering
split generation → training → candidate generation → prediction → evaluation →
summary.

Architecture (post-refactor; config and method classes moved out, this file is
entry point + orchestration + in-process prediction/tools):
  config.py               Single config source: workspace root + .env, each
                          method's repo/venv/script paths, data/experiment layout,
                          dataset-variant routing, scheme_paths(), per-model default hyperparams (MODELS)
  src/nel_models.py       NELModel framework + NEL_dataset (the 4 split methods). Framework methods:
                            - chatel rerankers: In_Context_EL / DeepEL / SumMC
                            - subprocess prediction: onenet / llmd / mad / llm_tool
                            - trainable (train_<name>): blink / mgenre / refined
                            - candidate-constrained (<name>_blink10): mgenre / refined
                            - candidate provider: Blink.provide_candidates (gen_blink_cands)
  src/run_all.py (this file)  argparse entry point + registration (register_methods/_trainers/
                          _constrained_methods), eval, make_splits/gen_trie,
                          all/matrix orchestrators (config and methods moved out)

Every subcommand accepts --scheme {original,pair_disjoint,entity_disjoint,temporal,book_disjoint}.
Per-scheme paths come from config.scheme_paths(); variants switch via the YAKUSYA_DATASET env var (see config).

Subcommand overview (in pipeline order; ★ = registered by nel_models' NELModel classes):

  data / candidates:
    make_splits     generate the 4 splits + link entities (split logic in NEL_dataset)
    gen_trie        per-split mGENRE trie (entities.jsonl → trie.pkl)
    gen_blink_cands per-split BLINK top-10 candidates (chatel JSON; = Blink candidate source)

  training (all ★, registered by nel_models' *NELModel classes; single subprocess via train_cmd, multi-step via train() override):
    train_blink / train_mgenre / train_refined   single subprocess (Blink/MGenre/ReFinED)
    train_t5ja_ft   T5-Japanese seq2seq (train + predict, in-process, T5ja)
    train_dyvo      prep + lsr.train + trec→JSONL (DyVo)
    train_refined_pem  rebuild PEM + train + predict (ReFinED)
    train_blink_llmael LLMAEL-augmented split + train BLINK + predict (Blink)

  prediction (full pool, in-process torch, each launched in its own venv):
    blink ★ mgenre ★ refined ★

  prediction (candidate-constrained / LLM):
    mgenre_blink10 ★ refined_blink10 ★  (BLINK top-10 constraint, subprocess)
    dyvo_blink10 ★                      (BLINK top-10 constraint, in-process adapter/torch)
    onenet ★ llmd ★ mad ★                OneNet-70B / llm_disambiguator / MAD debate
    In_Context_EL ★ DeepEL ★ SumMC ★      chatel LLM rerankers
    llm_tool ★                            tool-calling LLM direct disambiguation

  evaluation / summary / adaptation:
    eval            one prediction JSONL → ID-match + Title-match
    all             full train + predict + eval under one scheme
    matrix          all 5 splits × all models, writes SUMMARY.md

Every prediction emits a unified JSONL: {sample_idx, gold_id, pred_id, pred_title, mention, type}
gold_id/pred_id are in numeric_id space (not line index).

Examples:
  python run_all.py matrix                              # full pipeline, all splits
  python run_all.py all --scheme pair_disjoint          # full pipeline, one scheme
  python run_all.py make_splits                         # generate splits only
  python run_all.py train_mgenre --scheme pair_disjoint --gpu 0
  python run_all.py mgenre --scheme pair_disjoint --model models/mgenre_ft --out preds/mgenre_ft.jsonl
  python run_all.py In_Context_EL --scheme original --chatel cands.json --test t.jsonl --entities e.jsonl
"""

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict

import config  # central run config (loaded once at import: .env, paths, MODELS)
import nel_models  # NELModel framework + NEL_dataset (the 4 split methods live here)

config.load()  # explicit "load the config" at startup


# ===================== Paths / venvs =====================
# All paths / repo dirs / venvs / scripts / data-layout / splits live in config.py
# (single source). The dataset-variant routing (YAKUSYA_DATASET) is resolved there.
from config import (  # noqa: E402
    BLINK_DIR,
    BLINK_PY,
    EXPERIMENTS,
    GENRE_DIR,
    LLMD_PY,
    MGENRE_PY,
    ONENET_PY,
    ORIG_DATA,
    REFINED_DIR,
    REFINED_PY,
    REFINED_TRAIN_PY,
    SCHEMES,
    scheme_paths,
)

random.seed(config.SPLIT_SEED)


def normalize_title(s: str) -> str:
    return (s or "").replace("　", "").replace(" ", "").strip()


# ===================== utils =====================


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_subprocess(cmd, log_path=None, env=None, check=False):
    """Run a subprocess, optionally tee to log_path. Returns return code."""
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    e = os.environ.copy()
    if env:
        e.update(env)
    shell = isinstance(cmd, str)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as logf:
            ret = subprocess.run(cmd, shell=shell, stdout=logf, stderr=subprocess.STDOUT, env=e)
    else:
        ret = subprocess.run(cmd, shell=shell, env=e)
    if check and ret.returncode != 0:
        raise RuntimeError(f"command failed (rc={ret.returncode}): {cmd}")
    return ret.returncode


# ===================== lane scheduler =====================


class LaneScheduler:
    """Run jobs across named lanes. Each lane runs its jobs sequentially in its
    own thread; lanes run in parallel. A job waits for its dependency jobs
    (which may live in OTHER lanes) before it starts.

    Designed for: one lane per GPU + one lane for API/LLM methods, so both GPUs
    stay busy and remote-LLM calls never block GPU work. Cross-lane deps capture
    the real pipeline DAG (e.g. refined_blink10 on gpu0 needs train_refined on
    gpu1 and gen_blink_cands on gpu0).
    """

    def __init__(self, dry_run=False):
        self.lanes = {}  # lane name -> [job dict]
        self.events = {}  # job name -> threading.Event (set when done)
        self.results = {}  # job name -> return code / Exception
        self.dry_run = dry_run
        self._lock = threading.Lock()

    def add(self, name, lane, fn, deps=()):
        """Queue `fn` (a 0-arg callable) on `lane`, after `deps` (job names)."""
        self.lanes.setdefault(lane, []).append({"name": name, "fn": fn, "deps": list(deps)})
        self.events.setdefault(name, threading.Event())
        return name

    def _worker(self, lane):
        for job in self.lanes[lane]:
            for d in job["deps"]:
                ev = self.events.get(d)
                if ev:
                    ev.wait()
            name = job["name"]
            if self.dry_run:
                with self._lock:
                    print(
                        f"  [{lane}] {name}"
                        + (f"   ⟵ {', '.join(job['deps'])}" if job["deps"] else "")
                    )
                self.results[name] = 0
            else:
                with self._lock:
                    print(f"  [{lane}] ▶ start {name} ({time.strftime('%H:%M:%S')})")
                try:
                    self.results[name] = job["fn"]()
                except Exception as e:  # keep other lanes alive
                    self.results[name] = e
                    with self._lock:
                        print(f"  [{lane}] ✗ {name} FAILED: {e}")
                else:
                    with self._lock:
                        print(f"  [{lane}] ✓ done {name} ({time.strftime('%H:%M:%S')})")
            self.events[name].set()

    def run(self):
        """Start all lanes, block until every job finishes. Returns results dict."""
        ts = [threading.Thread(target=self._worker, args=(l,), daemon=True) for l in self.lanes]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return self.results


# ===================== make_splits =====================
# The 4 split methods (pair/entity/temporal/book disjoint) live on
# nel_models.NEL_dataset; run_all delegates to it so the logic exists once.
SPLIT_NAMES = nel_models.NEL_dataset.SPLITS


def cmd_make_splits(args):
    """Generate the 4 alternative splits under experiments/<scheme>/data/."""
    ds = nel_models.NEL_dataset(ORIG_DATA)
    schemes = args.schemes or list(SPLIT_NAMES)
    print(f"[make_splits] total={len(ds)} schemes={schemes}")
    for name in schemes:
        if name not in SPLIT_NAMES:
            print(f"  [skip] unknown scheme {name}")
            continue
        tr, va, te = ds.split(name)
        out_dir = os.path.join(EXPERIMENTS, name, "data")
        write_jsonl(tr, os.path.join(out_dir, "train.jsonl"))
        write_jsonl(va, os.path.join(out_dir, "valid.jsonl"))
        write_jsonl(te, os.path.join(out_dir, "test.jsonl"))
        ent_link = os.path.join(out_dir, "entities.jsonl")
        if os.path.lexists(ent_link):
            os.remove(ent_link)
        os.symlink(os.path.join(ORIG_DATA, "entities.jsonl"), ent_link)
        train_pairs = {(s["mention"], s["label_id"]) for s in tr}
        train_ents = {s["label_id"] for s in tr}
        leak_p = sum(1 for s in te if (s["mention"], s["label_id"]) in train_pairs)
        leak_e = sum(1 for s in te if s["label_id"] in train_ents)
        print(
            f"  [{name}] train={len(tr)} valid={len(va)} test={len(te)} "
            f"pair-leak={leak_p}/{len(te)}={leak_p / max(len(te), 1) * 100:.1f}% "
            f"ent-overlap={leak_e}/{len(te)}={leak_e / max(len(te), 1) * 100:.1f}%"
        )


# ===================== gen_trie =====================


def cmd_gen_trie(args):
    """Build mGENRE trie from per-split entities.jsonl."""
    sys.path.insert(0, GENRE_DIR)
    from genre.hf_model import mGENRE
    from genre.trie import Trie

    paths = scheme_paths(args.scheme)
    if os.path.exists(paths["trie"]) and not args.overwrite:
        print(f"[gen_trie] exists, skip: {paths['trie']}")
        return
    print("[gen_trie] loading mGENRE-pre tokenizer (for trie encoding)...")
    model = mGENRE.from_pretrained("facebook/mgenre-wiki")
    ents = load_jsonl(paths["entities"])
    print(f"[gen_trie] encoding {len(ents)} entity titles...")
    entries = []
    for e in ents:
        encoded = model.encode(f"{e['title']} >> ja").tolist()
        entries.append([2] + encoded[1:])
    trie = Trie(entries)
    os.makedirs(os.path.dirname(paths["trie"]), exist_ok=True)
    with open(paths["trie"], "wb") as f:
        pickle.dump(trie, f)
    print(f"[gen_trie] saved → {paths['trie']}")


# ============================================================================
# All NEL method subcommands (rerankers / onenet / llmd / mad / llm_tool /
# blink·mgenre·refined·dyvo·t5ja·llmael train+predict+blink10 / gen_blink_cands)
# are registered in _build_parser() from nel_models.py. Only the
# orchestration/util commands below stay here: eval / make_splits / gen_trie /
# all / matrix. See the module docstring for the full map.
# ============================================================================


# ===================== eval =====================


def _load_multi_cand_set():
    """Return set of unique sample keys whose nel_dataset record had >1
    candidates. Key = (mention, label_id, last 100 chars of left context),
    which uniquely identifies mention occurrences across all splits.
    nel_dataset uses 'left_context' field; split files use 'context_left' —
    same content so we just take the last 100 chars."""
    src = os.path.join(ORIG_DATA, "nel_dataset.jsonl")
    if not os.path.exists(src):
        return None
    multi = set()
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("candidates", [])) <= 1:
                continue
            key = (r.get("mention", ""), r.get("ans_id"), (r.get("left_context", "") or "")[-100:])
            multi.add(key)
    return multi


def cmd_eval(args):
    """ID-match + Title-match (per type) for one prediction JSONL.

    --multi_candidate_only filters to samples where >1 candidate existed
    at dataset construction (i.e., genuine disambiguation cases). Single-
    candidate samples (~65% of dataset) are excluded as they are trivial.
    """
    paths = scheme_paths(args.scheme) if args.scheme else None
    test_path = args.test or (paths["test"] if paths else None)
    ent_path = args.entities or (paths["entities"] if paths else None)
    test = load_jsonl(test_path)
    ents = load_jsonl(ent_path)
    id2title_norm = {int(e["numeric_id"]): normalize_title(e["title"]) for e in ents}
    title2ids = defaultdict(set)
    for e in ents:
        title2ids[normalize_title(e["title"])].add(int(e["numeric_id"]))

    # Canonicalize duplicate person-records so ID-match scores people, not record
    # ids (see entity_canon). canon[id] -> representative id of the merge group.
    from entity_canon import build_canon_map

    canon, canon_stats = build_canon_map(ents, mode=args.canon)

    def cid(i):
        return canon.get(i, i)

    # Filter test to multi-candidate samples if requested
    if args.multi_candidate_only:
        multi_set = _load_multi_cand_set()
        if multi_set is None:
            print(
                "[eval] WARNING: --multi_candidate_only failed (nel_dataset.jsonl missing); using all samples"
            )
        else:

            def _key(r):
                return (
                    r.get("mention", ""),
                    r.get("label_id"),
                    (r.get("context_left", "") or "")[-100:],
                )

            keep_idx = {i for i, r in enumerate(test) if _key(r) in multi_set}
            test = [(i, r) for i, r in enumerate(test) if i in keep_idx]

    preds = {}
    with open(args.preds) as f:
        for line in f:
            r = json.loads(line)
            preds[r["sample_idx"]] = r

    # Normalize iteration so we have orig_idx in both filter modes
    if args.multi_candidate_only and isinstance(test, list) and test and isinstance(test[0], tuple):
        items = test
    else:
        items = list(enumerate(test))

    n = len(items)
    id_correct = 0
    title_correct = 0
    by_type_id = defaultdict(lambda: [0, 0])
    by_type_title = defaultdict(lambda: [0, 0])
    missing = 0
    for orig_idx, sample in items:
        if orig_idx not in preds:
            missing += 1
            continue
        p = preds[orig_idx]
        gold_id = sample.get("label_id", p.get("gold_id"))
        gold_title_norm = id2title_norm.get(gold_id, "")
        etype = sample.get("type", "?")
        by_type_id[etype][1] += 1
        by_type_title[etype][1] += 1
        pred_id = p.get("pred_id", -1)
        if pred_id == -1 and "pred_title" in p:
            cand = title2ids.get(normalize_title(p["pred_title"]), set())
            if cand:
                pred_id = next(iter(cand))
        if cid(pred_id) == cid(gold_id):
            id_correct += 1
            by_type_id[etype][0] += 1
        pred_title_norm = (
            id2title_norm.get(pred_id, "")
            if pred_id != -1
            else normalize_title(p.get("pred_title", ""))
        )
        if pred_title_norm and pred_title_norm == gold_title_norm:
            title_correct += 1
            by_type_title[etype][0] += 1

    name = args.name or os.path.basename(args.preds).replace(".jsonl", "")
    suffix = " (multi-cand only)" if args.multi_candidate_only else ""
    print(f"\n=== {name}{suffix} ===")
    print(f"Test samples evaluated: {n - missing}/{n} (missing preds: {missing})")
    if args.canon != "none":
        print(
            f"Canon       : {args.canon} (merged {canon_stats['merged']} "
            f"duplicate records -> {canon_stats['groups']} entities)"
        )
    if n > 0:
        print(f"ID-match    : {id_correct}/{n} = {id_correct / n * 100:.2f}%")
        print(f"Title-match : {title_correct}/{n} = {title_correct / n * 100:.2f}%")
    if args.show_by_type:
        print("\nBy type (n / id-match / title-match):")
        for t in sorted(by_type_id, key=lambda x: -by_type_id[x][1]):
            ic, tot = by_type_id[t]
            tc, _ = by_type_title[t]
            if tot > 0:
                print(
                    f"  {t:10s} n={tot:4d}  id={ic:4d}/{tot}={ic / tot * 100:5.1f}%  "
                    f"title={tc:4d}/{tot}={tc / tot * 100:5.1f}%"
                )
    return (id_correct / n * 100) if n > 0 else None


# ===================== orchestrators =====================


def cmd_all(args):
    """Train + predict + eval for ONE scheme via the LaneScheduler.

    Three lanes run concurrently, each sequential internally; cross-lane deps
    encode the real DAG so both GPUs stay busy and LLM/API work never blocks GPU:
      gpu0: train_blink -> gen_blink_cands -> blink + mGENRE-family preds + DPO
      gpu1: train_refined + train_dyvo + their preds (starts in parallel with gpu0)
      llm : llmd (+ mad/maps/rerank/zscot/selfcon/plansolve/cove with --with_llm) — API only, no GPU
    onenet (70B, needs both GPUs) runs after the lanes. --dry_run prints the plan.
    """
    scheme = args.scheme
    print(f"\n{'=' * 60}\n  SCHEME: {scheme}\n{'=' * 60}")
    paths = scheme_paths(scheme)
    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs(paths["predictions"], exist_ok=True)
    os.makedirs(paths["logs"], exist_ok=True)
    self = os.path.abspath(__file__)
    G0, G1 = args.gpu0, args.gpu1
    blink_out = os.path.join(paths["models"], "blink_v2")
    refined_data = paths["refined_data"]
    P = paths["predictions"]

    def run(cmd, log_name, env_extra=None):
        env = {**os.environ, **(env_extra or {})}
        return run_subprocess(cmd, log_path=os.path.join(paths["logs"], log_name + ".log"), env=env)

    def pred_job(name, py, sub, gpu):
        """A full-pool predict thunk that self-skips when its output exists."""
        out = sub[sub.index("--out") + 1]

        def f():
            if args.skip_existing and os.path.exists(out):
                print(f"  [skip exists] {name}")
                return 0
            return run_subprocess(
                [py, self] + sub,
                log_path=os.path.join(paths["logs"], f"predict_{name}.log"),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )

        return f

    def b10_job(name, cmd):
        out = os.path.join(P, f"{name}.jsonl")

        def f():
            if args.skip_existing and os.path.exists(out):
                print(f"  [skip exists] {name}")
                return 0
            return run_subprocess(cmd, log_path=os.path.join(paths["logs"], f"predict_{name}.log"))

        return f

    # gen_trie is fast and feeds train_mgenre; do it up-front (not a lane job)
    if not args.skip_train and not args.dry_run:
        cmd_gen_trie(argparse.Namespace(scheme=scheme, overwrite=False))

    sch = LaneScheduler(dry_run=args.dry_run)

    # ===== gpu0 lane: BLINK + candidates + mGENRE family + DPO =====
    if not args.skip_train:
        _bmc = int(os.environ.get("BLINK_MAX_CTX", "128"))  # context-window ablation knob
        sch.add(
            "train_blink",
            "gpu0",
            lambda: run(
                f"cd {BLINK_DIR} && CUDA_VISIBLE_DEVICES={G0} {BLINK_PY} "
                f"-m blink.biencoder.train_biencoder --data_path {paths['data']} "
                f"--output_path {blink_out} --bert_model bert-base-multilingual-cased "
                f"--max_context_length {_bmc} --max_cand_length 128 --max_seq_length {_bmc + 128} "
                f"--num_train_epochs {args.blink_epochs} --train_batch_size 16 --learning_rate 2e-5 "
                f"--eval_batch_size 8 --type_optimization all_encoder_layers --save_interval 1 "
                f"--print_interval 50 --eval_interval 500 --shuffle True --zeshel True",
                "train_blink_v2",
                {"PYTHONPATH": BLINK_DIR},
            ),
        )
        sch.add(
            "gen_blink_cands",
            "gpu0",
            lambda: run(
                [
                    BLINK_PY,
                    self,
                    "gen_blink_cands",
                    "--scheme",
                    scheme,
                    "--bert_model",
                    "bert-base-multilingual-cased",
                    "--top_k",
                    "10",
                    *(["--overwrite"] if not args.skip_existing else []),
                ],
                "gen_blink_cands",
                {"CUDA_VISIBLE_DEVICES": str(G0)},
            ),
            deps=["train_blink"],
        )
        sch.add(
            "train_mgenre",
            "gpu0",
            lambda: run(
                [
                    MGENRE_PY,
                    self,
                    "train_mgenre",
                    "--scheme",
                    scheme,
                    "--gpu",
                    str(G0),
                    "--epochs",
                    str(args.mgenre_epochs),
                    "--batch_size",
                    "16",
                    *(["--skip_existing"] if args.skip_existing else []),
                ],
                "train_mgenre",
            ),
        )
    sch.add(
        "blink_v2",
        "gpu0",
        pred_job(
            "blink_v2",
            BLINK_PY,
            [
                "blink",
                "--scheme",
                scheme,
                "--model_path",
                blink_out,
                "--bert_model",
                "bert-base-multilingual-cased",
                "--data_path",
                paths["data"],
                "--out",
                os.path.join(P, "blink_v2.jsonl"),
                "--max_context_length",
                os.environ.get("BLINK_MAX_CTX", "128"),
            ],
            G0,
        ),
        deps=["train_blink"],
    )
    sch.add(
        "mgenre_ft",
        "gpu0",
        pred_job(
            "mgenre_ft",
            MGENRE_PY,
            [
                "mgenre",
                "--scheme",
                scheme,
                "--model",
                os.path.join(paths["models"], "mgenre_ft"),
                "--out",
                os.path.join(P, "mgenre_ft.jsonl"),
            ],
            G0,
        ),
        deps=["train_mgenre"],
    )
    sch.add(
        "mgenre_ft_blink10",
        "gpu0",
        b10_job(
            "mgenre_ft_blink10",
            [
                MGENRE_PY,
                self,
                "mgenre_blink10",
                "--scheme",
                scheme,
                "--kind",
                "ft",
                "--gpu",
                str(G0),
            ],
        ),
        deps=["train_mgenre", "gen_blink_cands"],
    )

    # ===== gpu1 lane: ReFinED + DyVo (+ their preds); starts parallel to gpu0 =====
    if not args.skip_train:
        sch.add(
            "train_refined",
            "gpu1",
            lambda: run(
                f"cd {REFINED_DIR} && CUDA_VISIBLE_DEVICES={G1} {REFINED_PY} {REFINED_TRAIN_PY} "
                f"--epochs {args.refined_epochs} --batch-size 1 --yakusya_dir {paths['data']} "
                f"--output_dir {refined_data}",
                "train_refined",
            ),
        )
        sch.add(
            "train_dyvo",
            "gpu1",
            lambda: run(
                [
                    BLINK_PY,
                    self,
                    "train_dyvo",
                    "--scheme",
                    scheme,
                    "--gpu",
                    str(G1),
                    "--max_steps",
                    str(args.dyvo_steps),
                    *(["--skip_existing"] if args.skip_existing else []),
                ],
                "train_dyvo",
            ),
            deps=["gen_blink_cands"],
        )
    sch.add(
        "refined",
        "gpu1",
        pred_job(
            "refined",
            REFINED_PY,
            [
                "refined",
                "--scheme",
                scheme,
                "--model_pt",
                os.path.join(refined_data, "best_model/model.pt"),
                "--out",
                os.path.join(P, "refined.jsonl"),
            ],
            G1,
        ),
        deps=["train_refined"],
    )
    sch.add(
        "refined_blink10",
        "gpu1",
        b10_job(
            "refined_blink10",
            [BLINK_PY, self, "refined_blink10", "--scheme", scheme, "--gpu", str(G1)],
        ),
        deps=["train_refined", "gen_blink_cands"],
    )
    sch.add(
        "dyvo_blink10",
        "gpu1",
        b10_job("dyvo_blink10", [BLINK_PY, self, "dyvo_blink10", "--scheme", scheme]),
        deps=["train_dyvo", "gen_blink_cands"],
    )

    # ===== llm lane: API-only methods, fully parallel to the GPU lanes =====
    if (
        not args.skip_subprocess_models
        and os.path.exists(LLMD_PY)
        and os.environ.get("OPENAI_API_KEY")
    ):
        sch.add(
            "llmd",
            "llm",
            lambda: run(
                [
                    BLINK_PY,
                    self,
                    "llmd",
                    "--scheme",
                    scheme,
                    "--model",
                    "gpt-4o-mini",
                    "--num_workers",
                    "20",
                    *(["--skip_existing"] if args.skip_existing else []),
                ],
                "llmd_wrap",
            ),
        )
    if args.with_llm and os.environ.get("OPENAI_API_KEY"):
        for m in ("mad", "maps", "rerank", "zscot", "selfcon", "plansolve", "cove"):
            sch.add(
                m,
                "llm",
                (
                    lambda mm: (
                        lambda: run_subprocess(
                            [sys.executable, self, mm, "--scheme", scheme, "--num_workers", "20"],
                            log_path=os.path.join(paths["logs"], f"{mm}_wrap.log"),
                        )
                    )
                )(m),
            )

    print(
        f"\n[scheduler] lanes: {', '.join(sch.lanes)}  ({'DRY RUN' if args.dry_run else 'running'})"
    )
    sch.run()

    # onenet needs BOTH GPUs -> run after the lanes free them
    if not args.skip_subprocess_models and not args.dry_run and os.path.exists(ONENET_PY):
        run(
            [
                BLINK_PY,
                self,
                "onenet",
                "--scheme",
                scheme,
                "--gpu",
                f"{G0},{G1}",
                "--num_cands",
                "10",
                *(["--skip_existing"] if args.skip_existing else []),
            ],
            "onenet_wrap",
        )

    if args.dry_run:
        print("\n[dry run] no jobs executed.")
        return {}

    # ---------- eval all ----------
    print("\n[eval]")
    eval_results = {}
    for name in os.listdir(paths["predictions"]):
        if not name.endswith(".jsonl"):
            continue
        model_name = name.replace(".jsonl", "")
        eargs = argparse.Namespace(
            preds=os.path.join(paths["predictions"], name),
            name=f"{scheme}/{model_name}",
            scheme=scheme,
            test=None,
            entities=None,
            multi_candidate_only=False,
            show_by_type=False,
            canon="full",
        )
        try:
            score = cmd_eval(eargs)
            eval_results[model_name] = score
        except Exception as e:
            print(f"  [eval fail] {model_name}: {e}")
    print(f"\n[scheme {scheme}] {len(eval_results)} models evaluated")
    return eval_results


def cmd_matrix(args):
    """Run all 5 schemes × all models, write SUMMARY.md."""
    schemes = args.schemes or SCHEMES
    if "original" in schemes:
        # Original split is treated specially (no train_*; reuse pre-trained models)
        pass
    else:
        # Generate splits if needed
        ns = argparse.Namespace(schemes=[s for s in schemes if s != "original"])
        cmd_make_splits(ns)

    all_results = {}
    for scheme in schemes:
        sub_args = argparse.Namespace(
            scheme=scheme,
            gpu0=args.gpu0,
            gpu1=args.gpu1,
            blink_epochs=args.blink_epochs,
            refined_epochs=args.refined_epochs,
            mgenre_epochs=args.mgenre_epochs,
            dyvo_steps=args.dyvo_steps,
            skip_existing=args.skip_existing,
            skip_train=(scheme == "original") or args.skip_train,
            skip_subprocess_models=args.skip_subprocess_models,
            with_llm=getattr(args, "with_llm", False),
            dry_run=getattr(args, "dry_run", False),
        )
        try:
            all_results[scheme] = cmd_all(sub_args)
        except Exception as e:
            print(f"[matrix] scheme {scheme} FAILED: {e}")

    # Write SUMMARY.md
    out_path = os.path.join(EXPERIMENTS, "SUMMARY.md")
    lines = ["# Yakusya entity linking — Split × Model matrix\n"]
    lines.append("ID-match top-1 (%, numeric_id space)\n")
    all_models = sorted({m for r in all_results.values() for m in r})
    lines.append("| Model | " + " | ".join(schemes) + " |")
    lines.append("|---|" + "|".join(["---"] * len(schemes)) + "|")
    for m in all_models:
        row = [f"| {m}"]
        for s in schemes:
            v = all_results.get(s, {}).get(m)
            row.append(f"{v:.1f}" if v is not None else "—")
        lines.append(" | ".join(row) + " |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[matrix] SUMMARY → {out_path}")


# ===================== test (held-out only, cost-tiered) =====================

# gpt-4o-mini unit price (USD/token, 2025-06) and a rough per-call footprint
# (candidates + context + prompt in, one answer out). Used only for the up-front
# cost ESTIMATE printed by `test`; actual spend is whatever the API bills.
_MINI_IN, _MINI_OUT = 0.15 / 1e6, 0.60 / 1e6
_TOK_IN, _TOK_OUT = 2500, 400
_UNIT = _TOK_IN * _MINI_IN + _TOK_OUT * _MINI_OUT  # ~$0.000615 per LLM call

# LLM API calls per item. Multi-turn protocols (debate, reflexion, tool loop)
# issue several model calls per mention, so they cost proportionally more.
_LLM_CALLS = {
    "zscot": 1,
    "plansolve": 1,
    "rerank": 1,
    "In_Context_EL": 1,
    "SumMC": 1,
    "llmd": 1,
    "cove": 2,
    "DeepEL": 2,
    "maps": 3,
    "reflexion": 3,
    "selfcon": 5,
    "mad": 5,
}

# Three cost tiers (the axis that decides whether a method is free to run):
#   local     trained encoders/generators — GPU only, $0 API. Trained on the FULL
#             train split; scored on the held-out test.
#   zeroshot  LLM prompting methods — API cost, ~one pass of N test items each.
#             Running on the 230-item test instead of the full 2214 cuts calls ~90%.
#   hybrid    OneNet (local 70B) — no API, but heavy GPU; gated on compute, not $.
_TEST_TIERS = {
    "local": [
        "blink_v2",
        "mgenre_ft",
        "mgenre_ft_blink10",
        "dyvo",
        "dyvo_blink10",
        "refined",
        "refined_blink10",
        "t5ja_ft",
    ],
    "zeroshot": [
        "zscot",
        "plansolve",
        "selfcon",
        "cove",
        "maps",
        "rerank",
        "mad",
        "reflexion",
        "In_Context_EL",
        "DeepEL",
        "SumMC",
        "llmd",
    ],
    "hybrid": ["onenet"],
}
# method -> subcommand to (re)run it under the current variant (--scheme original is
# an unused placeholder for variant runs; YAKUSYA_DATASET drives the routing).
_TEST_SUBCMD = {"llmd": "llmd", "blink_v2": "blink"}  # rest: subcommand == name
# prediction-file stem aliases (filename on disk differs from the method name).
_TEST_PREDFILE = {"llmd": "llm_disambiguator"}


def _find_pred(preds_dir, name):
    """Existing prediction file for `name` in preds_dir, or None. Tries the bare
    name, the gpt-4o-mini-suffixed name, and any known filename alias."""
    stem = _TEST_PREDFILE.get(name, name)
    for cand in (stem, f"{stem}_gpt-4o-mini"):
        p = os.path.join(preds_dir, f"{cand}.jsonl")
        if os.path.exists(p):
            return p
    return None


def cmd_test(args):
    """Evaluate on the held-out TEST split only (~230 items), grouped into three
    cost tiers, with an up-front API-cost estimate so the paid tiers can be gated.

    Default is estimate-only (prints the plan + cost, runs nothing). Pass
    --run local,zeroshot,hybrid to execute those tiers; --difficulty <method|top>
    runs a single (highest-accuracy) method on the FULL 2214 for the
    generation-count curve, whose tail bins are too sparse at n=230.
    """
    ds = args.dataset
    os.environ["YAKUSYA_DATASET"] = ds  # route every CHILD to the variant
    # config's variant globals are import-time bound, so scheme_paths() would still
    # point at the process's launch-time dataset. Compute the variant paths from ds
    # directly (mirrors config's predictions_<ds>/logs_<ds> convention).
    data_dir = config.dataset_data_dir(ds)
    test_path = os.path.join(data_dir, "test.jsonl")
    ent_path = os.path.join(data_dir, "entities.jsonl")
    preds_dir = os.path.join(config.PROJ_EXP, f"predictions_{ds}")
    logs_dir = os.path.join(config.PROJ_LOG, f"logs_{ds}")
    os.makedirs(logs_dir, exist_ok=True)
    N = sum(1 for _ in open(test_path))
    to_run = set((args.run or "").replace(" ", "").split(",")) - {""}
    self = os.path.abspath(__file__)

    print(f"\n{'=' * 66}\n  TEST mode — dataset={ds}  (held-out test n={N})\n{'=' * 66}")
    grand_calls = grand_cost = 0.0
    plan = {t: [] for t in _TEST_TIERS}
    for tier, methods in _TEST_TIERS.items():
        for m in methods:
            have = _find_pred(preds_dir, m)
            calls = _LLM_CALLS.get(m, 0) * N if tier == "zeroshot" else 0
            cost = calls * _UNIT
            plan[tier].append((m, have is not None, calls, cost))
            grand_calls += calls
            grand_cost += cost

    for tier in ("local", "zeroshot", "hybrid"):
        rows = plan[tier]
        tcost = sum(r[3] for r in rows)
        note = {
            "local": "GPU only, $0 API",
            "zeroshot": "gpt-4o-mini API",
            "hybrid": "local 70B, $0 API (GPU-bound)",
        }[tier]
        print(f"\n[{tier}]  {note}")
        for m, have, calls, cost in rows:
            tag = "have" if have else "MISSING"
            extra = f"  ~{calls} calls  ~${cost:.2f}" if tier == "zeroshot" else ""
            print(f"    {m:<20} {tag:<8}{extra}")
        if tier == "zeroshot":
            miss = sum(c for m, h, c, _ in rows if not h)
            print(f"    -- tier total ~${tcost:.2f}  (missing only: ~${miss * _UNIT:.2f})")
    print(
        f"\n  GRAND API estimate (all zeroshot on n={N}): ~${grand_cost:.2f}  "
        f"[vs full 2214 ≈ ${grand_cost * 2214 / max(N, 1):.2f}]"
    )

    if not to_run and not args.difficulty:
        print("\n  estimate-only (pass --run <tiers> to execute, --difficulty to plot).")
        return

    # ---- execute requested tiers ----
    # Every child is a fresh process, so YAKUSYA_DATASET in its env reroutes config
    # correctly at the child's import (unlike this in-process call).
    child_env = {**os.environ, "YAKUSYA_DATASET": ds}

    def run_method(m, gpu=None):
        if _find_pred(preds_dir, m) and args.skip_existing:
            print(f"  [skip exists] {m}")
            return
        sub = _TEST_SUBCMD.get(m, m)
        cmd = [sys.executable, self, sub, "--scheme", "original"]
        if _TEST_TIERS_TIER(m) == "zeroshot":
            cmd += ["--num_workers", str(args.num_workers)]
        env = dict(child_env)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        run_subprocess(cmd, log_path=os.path.join(logs_dir, f"test_{m}.log"), env=env)

    if "local" in to_run:
        # predict-only pass of the full pipeline (no API, no onenet lane), as a
        # subprocess so the variant routing takes effect.
        print("\n[run local] `all --scheme original --skip_train --skip_subprocess_models`")
        run_subprocess(
            [
                sys.executable,
                self,
                "all",
                "--scheme",
                "original",
                "--skip_train",
                "--skip_subprocess_models",
                *(["--skip_existing"] if args.skip_existing else []),
            ],
            log_path=os.path.join(logs_dir, "test_local.log"),
            env=child_env,
        )
    if "zeroshot" in to_run:
        if not os.environ.get("OPENAI_API_KEY"):
            print("  [zeroshot] no OPENAI_API_KEY; skipping")
        else:
            for m in _TEST_TIERS["zeroshot"]:
                run_method(m)
    if "hybrid" in to_run:
        run_method("onenet", gpu=f"{args.gpu0},{args.gpu1}")

    # ---- difficulty curve: one top method on the FULL 2214 ----
    if args.difficulty:
        _run_difficulty(args, self, logs_dir)

    # ---- final leaderboard on the held-out test ----
    print("\n[eval on held-out test]")
    results = {}
    for name in sorted(os.listdir(preds_dir)):
        if not name.endswith(".jsonl") or name.endswith("_raw.json"):
            continue
        try:
            results[name] = cmd_eval(
                argparse.Namespace(
                    preds=os.path.join(preds_dir, name),
                    name=name.replace(".jsonl", ""),
                    scheme=None,
                    test=test_path,
                    entities=ent_path,
                    multi_candidate_only=False,
                    show_by_type=False,
                    canon="full",
                )
            )
        except Exception as e:
            print(f"  [eval fail] {name}: {e}")
    print()
    for m, s in sorted(results.items(), key=lambda x: -(x[1] or 0)):
        print(f"    {m:<40} {s:.1f}%" if s is not None else f"    {m:<40}  n/a")


def _TEST_TIERS_TIER(m):
    for t, ms in _TEST_TIERS.items():
        if m in ms:
            return t
    return "zeroshot"


def _run_difficulty(args, self, logs_dir):
    """Run the single highest-accuracy method on daime_eval_full (2214) so the
    accuracy-by-generation-count curve has enough items per bin (n=230 has 3-5 in
    the long tail). --difficulty=top picks the cheapest strong single-call protocol."""
    m = "plansolve" if args.difficulty == "top" else args.difficulty
    calls = _LLM_CALLS.get(m, 1) * 2214
    print(
        f"\n[difficulty] running {m} on daime_eval_full (2214) — "
        f"~{calls} calls ~${calls * _UNIT:.2f}"
    )
    env = {**os.environ, "YAKUSYA_DATASET": "daime_eval_full"}
    sub = _TEST_SUBCMD.get(m, m)
    cmd = [
        sys.executable,
        self,
        sub,
        "--scheme",
        "original",
        "--num_workers",
        str(args.num_workers),
    ]
    run_subprocess(cmd, log_path=os.path.join(logs_dir, f"difficulty_{m}.log"), env=env)


# ===================== argparse =====================


def _add_scheme(p, required=False):
    p.add_argument("--scheme", choices=SCHEMES, required=required)


def _add_orchestrator_args(p):
    """Shared GPU/epoch/skip flags for the all/matrix orchestrators."""
    p.add_argument("--gpu0", type=int, default=0)
    p.add_argument("--gpu1", type=int, default=1)
    p.add_argument("--blink_epochs", type=int, default=3)
    p.add_argument("--refined_epochs", type=int, default=3)
    p.add_argument("--mgenre_epochs", type=int, default=3)
    p.add_argument("--dyvo_steps", type=int, default=10000)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument(
        "--skip_train", action="store_true", help="predict + evaluate only, no re-training"
    )
    p.add_argument(
        "--skip_subprocess_models",
        action="store_true",
        help="skip onenet + llmd (these have external dependencies)",
    )
    p.add_argument(
        "--with_llm",
        action="store_true",
        help="also add mad/maps etc. to the llm lane (runs parallel to GPU training)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="only print the lane schedule (tasks and deps per queue), do not execute",
    )


def _build_parser():
    ap = argparse.ArgumentParser(
        description="yakusya entity-linking unified runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- data prep ----
    p = sub.add_parser("make_splits", help="generate 4 alternative splits")
    p.add_argument(
        "--schemes",
        nargs="*",
        help="subset of {pair_disjoint, entity_disjoint, temporal, book_disjoint}",
    )
    p.set_defaults(func=cmd_make_splits)

    p = sub.add_parser("gen_trie", help="build mGENRE trie from entities.jsonl")
    _add_scheme(p, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_gen_trie)

    # ---- all NEL methods come from nel_models (see module docstring) ----
    _nel_ctx = nel_models.Ctx(
        scheme_paths=scheme_paths,
        run_subprocess=run_subprocess,
        add_scheme=_add_scheme,
        experiments=EXPERIMENTS,
        run_all_py=os.path.abspath(__file__),
        write_jsonl=write_jsonl,
    )
    nel_models.register_methods(sub, _nel_ctx)  # rerankers + onenet/llmd/mad/llm_tool
    nel_models.register_trainers(
        sub, _nel_ctx
    )  # train_* (+ refined_pem/dpo/blink_llmael/t5ja_blink10)
    nel_models.register_constrained_methods(sub, _nel_ctx)  # mgenre/refined/dyvo _blink10
    nel_models.register_inproc_predictors(
        sub, _nel_ctx
    )  # blink/mgenre/refined predict + gen_blink_cands

    # ---- eval ----
    p = sub.add_parser("eval", help="ID-match + Title-match for one prediction JSONL")
    _add_scheme(p)
    p.add_argument("--preds", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--test")
    p.add_argument("--entities")
    p.add_argument(
        "--multi_candidate_only",
        action="store_true",
        help="only evaluate samples that had >1 candidate at "
        "dataset construction (real disambiguation cases)",
    )
    p.add_argument(
        "--show_by_type",
        action="store_true",
        help="print per-entity-type breakdown (default: hide)",
    )
    p.add_argument(
        "--canon",
        choices=["none", "source", "full"],
        default="full",
        help="entity canonicalization for ID-match: none=strict "
        "numeric_id; source=merge same-source duplicates (A); "
        "full=A + same-person name-history records (B). "
        "default: full",
    )
    p.set_defaults(func=cmd_eval)

    # ---- orchestrators ----
    p = sub.add_parser("all", help="full pipeline for ONE scheme")
    _add_scheme(p, required=True)
    _add_orchestrator_args(p)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("matrix", help="full pipeline for ALL schemes + write SUMMARY.md")
    p.add_argument("--schemes", nargs="*", choices=SCHEMES, help="default: all 5")
    _add_orchestrator_args(p)
    p.set_defaults(func=cmd_matrix)

    # ---- test: held-out-only, cost-tiered ----
    p = sub.add_parser(
        "test",
        help="evaluate on the held-out test split only (cost-tiered: local / zeroshot / hybrid)",
    )
    p.add_argument(
        "--dataset",
        default="daime_ft_w100",
        help="dataset variant whose test split (~230) is scored (default: daime_ft_w100)",
    )
    p.add_argument(
        "--run",
        default="",
        help="comma list of tiers to execute: local,zeroshot,hybrid "
        "(default: none = print estimate only)",
    )
    p.add_argument(
        "--difficulty",
        default=None,
        help="run ONE method (name, or 'top') on the full 2214 for the generation-count curve",
    )
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--gpu0", type=int, default=0)
    p.add_argument("--gpu1", type=int, default=1)
    p.set_defaults(func=cmd_test)

    return ap


def main():
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
