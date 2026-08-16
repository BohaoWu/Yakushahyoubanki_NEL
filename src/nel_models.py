#!/usr/bin/env python3
"""NELModel framework for run_all.py.

A base class that captures the common shape of an entity-linking method exposed
as a run_all subcommand: project paths (via scheme_paths), its own venv python +
predict script, dataset loading, a subprocess run, output adaptation and eval.
Each new method = a small subclass overriding `cmd()` (and maybe `adapt()`).

`NEL_dataset` loads a yakusya EL dataset (train/valid/test/entities jsonl) and
produces the 4 alternative splits (pair_disjoint / entity_disjoint / temporal /
book_disjoint) with the same seed and ratios as run_all.make_splits, so the
splits it generates are byte-for-byte identical.

run_all helpers (scheme_paths / run_subprocess / _adapt_chatel_pred / _add_scheme)
are injected via Ctx at registration time, so this module imports nothing from
run_all (no circular import).
"""

import json
import os
import pickle
import sys
from collections import defaultdict

# Load the central config (one dir up, in config/). It owns .env loading, the
# workspace root, the per-method paths and MODELS (per-model run defaults).
import config  # noqa: E402

WS = config.WS
SPLIT_SEED = config.SPLIT_SEED
_SRC_DIR = config.SRC_DIR
BLINK_DIR, BLINK_PY = config.BLINK_DIR, config.BLINK_PY
GENRE_DIR, MGENRE_PY = config.GENRE_DIR, config.MGENRE_PY
MGENRE_TRAIN_PY, MGENRE_BLINK10_PY = config.MGENRE_TRAIN_PY, config.MGENRE_BLINK10_PY
REFINED_DIR, REFINED_PY = config.REFINED_DIR, config.REFINED_PY
REFINED_TRAIN_PY, REFINED_BLINK10_PY = config.REFINED_TRAIN_PY, config.REFINED_BLINK10_PY
DYVO_DIR, DYVO_PY, DYVO_PREP_PY = config.DYVO_DIR, config.DYVO_PY, config.DYVO_PREP_PY
PROJ_EXP, ORIG_DATA = config.PROJ_EXP, config.ORIG_DATA
EXPERIMENTS, LLMAEL_DATA_DIR = config.EXPERIMENTS, config.LLMAEL_DATA_DIR
MODELS = config.MODELS


# Small shared jsonl/title utils (mirror run_all; pure, never diverge).
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def normalize_title(s):
    return (s or "").replace("　", "").replace(" ", "").strip()


class Ctx:
    """Injected run_all helpers (so this module imports nothing from run_all)."""

    def __init__(
        self,
        scheme_paths,
        run_subprocess,
        add_scheme,
        experiments=None,
        run_all_py=None,
        write_jsonl=None,
    ):
        self.scheme_paths = scheme_paths
        self.run = run_subprocess
        self.add_scheme = add_scheme
        self.experiments = experiments  # EXPERIMENTS root (for llmd raw path)
        self.run_all_py = run_all_py  # path to run_all.py (subprocess-to-self, e.g. provider)
        self.write_jsonl = write_jsonl  # run_all.write_jsonl (predictors + adapters)


# The dataset abstraction (read/clean/save/split, both yakusya & HIPE layouts)
# now lives in dataset.py; NEL_dataset is kept as a backward-compatible alias so
# run_all (nel_models.NEL_dataset(...).split(...)) is unchanged.
from dataset import Dataset, NEL_dataset  # noqa: E402,F401


class NELModel:
    """Base for an EL method exposed as a run_all subcommand.

    The shared orchestration (predict) is: resolve per-scheme paths, skip if the
    output already exists, run the method's own venv+script as a subprocess
    (optionally on specific GPUs), then optionally adapt the raw output into the
    unified eval JSONL. A subclass implements cmd() and declares its argparse
    args via add_extra_args(); it overrides out_paths()/adapt()/env() only when
    its naming or output shape differs from the default."""

    name = None  # subcommand name
    help = ""
    venv = sys.executable  # python for the predict subprocess
    script = None  # predict script path
    url = None  # source GitHub repo
    paper_title = None  # title of the originating paper
    paper_url = None  # paper link (ACL Anthology / arXiv)
    gpu_arg = None  # args attr holding CUDA_VISIBLE_DEVICES (e.g. "gpu"); None = CPU
    # candidate-list (constrained) prediction:
    default_num_cands = 10  # config default k when restricting to a candidate list
    candidate_provider = None  # a CandidateProvider (e.g. Blink()) or None = require a file
    constrained_help = None

    # ---- hooks a subclass overrides ----
    def cmd(self, ctx, *, out, raw, chatel, test, entities, args):
        """Return the subprocess command list. Must be implemented.
        `out` is the final eval JSONL; `raw` is the intermediate the adapter reads."""
        raise NotImplementedError

    def adapt(self, ctx, raw, test, entities, out):
        """Convert raw subprocess output -> unified eval JSONL. Default: no-op
        (the subprocess already wrote the final eval JSONL directly to `out`)."""

    def add_extra_args(self, p):
        """Subclass-specific argparse args (beyond --scheme/--out/--skip_existing)."""

    def env(self, args):
        """Extra env for the subprocess (e.g. CUDA_VISIBLE_DEVICES). None = inherit."""
        if self.gpu_arg:
            return {"CUDA_VISIBLE_DEVICES": str(getattr(args, self.gpu_arg))}
        return None

    # ---- common orchestration (shared by all methods) ----
    def out_paths(self, ctx, paths, args):
        """(out, raw, log) for this run. Default: <name>.jsonl with no tag."""
        out = args.out or os.path.join(paths["predictions"], f"{self.name}.jsonl")
        raw = os.path.join(paths["predictions"], f"{self.name}_raw.json")
        log = os.path.join(paths["logs"], f"{self.name}.log")
        return out, raw, log

    def predict(self, ctx, args):
        paths = ctx.scheme_paths(args.scheme)
        chatel = getattr(args, "chatel", None) or paths.get("chatel")
        test = getattr(args, "test", None) or paths["test"]
        entities = getattr(args, "entities", None) or paths["entities"]
        out, raw, log = self.out_paths(ctx, paths, args)
        if getattr(args, "skip_existing", False) and os.path.exists(out):
            print(f"[{self.name}] exists, skip: {out}")
            return
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        cmd = self.cmd(
            ctx, out=out, raw=raw, chatel=chatel, test=test, entities=entities, args=args
        )
        if ctx.run(cmd, log_path=log, env=self.env(args)) != 0:
            print(f"[{self.name}] FAILED (see {log})")
            return
        self.adapt(ctx, raw, test, entities, out)

    def register(self, sub, ctx):
        p = sub.add_parser(self.name, help=self.help)
        ctx.add_scheme(p, required=True)
        self.add_extra_args(p)
        p.add_argument("--out")
        p.add_argument("--skip_existing", action="store_true")
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.predict(_c, a))
        return p

    # ===== candidate-list (constrained) prediction =====
    # The second prediction mode: restrict to a per-mention candidate list (chatel
    # top-k) supplied by another model, instead of scoring the full entity pool.
    def candidates(self, ctx, args, paths):
        """Resolve the candidate-list JSON path. Priority:
        explicit (--candidates / --ice) > the scheme's chatel file. If a
        candidate_provider is set AND the scheme chatel is missing, generate it
        via the provider (count = --num_cands or default_num_cands)."""
        explicit = getattr(args, "candidates", None) or getattr(args, "ice", None)
        if explicit:
            return explicit
        chatel = paths.get("chatel")
        if self.candidate_provider and chatel and not os.path.exists(chatel):
            k = getattr(args, "num_cands", None) or self.default_num_cands
            return self.candidate_provider.provide_candidates(ctx, args, k)
        return chatel

    def constrained_out_paths(self, ctx, paths, args):
        """(out, log) for the constrained run. Default: <name>_blink10.jsonl."""
        base = f"{self.name}_blink10"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, log

    # constrained_inproc=True: predict in this process (a pure-Python adapter like
    # dyvo) instead of spawning a subprocess; such a model implements
    # constrained_predict_inproc() and gets no --ice/--num_cands/--gpu args.
    constrained_inproc = False

    def constrained_cmd(self, ctx, paths, candidates, out, args):
        """Return the constrained-predict subprocess command (subprocess models)."""
        raise NotImplementedError

    def constrained_predict_inproc(self, ctx, paths, out, args):
        """In-process constrained prediction (constrained_inproc models, e.g. dyvo)."""
        raise NotImplementedError

    def add_constrained_args(self, p):
        """Subclass-specific constrained-predict args (beyond --out/--overwrite and,
        for subprocess models, --ice/--num_cands/--gpu)."""

    def predict_constrained(self, ctx, args):
        paths = ctx.scheme_paths(args.scheme)
        out, log = self.constrained_out_paths(ctx, paths, args)
        if os.path.exists(out) and not getattr(args, "overwrite", False):
            print(f"[{self.name}_blink10] exists, skip: {out}")
            return
        if self.constrained_inproc:
            self.constrained_predict_inproc(ctx, paths, out, args)
            return
        cands = self.candidates(ctx, args, paths)
        cmd = self.constrained_cmd(ctx, paths, cands, out, args)
        env = {"CUDA_VISIBLE_DEVICES": str(args.gpu)} if hasattr(args, "gpu") else None
        if ctx.run(cmd, log_path=log, env=env) != 0:
            print(f"[{self.name}_blink10] FAILED (see {log})")

    def register_constrained(self, sub, ctx):
        p = sub.add_parser(
            f"{self.name}_blink10",
            help=self.constrained_help or f"predict {self.name} restricted to a candidate list",
        )
        ctx.add_scheme(p, required=True)
        self.add_constrained_args(p)
        if not self.constrained_inproc:  # subprocess models only
            p.add_argument("--ice", help="override candidate-list (chatel) json")
            p.add_argument(
                "--num_cands", type=int, help="top-k when a provider must generate candidates"
            )
            p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--out")
        p.add_argument("--overwrite", action="store_true")
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.predict_constrained(_c, a))
        return p

    # ===== in-process (torch) global prediction =====
    # Some methods (blink/mgenre/refined) load a model in this process and predict
    # over the full entity pool, instead of spawning a subprocess. The <name>
    # subcommand is registered to predict_inproc; the method must be run under the
    # right venv interpreter (cmd_all launches e.g. REFINED_PY run_all.py refined).
    predict_help = None

    def predict_inproc(self, ctx, args):
        """In-process (torch) full-pool prediction. Must be implemented by such
        models. Heavy deps (torch, the method's package) are imported lazily inside."""
        raise NotImplementedError

    def add_predict_args(self, p):
        """argparse args for the in-process <name> predict subcommand."""

    def register_predict(self, sub, ctx):
        p = sub.add_parser(
            self.name, help=self.predict_help or self.help or f"predict with {self.name}"
        )
        ctx.add_scheme(p, required=False)
        self.add_predict_args(p)
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.predict_inproc(_c, a))
        return p


class CandidateProvider:
    """Role: a model that can produce a per-mention top-k candidate list (chatel
    JSON). Consumers reference one via NELModel.candidate_provider."""

    def provide_candidates(self, ctx, args, k):
        """Return the candidate-list JSON path (generating it if missing)."""
        raise NotImplementedError


class TrainableNELModel(NELModel):
    """A NELModel with an additional training stage, exposed as `train_<name>`.

    The training stage is a SINGLE subprocess into the method's training venv +
    script (mgenre, refined, ...). Multi-step training pipelines (dyvo, dpo) do
    not fit this single-command shape and stay as bespoke run_all functions.

    Prediction is orthogonal: mgenre/refined predict in-process (torch inside
    run_all), so their `predict()` is NOT wired through this framework — only the
    train stage is. A subclass implements train_cmd() (+ train_done()/train_log())
    and declares train defaults via the train_* class attributes.
    """

    train_help = None
    train_epochs = 3
    train_batch_size = 16

    # ---- hooks a subclass overrides ----
    def train_cmd(self, ctx, paths, args):
        """Return the training subprocess command (a shell string, or an argv list)."""
        raise NotImplementedError

    def train_done(self, ctx, paths, args):
        """Path whose existence means 'already trained' (for --skip_existing).
        Return None to never skip."""
        return None

    def train_log(self, paths, args):
        return os.path.join(paths["logs"], f"train_{self.name}.log")

    def train_env(self, args):
        """Extra env for the training subprocess (e.g. PYTHONPATH). None = inherit."""
        return None

    def add_train_args(self, p):
        """Extra train args beyond --gpu/--epochs/--batch_size/--out/--skip_existing."""

    # ---- orchestration ----
    def train(self, ctx, args):
        paths = ctx.scheme_paths(args.scheme)
        done = self.train_done(ctx, paths, args)
        if args.skip_existing and done and os.path.exists(done):
            print(f"[train_{self.name}] exists, skip: {done}")
            return
        log = self.train_log(paths, args)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if ctx.run(self.train_cmd(ctx, paths, args), log_path=log, env=self.train_env(args)) != 0:
            print(f"[train_{self.name}] FAILED (see {log})")

    def register_train(self, sub, ctx):
        p = sub.add_parser(
            f"train_{self.name}", help=self.train_help or f"train {self.name} per-split"
        )
        ctx.add_scheme(p, required=True)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--epochs", type=int, default=self.train_epochs)
        p.add_argument("--batch_size", type=int, default=self.train_batch_size)
        p.add_argument("--out")
        p.add_argument("--skip_existing", action="store_true")
        self.add_train_args(p)
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.train(_c, a))
        return p


class ChatelReranker(NELModel):
    """LLM reranker over a chatel candidate set: --input chatel -> raw -> adapt.

    Subclasses set name/venv/script (+ chatel_required). cmd()/adapt()/out_paths()
    are shared (the chatel adapter reads entities.predict_entity_names; output is
    <name>_<tag>.jsonl where tag defaults to the model basename)."""

    chatel_required = False

    def add_extra_args(self, p):
        if self.chatel_required:
            p.add_argument(
                "--chatel", required=True, help="candidate json (chatel w/ entity_candidates)"
            )
        else:
            p.add_argument(
                "--chatel", help="candidate json (default: scheme gen_blink_cands output)"
            )
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"])
        p.add_argument("--num_workers", type=int, default=cfg["num_workers"])
        p.add_argument("--test", help="override test jsonl")
        p.add_argument("--entities", help="override entities jsonl")
        p.add_argument("--limit", type=int)
        p.add_argument("--tag")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        out = args.out or os.path.join(paths["predictions"], f"{self.name}_{tag}.jsonl")
        raw = os.path.join(paths["predictions"], f"{self.name}_{tag}_raw.json")
        log = os.path.join(paths["logs"], f"{self.name}_{tag}.log")
        return out, raw, log

    def adapt(self, ctx, raw, test, entities, out):
        """chatel output (entities.predict_entity_names) -> unified eval JSONL
        (sample_idx/gold_id/pred_id/pred_title)."""
        ents = load_jsonl(entities)
        title_to_numids = defaultdict(list)
        for e in ents:
            title_to_numids[normalize_title(e["title"])].append(int(e["numeric_id"]))
        rows = load_jsonl(test)
        data = json.load(open(raw))
        recs = []
        for idx, rec in enumerate(rows):
            doc = data.get(f"yakusya_test_{idx}", {})
            preds = doc.get("entities", {}).get("predict_entity_names") or [""]
            pred = preds[0] if preds else ""
            cands = title_to_numids.get(normalize_title(pred), [])
            gold_id = rec.get("label_id")
            pred_id = gold_id if gold_id in cands else (cands[0] if cands else -1)
            recs.append(
                {
                    "sample_idx": idx,
                    "gold_id": gold_id,
                    "gold_title": rec.get("label_title", ""),
                    "pred_id": pred_id,
                    "pred_title": pred,
                    "mention": rec.get("mention"),
                    "type": rec.get("type"),
                }
            )
        ctx.write_jsonl(recs, out)
        correct = sum(1 for r in recs if r["pred_id"] == r["gold_id"])
        print(f"[adapt] {correct}/{len(recs)} = {correct / len(recs) * 100:.2f}% → {out}")

    def cmd(self, ctx, *, chatel, raw, args, **_):
        c = [
            self.venv,
            self.script,
            "--input",
            chatel,
            "--output",
            raw,
            "--model",
            args.model,
            "--num_workers",
            str(args.num_workers),
        ]
        if args.limit:
            c += ["--max_samples", str(args.limit)]
        return c


class Incel(ChatelReranker):
    name = "In_Context_EL"
    help = "In-Context-EL LLM reranker over a chatel candidate set"
    venv = f"{WS}/In_Context_EL/venv/bin/python"
    script = f"{WS}/In_Context_EL/scripts/run_chatel_yakusya.py"
    url = "https://github.com/yifding/In_Context_EL"
    paper_title = "ChatEL: Entity Linking with Chatbots"
    paper_url = "https://aclanthology.org/2024.lrec-main.275/"


class DeepEL(ChatelReranker):
    name = "DeepEL"
    help = "DeepEL self-validation on an In-Context-EL chatel output"
    venv = f"{WS}/DeepEL/venv/bin/python"
    script = f"{WS}/DeepEL/scripts/run_deepel_yakusya.py"
    chatel_required = True
    url = "https://github.com/SStan1/DeepEL"
    paper_title = "Harnessing Deep LLM Participation for Robust Entity Linking"
    paper_url = "https://arxiv.org/abs/2511.14181"

    def cmd(self, ctx, *, chatel, raw, args, **_):
        # DeepEL refines In_Context_EL's predictions: its --input must carry
        # predict_entity_names + multi_choice_prompt_results, which live in the
        # In_Context_EL raw output, NOT in the bare candidate file. Chain from
        # that output when present; fall back to the given chatel otherwise.
        ic_raw = os.path.join(
            os.path.dirname(raw), os.path.basename(raw).replace("DeepEL_", "In_Context_EL_")
        )
        src = ic_raw if os.path.exists(ic_raw) else chatel
        c = [
            self.venv,
            self.script,
            "--input",
            src,
            "--output",
            raw,
            "--model",
            args.model,
            "--num_workers",
            str(args.num_workers),
        ]
        if args.limit:
            c += ["--max_samples", str(args.limit)]
        return c


class SumMC(ChatelReranker):
    name = "SumMC"
    help = "SumMC unsupervised summarize + multiple-choice over chatel candidates"
    venv = f"{WS}/SumMC/venv/bin/python"
    script = f"{WS}/SumMC/code/run_summc_yakusya.py"
    url = "https://github.com/JeffreyCh0/SumMC"
    paper_title = (
        "Unsupervised Entity Linking with Guided Summarization and Multiple-Choice Selection"
    )
    paper_url = "https://aclanthology.org/2022.emnlp-main.638/"


class OneNet(NELModel):
    """OneNet (LLaMA-70B) reranker over BLINK top-10 candidates; writes eval JSONL
    directly (no adapt). Runs in the OneNet venv on the requested GPUs."""

    name = "onenet"
    help = "OneNet (LLaMA-70B) predict per-split"
    venv = config.ONENET_PY
    script = f"{WS}/OneNet/rerun_yakusya_blink_cands.py"
    gpu_arg = "gpu"
    url = "https://github.com/laquabe/OneNet"
    paper_title = "OneNet: A Fine-Tuning Free Framework for Few-Shot Entity Linking via Large Language Model Prompting"
    paper_url = "https://arxiv.org/abs/2410.07549"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--test", help="override test jsonl")
        p.add_argument("--entities", help="override entities jsonl")
        p.add_argument("--ice", help="override BLINK10 chatel json")
        p.add_argument("--model", default=cfg["model"])
        p.add_argument("--num_cands", type=int, default=cfg["num_cands"])
        p.add_argument("--gpu", default=cfg["gpu"])

    def cmd(self, ctx, *, out, chatel, test, entities, args, **_):
        return [
            self.venv,
            self.script,
            "--model",
            args.model,
            "--num_cands",
            str(args.num_cands),
            "--out",
            out,
            "--test",
            test,
            "--entities",
            entities,
            "--ice",
            args.ice or chatel,
        ]


class Llmd(NELModel):
    """llm_disambiguator (gpt-4o-mini etc.): runs the project's run_yakusya.py to a
    raw results.json, then adapts it (predicted title -> numeric_id) to eval JSONL."""

    name = "llmd"
    help = "llm_disambiguator (gpt-4o-mini) per-split"
    venv = config.LLMD_PY
    script = f"{WS}/llm_disambiguator/scripts/run_yakusya.py"
    url = "https://github.com/ChristopheYe/llm_disambiguator"
    paper_title = "LLM as Entity Disambiguator for Biomedical Entity Linking"
    paper_url = "https://aclanthology.org/2025.acl-short.25/"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"])
        p.add_argument("--num_workers", type=int, default=cfg["num_workers"])
        p.add_argument("--src", help="override data dir (reads train/test/entities)")
        p.add_argument("--candidates", help="override BLINK10 chatel json")
        p.add_argument("--test", help="override test jsonl (for eval adapter)")
        p.add_argument("--entities", help="override entities jsonl (for eval adapter)")
        p.add_argument("--tag", default="", help="output filename suffix")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or ""
        out = args.out or os.path.join(paths["predictions"], f"llm_disambiguator{tag}.jsonl")
        # raw: legacy in-repo path for the bare original run; else under experiments/<scheme>/
        if args.scheme == "original" and not args.out:
            raw = f"{WS}/llm_disambiguator/results/yakusya/results.json"
        else:
            raw = os.path.join(ctx.experiments, args.scheme, f"llmd_results{tag}.json")
        log = os.path.join(paths["logs"], f"llm_disambiguator{tag}.log")
        return out, raw, log

    def cmd(self, ctx, *, raw, chatel, args, **_):
        src = args.src or ctx.scheme_paths(args.scheme)["data"]
        candidates = args.candidates or chatel
        return [
            self.venv,
            self.script,
            "--src",
            src,
            "--candidates",
            candidates,
            "--output",
            raw,
            "--model",
            args.model,
            "--num_workers",
            str(args.num_workers),
        ]

    def adapt(self, ctx, raw, test, entities, out):
        """llm_disambiguator results.json (predicted title) -> unified eval JSONL."""
        ents = load_jsonl(entities)
        title_to_numids = defaultdict(list)
        for e in ents:
            title_to_numids[normalize_title(e["title"])].append(int(e["numeric_id"]))
        rows = load_jsonl(test)
        raw_data = json.load(open(raw))
        recs = []
        for idx, rec in enumerate(rows):
            key = f"yakusya_test_{idx}"
            pred = raw_data.get(key, {}).get("predicted", "") if isinstance(raw_data, dict) else ""
            cands = title_to_numids.get(normalize_title(pred), [])
            gold_id = rec.get("label_id")
            if gold_id in cands:
                pred_id = gold_id
            elif cands:
                pred_id = cands[0]
            else:
                pred_id = -1
            recs.append(
                {
                    "sample_idx": idx,
                    "gold_id": gold_id,
                    "gold_title": rec.get("label_title", ""),
                    "pred_id": pred_id,
                    "pred_title": pred,
                    "mention": rec.get("mention"),
                    "type": rec.get("type"),
                }
            )
        ctx.write_jsonl(recs, out)
        correct = sum(1 for r in recs if r["pred_id"] == r["gold_id"])
        print(f"[llmd] {correct}/{len(recs)} = {correct / len(recs) * 100:.2f}% → {out}")


class MAD(NELModel):
    """MAD: Multi-Agent Debate disambiguator (Affirmative/Negative/Moderator).
    Wraps methods/disambiguators/llm_debate_disambiguator.py; writes eval JSONL
    directly (no adapt)."""

    name = "mad"
    help = "MAD multi-agent debate disambiguator per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/llm_debate_disambiguator.py")
    url = "https://github.com/Skytliang/Multi-Agents-Debate"
    paper_title = (
        "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate"
    )
    paper_url = "https://aclanthology.org/2024.emnlp-main.992/"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument(
            "--model", default=cfg["model"], help="debater model id (OpenAI or vLLM endpoint)"
        )
        p.add_argument("--judge_model", help="moderator/verifier model (default: --model)")
        p.add_argument(
            "--rounds",
            type=int,
            default=cfg["rounds"],
            help="aff/neg debate rounds before the moderator decides",
        )
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists to debate among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--verify",
            action="store_true",
            help="adversarial verification of the moderator pick (off: net-negative)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--rounds",
            str(args.rounds),
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.judge_model:
            c += ["--judge_model", args.judge_model]
        if args.verify:
            c.append("--verify")
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class TransMAD(MAD):
    """TransMAD: a translation-augmented derivative of MAD for this dataset. Before
    the debate, an English-prompted interpreter clarifies the archaic Edo-period
    playbill context and extracts the verbatim co-star names (the target's same-
    playbill contemporaries) — the strongest cue for pinning the actor's generation
    (daime). Reuses MAD's debate script with --translate; actor names stay verbatim
    Japanese, only the interpretation prose is English."""

    name = "transmad"
    help = "Translation-augmented MAD (clarify context + co-star extraction, then debate)"
    url = "https://github.com/Skytliang/Multi-Agents-Debate"  # base method; interpreter stage is our extension
    paper_title = None
    paper_url = None

    def cmd(self, ctx, *, out, test, entities, args, **_):
        return super().cmd(ctx, out=out, test=test, entities=entities, args=args) + ["--translate"]


class Reflexion(NELModel):
    """Reflexion disambiguator: a Reflexion-style (Shinn et al. 2023) derivative of
    MAD. Reuses MAD's candidate gathering + structured kaimeihyou/year evidence (by
    import, MAD code unchanged) but replaces the 3-role debate with a single agent's
    self-reflection retry loop: solve -> gold-free self-verify (year/kaimeihyou/daime
    consistency) -> reflect on the critique -> retry, up to max_trials. Writes the
    eval JSONL directly (no adapt)."""

    name = "reflexion"
    help = "Reflexion self-reflection retry disambiguator (MAD evidence, no debate)"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/reflexion_disambiguator.py")
    url = "https://github.com/noahshinn/reflexion"
    paper_title = "Reflexion: Language Agents with Verbal Reinforcement Learning"
    paper_url = "https://arxiv.org/abs/2303.11366"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"], help="solver/agent model id")
        p.add_argument("--judge_model", help="verifier model (default: --model)")
        p.add_argument(
            "--max_trials", type=int, default=cfg["max_trials"], help="reflexion trials per mention"
        )
        p.add_argument("--k", type=int, default=cfg["k"], help="number of year-ranked finalists")
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--max_trials",
            str(args.max_trials),
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.judge_model:
            c += ["--judge_model", args.judge_model]
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class MapsEL(NELModel):
    """MAPS-EL: the MAPS translation strategy (He et al., TACL 2024) transplanted
    onto daime disambiguation. Mines three orthogonal disambiguation views
    (contemporaries / chronology / nomenclature), integrates each into an
    independent grounded entity pick, then a QE-style selector picks the final
    entity (consensus is taken for free). Reuses MAD's candidate gathering +
    structured evidence by import; writes the eval JSONL directly (no adapt)."""

    name = "maps"
    help = "MAPS multi-aspect prompting + selection disambiguator per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/maps_rerank.py")
    url = "https://github.com/zwhe99/MAPS-mt"
    paper_title = "Exploring Human-Like Translation Strategy with Large Language Models"
    paper_url = "https://aclanthology.org/2024.tacl-1.13/"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument(
            "--model",
            default=cfg["model"],
            help="mining/integration model id (OpenAI or vLLM endpoint)",
        )
        p.add_argument("--select_model", help="selection-stage model (default: --model)")
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists the views choose among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--no_mine",
            action="store_true",
            help="ablation: skip aspect mining (base pick + selection only)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            self.name,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.select_model:
            c += ["--select_model", args.select_model]
        if args.no_mine:
            c.append("--no_mine")
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class Rerank(NELModel):
    """Rerank: the MAPS paper's `Rerank` baseline (temperature-sample N candidate
    translations, then an external reference-free QE model picks the best)
    transplanted onto daime disambiguation. Samples N stochastic + 1 greedy
    entity picks from one grounded prompt, then an independent LLM QE judge scores
    each distinct candidate 0-100 (COMET-QE analogue) and the top is kept (ties
    break on self-consistency votes, then year plausibility). Reuses MAD's
    candidate gathering + evidence by import; writes the eval JSONL directly."""

    name = "rerank"
    help = "Sample-and-QE-rerank disambiguator (MAPS Rerank baseline) per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/maps_rerank.py")
    url = "https://github.com/zwhe99/MAPS-mt"
    paper_title = "Exploring Human-Like Translation Strategy with Large Language Models"
    paper_url = "https://aclanthology.org/2024.tacl-1.13/"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument(
            "--model", default=cfg["model"], help="sampling model id (OpenAI or vLLM endpoint)"
        )
        p.add_argument("--qe_model", help="external QE-judge model (default: --model)")
        p.add_argument(
            "--n_samples",
            type=int,
            default=cfg["n_samples"],
            help="stochastic samples per mention (a greedy sample is always added)",
        )
        p.add_argument(
            "--temperature",
            type=float,
            default=cfg["temperature"],
            help="sampling temperature for the stochastic picks",
        )
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists to sample/rerank among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            self.name,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--n_samples",
            str(args.n_samples),
            "--temperature",
            str(args.temperature),
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.qe_model:
            c += ["--qe_model", args.qe_model]
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class ZeroShotCoT(NELModel):
    """Zero-shot-CoT: the Kojima et al. (NeurIPS 2022) two-stage prompt ("Let's
    think step by step" -> reasoning, then an answer-extraction trigger)
    transplanted onto daime disambiguation. The simplest LLM baseline here — one
    agent, two calls, no debate / sampling / multi-aspect prompting. `--no_cot`
    runs the paper's plain Zero-shot ablation. Reuses MAD's candidate gathering +
    evidence by import; writes the eval JSONL directly (no adapt)."""

    name = "zscot"
    help = "Zero-shot-CoT ('Let's think step by step') disambiguator per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/cot_family.py")
    url = "https://github.com/kojima-takeshi188/zero_shot_cot"
    paper_title = "Large Language Models are Zero-Shot Reasoners"
    paper_url = "https://arxiv.org/abs/2205.11916"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"], help="reasoner model id")
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists to reason among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--no_cot",
            action="store_true",
            help="ablation: plain Zero-shot (drop the step-by-step trigger)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            self.name,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.no_cot:
            c.append("--no_cot")
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class SelfConsistency(NELModel):
    """Self-Consistency: the Wang et al. (ICLR 2023) method (sample N CoT
    reasoning paths at temperature, majority-vote the answers) transplanted onto
    daime disambiguation. Same sampling stage as Rerank, but selection is plain
    majority vote over the sampled entity_ids (ties -> year plausibility), no
    external QE scorer. Reuses MAD's candidate gathering + evidence by import;
    writes the eval JSONL directly (no adapt)."""

    name = "selfcon"
    help = "Self-Consistency (sample CoT paths + majority vote) disambiguator per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/sampling.py")
    url = "https://github.com/dj-sorry/self_consistency"  # no official repo; community impl
    paper_title = "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
    paper_url = "https://arxiv.org/abs/2203.11171"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"], help="reasoner model id")
        p.add_argument(
            "--n_samples",
            type=int,
            default=cfg["n_samples"],
            help="number of sampled CoT reasoning paths to vote over",
        )
        p.add_argument(
            "--temperature",
            type=float,
            default=cfg["temperature"],
            help="sampling temperature for the CoT paths",
        )
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists to reason among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            self.name,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--n_samples",
            str(args.n_samples),
            "--temperature",
            str(args.temperature),
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class PlanSolve(NELModel):
    """Plan-and-Solve: the Wang et al. (ACL 2023) prompt (a richer Zero-shot-CoT
    trigger that devises a plan then carries it out) transplanted onto daime
    disambiguation. Two calls (plan+solve, then answer extraction); a training-
    free stand-in for the plan+execute two-layer idea. `--plus` uses the PS+
    trigger. Reuses MAD's candidate gathering + evidence by import; writes the
    eval JSONL directly (no adapt)."""

    name = "plansolve"
    help = "Plan-and-Solve prompting disambiguator per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/cot_family.py")
    url = "https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting"
    paper_title = "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models"
    paper_url = "https://aclanthology.org/2023.acl-long.147/"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"], help="reasoner model id")
        p.add_argument(
            "--plus", action="store_true", help="use the PS+ trigger (extract cues, then plan)"
        )
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists to reason among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            self.name,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.plus:
            c.append("--plus")
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class CoVe(NELModel):
    """Chain-of-Verification: the Dhuliawala et al. (2023) four-stage method
    (baseline -> plan verification questions -> execute them independently ->
    refine) transplanted onto daime disambiguation. The independent verification
    runs against the deterministic structured evidence, so it catches where the
    baseline pick contradicts the year / kaimeihyou facts. Reuses MAD's candidate
    gathering + evidence by import; writes the eval JSONL directly (no adapt)."""

    name = "cove"
    help = "Chain-of-Verification disambiguator per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/disambiguators/sampling.py")
    url = "https://github.com/ritun16/chain-of-verification"
    paper_title = "Chain-of-Verification Reduces Hallucination in Large Language Models"
    paper_url = "https://aclanthology.org/2024.findings-acl.212/"

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument("--model", default=cfg["model"], help="model id (all four stages)")
        p.add_argument(
            "--k",
            type=int,
            default=cfg["k"],
            help="number of year-ranked finalists to verify among",
        )
        p.add_argument(
            "--max_steps",
            type=int,
            default=cfg["max_steps"],
            help="tool-use steps for the baseline answer",
        )
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY)")
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl (must match --test variant)")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            self.name,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--k",
            str(args.k),
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class LLMTool(NELModel):
    """LLMTool: our tool-use LLM agent. It answers each mention by calling lookup
    tools (kaimeihyou name periods, entity catalog, ...) over any OpenAI-compatible
    endpoint — the OpenAI API (default) or a local vLLM server (--base_url --model).
    Wraps methods/core/llm_tool_disambiguator.py; writes eval JSONL
    directly (no adapt). --fair strips the 3 leaky kaimeihyou tools."""

    name = "llm_tool"
    help = "tool-use LLM agent (OpenAI or local vLLM) per-split"
    venv = sys.executable
    script = os.path.join(_SRC_DIR, "methods/core/llm_tool_disambiguator.py")

    def add_extra_args(self, p):
        cfg = MODELS[self.name]
        p.add_argument(
            "--model",
            default=cfg["model"],
            help="model id (e.g. gpt-4o-mini, meta-llama/Llama-3.1-70B-Instruct)",
        )
        p.add_argument(
            "--base_url", help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1"
        )
        p.add_argument("--api_key", help="API key (defaults to $OPENAI_API_KEY or EMPTY)")
        p.add_argument("--max_steps", type=int, default=cfg["max_steps"])
        p.add_argument(
            "--num_workers",
            type=int,
            default=cfg["num_workers"],
            help="concurrent samples (remote LLM API calls run in parallel)",
        )
        p.add_argument("--test", help="override test jsonl (e.g. dataset_multi_daime/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl")
        p.add_argument("--fair", action="store_true", help="strip the 3 leaky kaimeihyou tools")
        p.add_argument("--limit", type=int)
        p.add_argument("--start", type=int)
        p.add_argument("--tag", help="suffix for output filename (default: model basename)")

    def out_paths(self, ctx, paths, args):
        tag = args.tag or args.model.split("/")[-1]
        base = f"{self.name}_{tag}"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        raw = os.path.join(paths["predictions"], f"{base}_raw.json")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, raw, log

    def cmd(self, ctx, *, out, test, entities, args, **_):
        c = [
            self.venv,
            self.script,
            "--test",
            test,
            "--entities",
            entities,
            "--out",
            out,
            "--model",
            args.model,
            "--max_steps",
            str(args.max_steps),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.fair:
            c.append("--fair")
        if args.base_url:
            c += ["--base_url", args.base_url]
        if args.api_key:
            c += ["--api_key", args.api_key]
        if args.limit:
            c += ["--limit", str(args.limit)]
        if args.start:
            c += ["--start", str(args.start)]
        return c


class Blink(TrainableNELModel, CandidateProvider):
    """BLINK bi-encoder. Training is a subprocess into the BLINK library (run as
    `python -m blink.biencoder.train_biencoder` with PYTHONPATH=BLINK_DIR);
    prediction (the `blink` subcommand) runs in-process in run_all, so only
    training goes through the framework here. Also a CandidateProvider: it
    generates the BLINK top-k chatel list other models restrict to."""

    name = "blink"
    train_help = "train BLINK bi-encoder per-split"
    train_batch_size = MODELS["blink"]["train_batch_size"]
    url = "https://github.com/facebookresearch/BLINK"
    paper_title = "Scalable Zero-shot Entity Linking with Dense Entity Retrieval"
    paper_url = "https://aclanthology.org/2020.emnlp-main.519/"

    def add_train_args(self, p):
        p.add_argument("--bert_model", default=MODELS["blink"]["bert_model"])

    def train_done(self, ctx, paths, args):
        out_model = args.out or os.path.join(paths["models"], "blink_v2")
        return os.path.join(out_model, "pytorch_model.bin")

    def train_log(self, paths, args):
        return os.path.join(paths["logs"], "train_blink_v2.log")

    def train_env(self, args):
        return {"PYTHONPATH": BLINK_DIR}

    def train_cmd(self, ctx, paths, args):
        out_model = args.out or os.path.join(paths["models"], "blink_v2")
        # BLINK_MAX_CTX overrides the mention-context token budget (default 128).
        # Used by the context-window ablation to let wider char windows actually
        # reach the encoder; capped by mBERT's 512 positional limit.
        mc = int(os.environ.get("BLINK_MAX_CTX", "128"))
        return (
            f"cd {BLINK_DIR} && CUDA_VISIBLE_DEVICES={args.gpu} {BLINK_PY} "
            f"-m blink.biencoder.train_biencoder "
            f"--data_path {paths['data']} --output_path {out_model} "
            f"--bert_model {args.bert_model} "
            f"--max_context_length {mc} --max_cand_length 128 --max_seq_length {mc + 128} "
            f"--num_train_epochs {args.epochs} --train_batch_size {args.batch_size} "
            f"--learning_rate 2e-5 --eval_batch_size 8 --type_optimization all_encoder_layers "
            f"--save_interval 1 --print_interval 50 --eval_interval 500 "
            f"--shuffle True --zeshel True"
        )

    def provide_candidates(self, ctx, args, k):
        """BLINK top-k candidate list (chatel JSON) for the scheme, cached. If the
        scheme chatel is missing, generate it via the gen_blink_cands subcommand
        (run under BLINK_PY so the torch deps are present)."""
        paths = ctx.scheme_paths(args.scheme)
        out = paths["chatel"]
        if os.path.exists(out):
            return out
        log = os.path.join(paths["logs"], "gen_blink_cands.log")
        ctx.run(
            [
                BLINK_PY,
                ctx.run_all_py,
                "gen_blink_cands",
                "--scheme",
                args.scheme,
                "--top_k",
                str(k),
            ],
            log_path=log,
        )
        return out

    # ---- in-process full-pool prediction (was run_all.cmd_blink) ----
    predict_help = "predict with BLINK bi-encoder"

    def add_predict_args(self, p):
        p.add_argument("--data_path")
        p.add_argument("--model_path", required=True)
        p.add_argument("--bert_model", default="bert-base-multilingual-cased")
        p.add_argument("--out", required=True)
        p.add_argument("--mode", default="test")
        p.add_argument("--max_context_length", type=int, default=128)
        p.add_argument("--max_cand_length", type=int, default=128)
        p.add_argument("--eval_batch_size", type=int, default=16)
        p.add_argument("--encode_batch_size", type=int, default=64)
        p.add_argument(
            "--entities-override",
            dest="entities_override",
            default=None,
            help="path to alternative entities.jsonl (for clean-catalog experiments)",
        )

    def predict_inproc(self, ctx, args):
        """BLINK bi-encoder vs full entity pool, save predictions."""
        paths = ctx.scheme_paths(args.scheme) if args.scheme else None
        data_path = args.data_path or (paths["data"] if paths else None)
        out = args.out
        sys.path.insert(0, BLINK_DIR)
        import blink.biencoder.data_process as bidata
        import blink.candidate_ranking.utils as utils
        import torch
        from blink.biencoder.biencoder import BiEncoderRanker
        from torch.utils.data import DataLoader, SequentialSampler
        from tqdm import tqdm

        params = {
            "path_to_model": os.path.join(args.model_path, "pytorch_model.bin"),
            "bert_model": args.bert_model,
            "max_context_length": args.max_context_length,
            "max_cand_length": args.max_cand_length,
            "max_seq_length": args.max_context_length,
            "lowercase": True,
            "no_cuda": False,
            "data_parallel": False,
            "out_dim": 1,
            "pull_from_layer": -1,
            "add_linear": False,
            "silent": True,
            "debug": False,
            "context_key": "context",
            "eval_batch_size": args.eval_batch_size,
            "output_path": args.model_path,
            "seed": 52313,
        }
        logger = utils.get_logger(params["output_path"])
        reranker = BiEncoderRanker(params)
        tokenizer = reranker.tokenizer
        device = reranker.device

        entity_path = getattr(args, "entities_override", None) or os.path.join(
            data_path, "entities.jsonl"
        )
        entities = load_jsonl(entity_path)
        print(f"[blink] {len(entities)} entities from {entity_path}, model={args.model_path}")

        cand_token_ids = []
        for ent in tqdm(entities, desc="tok ent"):
            rep = bidata.get_candidate_representation(
                ent.get("text", ent.get("description", "")),
                tokenizer,
                args.max_cand_length,
                ent.get("title", ent.get("label_title", "")),
            )
            cand_token_ids.append(rep["ids"])
        cand_pool = torch.LongTensor(cand_token_ids)
        reranker.model.eval()
        cand_encodes_chunks = []
        sampler = SequentialSampler(cand_pool)
        loader = DataLoader(cand_pool, sampler=sampler, batch_size=args.encode_batch_size)
        with torch.no_grad():
            for batch in tqdm(loader, desc="encode ent"):
                cand_encodes_chunks.append(reranker.encode_candidate(batch.to(device)))
        cand_encodes = torch.cat(cand_encodes_chunks, dim=0).to(device)

        samples = utils.read_dataset(args.mode, data_path)
        print(f"[blink] {len(samples)} {args.mode} samples")

        test_data, _ = bidata.process_mention_data(
            samples,
            tokenizer,
            args.max_context_length,
            args.max_cand_length,
            context_key="context",
            silent=True,
            logger=logger,
            debug=False,
        )
        context_vecs = test_data["context_vecs"]
        pred_line_idx = []
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), args.eval_batch_size), desc="score"):
                batch_ctx = context_vecs[i : i + args.eval_batch_size].to(device)
                scores = reranker.score_candidate(batch_ctx, None, cand_encs=cand_encodes)
                top1 = scores.argmax(dim=1).cpu().numpy().tolist()
                pred_line_idx.extend(top1)

        line_to_numid = [int(e.get("numeric_id", i)) for i, e in enumerate(entities)]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            for idx, (sample, pi) in enumerate(zip(samples, pred_line_idx)):
                f.write(
                    json.dumps(
                        {
                            "sample_idx": idx,
                            "gold_id": sample.get("label_id"),
                            "gold_title": sample.get("label_title", ""),
                            "pred_id": line_to_numid[pi],
                            "pred_title": entities[pi].get("title", ""),
                            "mention": sample.get("mention"),
                            "type": sample.get("type"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[blink] saved {len(pred_line_idx)} → {out}")

    # ---- candidate generation: BLINK top-k -> chatel JSON (was run_all.cmd_gen_blink_cands) ----
    def add_gen_cands_args(self, p):
        p.add_argument("--blink_model", help="default: <scheme>/models/blink_v2")
        p.add_argument("--bert_model", default="bert-base-multilingual-cased")
        p.add_argument("--data", help="override data dir (reads <dir>/test.jsonl)")
        p.add_argument("--entities", help="override entities jsonl")
        p.add_argument("--out")
        p.add_argument("--top_k", type=int, default=10)
        p.add_argument("--overwrite", action="store_true")

    def register_gen_cands(self, sub, ctx):
        p = sub.add_parser("gen_blink_cands", help="BLINK top-10 → chatel JSON")
        ctx.add_scheme(p, required=True)
        self.add_gen_cands_args(p)
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.gen_candidates(_c, a))
        return p

    def gen_candidates(self, ctx, args):
        """Generate per-split BLINK top-k candidates in chatel JSON format
        (the format OneNet/ReFinED-blink10/mgenre-blink10 all consume)."""
        sys.path.insert(0, BLINK_DIR)
        import blink.biencoder.data_process as bidata
        import blink.candidate_ranking.utils as utils
        import torch
        from blink.biencoder.biencoder import BiEncoderRanker
        from torch.utils.data import DataLoader, SequentialSampler
        from tqdm import tqdm

        paths = ctx.scheme_paths(args.scheme)
        blink_model = args.blink_model or os.path.join(paths["models"], "blink_v2")
        out = args.out or paths["chatel"]
        if os.path.exists(out) and not args.overwrite:
            print(f"[gen_blink_cands] exists, skip: {out}")
            return

        params = {
            "path_to_model": os.path.join(blink_model, "pytorch_model.bin"),
            "bert_model": args.bert_model,
            "max_context_length": 128,
            "max_cand_length": 128,
            "max_seq_length": 128,
            "lowercase": True,
            "no_cuda": False,
            "data_parallel": False,
            "out_dim": 1,
            "pull_from_layer": -1,
            "add_linear": False,
            "silent": True,
            "debug": False,
            "context_key": "context",
            "eval_batch_size": 16,
            "output_path": blink_model,
            "seed": 52313,
        }
        logger = utils.get_logger(blink_model)
        reranker = BiEncoderRanker(params)
        tok = reranker.tokenizer

        ents = load_jsonl(args.entities or paths["entities"])
        print(f"[gen_blink_cands] entities={len(ents)} model={blink_model}")
        cand_ids = []
        for e in tqdm(ents, desc="tok ent"):
            rep = bidata.get_candidate_representation(
                e.get("text", "") or "", tok, 128, e.get("title", "")
            )
            cand_ids.append(rep["ids"])
        cand_pool = torch.LongTensor(cand_ids)
        sampler = SequentialSampler(cand_pool)
        loader = DataLoader(cand_pool, sampler=sampler, batch_size=64)
        reranker.model.eval()
        encs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="enc ent"):
                encs.append(reranker.encode_candidate(batch.to(reranker.device)))
        cand_encs = torch.cat(encs, dim=0).to(reranker.device)

        samples = utils.read_dataset("test", args.data or paths["data"])
        test_data, _ = bidata.process_mention_data(
            samples, tok, 128, 128, context_key="context", silent=True, logger=logger, debug=False
        )
        context_vecs = test_data["context_vecs"]

        result = {}
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), 16), desc="retrieve"):
                cvec = context_vecs[i : i + 16].to(reranker.device)
                scores = reranker.score_candidate(cvec, None, cand_encs=cand_encs)
                topk = scores.topk(args.top_k, dim=1).indices.cpu().numpy().tolist()
                for j, top in enumerate(topk):
                    idx = i + j
                    cand_titles = [ents[k]["title"] for k in top]
                    # description per candidate (parallel to entity_candidates) —
                    # In_Context_EL / DeepEL require entity_candidates_descriptions
                    cand_descs = [(ents[k].get("text", "") or "")[:1000] for k in top]
                    result[f"yakusya_test_{idx}"] = {
                        "doc_name": f"yakusya_test_{idx}",
                        "sentence": (
                            samples[idx].get("context_left", "")
                            + samples[idx]["mention"]
                            + samples[idx].get("context_right", "")
                        ),
                        "entities": {
                            "starts": [len(samples[idx].get("context_left", ""))],
                            "ends": [
                                len(samples[idx].get("context_left", ""))
                                + len(samples[idx]["mention"])
                            ],
                            "entity_mentions": [samples[idx]["mention"]],
                            "entity_names": [samples[idx].get("label_title", "")],
                            "entity_candidates": [cand_titles],
                            "entity_candidates_descriptions": [cand_descs],
                        },
                    }
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"[gen_blink_cands] saved {len(result)} → {out}")


def _make_mgenre_input(rec):
    m = rec["mention"]
    l = rec.get("context_left", "")[-128:]
    r = rec.get("context_right", "")[:128]
    return f"{l} [START] {m} [END] {r}".strip()


class MGenre(TrainableNELModel):
    """mGENRE-FT: trie-constrained multilingual autoregressive EL. All stages here:
    train (subprocess) -> train_mgenre; full-pool predict (in-process, generative
    + trie) -> mgenre; BLINK10-constrained (subprocess) -> mgenre_blink10."""

    name = "mgenre"
    train_help = "train mGENRE-FT per-split"
    predict_help = "predict with mGENRE (full pool)"
    train_batch_size = MODELS["mgenre"]["train_batch_size"]
    constrained_help = "predict mGENRE constrained to BLINK top-10"
    url = "https://github.com/facebookresearch/GENRE"
    paper_title = "Multilingual Autoregressive Entity Linking"
    paper_url = "https://arxiv.org/abs/2103.12528"

    # ---- in-process full-pool prediction (was run_all.cmd_mgenre) ----
    def add_predict_args(self, p):
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--test")
        p.add_argument("--entities")
        p.add_argument("--trie")
        p.add_argument("--batch_size", type=int, default=32)

    def predict_inproc(self, ctx, args):
        """mGENRE with trie-constrained beam search over all entities."""
        sys.path.insert(0, GENRE_DIR)
        from genre.hf_model import mGENRE

        paths = ctx.scheme_paths(args.scheme) if args.scheme else None
        test_path = args.test or (paths["test"] if paths else None)
        ent_path = args.entities or (paths["entities"] if paths else None)
        trie_path = args.trie or (paths["trie"] if paths else None)

        print(f"[mgenre] loading {args.model}...")
        model = mGENRE.from_pretrained(args.model).eval()
        if hasattr(model, "cuda"):
            model = model.cuda()
        with open(trie_path, "rb") as f:
            trie = pickle.load(f)

        test = load_jsonl(test_path)
        ents = load_jsonl(ent_path)
        title2nids = defaultdict(list)
        for e in ents:
            title2nids[normalize_title(e["title"])].append(int(e.get("numeric_id")))
        print(f"[mgenre] test={len(test)} entities={len(ents)} unique_titles={len(title2nids)}")

        out = []
        bs = args.batch_size
        for i in range(0, len(test), bs):
            batch = test[i : i + bs]
            sentences = [_make_mgenre_input(r) for r in batch]
            eos = model.tokenizer.eos_token_id
            vocab_limit = len(model.tokenizer) - 1

            def _allowed(batch_id, sent):
                toks = [e for e in trie.get(sent.tolist()) if e < vocab_limit]
                return toks if toks else [eos]

            results = model.sample(
                sentences,
                prefix_allowed_tokens_fn=_allowed,
                num_beams=5,
                num_return_sequences=1,
            )
            for off, (rec, res) in enumerate(zip(batch, results)):
                sample_idx = i + off
                pred_text = res[0]["text"] if res else ""
                pred_title = (
                    pred_text.split(" >> ")[0] if " >> " in pred_text else pred_text
                ).strip()
                cand_ids = title2nids.get(normalize_title(pred_title), [])
                gold_id = rec.get("label_id")
                if gold_id in cand_ids:
                    pred_id = gold_id
                elif cand_ids:
                    pred_id = cand_ids[0]
                else:
                    pred_id = -1
                out.append(
                    {
                        "sample_idx": sample_idx,
                        "gold_id": gold_id,
                        "gold_title": rec.get("label_title", ""),
                        "pred_id": pred_id,
                        "pred_title": pred_title,
                        "mention": rec.get("mention"),
                        "type": rec.get("type"),
                    }
                )
            if (i + bs) % 200 == 0 or i + bs >= len(test):
                print(f"  [{min(i + bs, len(test))}/{len(test)}]")

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        ctx.write_jsonl(out, args.out)
        correct = sum(1 for r in out if r["pred_id"] == r["gold_id"])
        print(
            f"[mgenre] saved {len(out)} → {args.out}  (id-match {correct}/{len(out)} = {correct / len(out) * 100:.2f}%)"
        )

    def train_done(self, ctx, paths, args):
        out_model = args.out or os.path.join(paths["models"], "mgenre_ft")
        return os.path.join(out_model, "pytorch_model.bin")

    def train_log(self, paths, args):
        return os.path.join(paths["logs"], "train_mgenre_ft.log")

    def train_cmd(self, ctx, paths, args):
        out_model = args.out or os.path.join(paths["models"], "mgenre_ft")
        return (
            f"cd {GENRE_DIR} && CUDA_VISIBLE_DEVICES={args.gpu} {MGENRE_PY} "
            f"{MGENRE_TRAIN_PY} "
            f"--mode finetune --batch_size {args.batch_size} --epochs {args.epochs} "
            f"--entities {paths['entities']} "
            f"--train {paths['train']} --valid {paths['valid']} --test {paths['test']} "
            f"--output_dir {out_model} --trie_path {paths['trie']}"
        )

    # ---- constrained (candidate-list) prediction: wraps predict_mgenre_blink10_split.py ----
    def add_constrained_args(self, p):
        p.add_argument("--kind", choices=["ft"], default="ft")
        p.add_argument("--model", help="override model path")
        p.add_argument("--data", help="override data dir")

    def constrained_out_paths(self, ctx, paths, args):
        base = f"mgenre_{args.kind}_blink10"
        out = args.out or os.path.join(paths["predictions"], f"{base}.jsonl")
        log = os.path.join(paths["logs"], f"{base}.log")
        return out, log

    def constrained_cmd(self, ctx, paths, candidates, out, args):
        model = args.model or os.path.join(paths["models"], "mgenre_ft")
        return [
            MGENRE_PY,
            MGENRE_BLINK10_PY,
            "--model",
            model,
            "--data",
            args.data or paths["data"],
            "--ice",
            candidates,
            "--out",
            out,
        ]


class ReFinED(TrainableNELModel):
    """ReFinED with the original (shared) PEM. Three stages, all here:
      - train   (subprocess into the ReFinED venv)        -> train_refined
      - predict (in-process torch, full pool)             -> refined
      - blink10 (subprocess, BLINK top-10 as PEM)         -> refined_blink10
    The `refined` predict loads the model in this process, so it must run under
    the ReFinED venv interpreter (cmd_all launches REFINED_PY run_all.py refined)."""

    name = "refined"
    train_help = "train ReFinED per-split (original PEM)"
    predict_help = "predict with ReFinED (original PEM)"
    train_batch_size = MODELS["refined"]["train_batch_size"]
    constrained_help = "predict ReFinED with BLINK10 PEM"
    url = "https://github.com/amazon-science/ReFinED"
    paper_title = "ReFinED: An Efficient Zero-shot-capable Approach to End-to-End Entity Linking"
    paper_url = "https://arxiv.org/abs/2207.04108"

    # ---- in-process full-pool prediction (was run_all.cmd_refined) ----
    def add_predict_args(self, p):
        p.add_argument("--out", required=True)
        p.add_argument("--model_pt")
        p.add_argument("--refined_data")
        p.add_argument("--test")
        p.add_argument("--max_candidates", type=int, default=30)

    def predict_inproc(self, ctx, args):
        """ReFinED with original PEM, save predictions."""
        paths = ctx.scheme_paths(args.scheme) if args.scheme else None
        test = args.test or (paths["test"] if paths else None)
        refined_data = args.refined_data or (paths["refined_data"] if paths else None)
        model_pt = args.model_pt or os.path.join(refined_data, "best_model/model.pt")

        os.chdir(REFINED_DIR)
        sys.path.insert(0, REFINED_DIR)
        sys.path.insert(0, os.path.join(REFINED_DIR, "src"))
        import torch
        from refined.model_components.config import ModelConfig
        from refined.model_components.refined_model import RefinedModel
        from refined.utilities.preprocessing_utils import convert_doc_to_tensors

        # Re-use yakusya preprocessor
        from run_yakusya import (
            TRANSFORMER_NAME,
            YakusyaPreprocessor,
            load_yakusya_docs,
        )
        from torch.cuda.amp import autocast
        from tqdm import tqdm

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[refined] device={device} refined_data={refined_data}")
        preprocessor = YakusyaPreprocessor(
            data_dir=refined_data, max_candidates=args.max_candidates
        )
        test_docs = load_yakusya_docs(test, preprocessor)
        print(f"[refined] test docs={len(test_docs)}")

        config = ModelConfig(
            data_dir=refined_data,
            transformer_name=TRANSFORMER_NAME,
            max_candidates=args.max_candidates,
            ner_tag_to_ix=preprocessor.ner_tag_to_ix,
        )
        model = RefinedModel(
            config=config, preprocessor=preprocessor, use_precomputed_descriptions=False
        )
        ckpt = torch.load(model_pt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=False)
        model.to(device).eval()

        out = []
        seen = 0
        with torch.no_grad():
            for doc in tqdm(test_docs, desc="predict"):
                if not doc.spans:
                    continue
                try:
                    tns_iter = convert_doc_to_tensors(
                        doc,
                        preprocessor,
                        collate=True,
                        max_batch_size=16,
                        sort_by_tokens=False,
                        max_seq=preprocessor.max_seq,
                    )
                    for batch in tns_iter:
                        batch = batch.to(device)
                        with autocast():
                            output = model(batch=batch)
                        if output.ed_activations is None or output.cand_ids is None:
                            for s in doc.spans:
                                out.append(
                                    {
                                        "sample_idx": seen,
                                        "gold_id": int(s.gold_entity.wikidata_entity_id[1:]),
                                        "pred_id": -1,
                                        "mention": s.text,
                                    }
                                )
                                seen += 1
                            continue
                        cand_ids = torch.cat(
                            [
                                output.cand_ids,
                                torch.ones(
                                    (output.cand_ids.size(0), 1), device=device, dtype=torch.long
                                )
                                * -1,
                            ],
                            1,
                        )
                        preds = output.ed_activations.argmax(dim=1)
                        pred_qids = cand_ids[torch.arange(cand_ids.size(0)), preds].cpu().tolist()
                        for span, pid in zip(doc.spans, pred_qids):
                            out.append(
                                {
                                    "sample_idx": seen,
                                    "gold_id": int(span.gold_entity.wikidata_entity_id[1:]),
                                    "pred_id": int(pid),
                                    "mention": span.text,
                                }
                            )
                            seen += 1
                except Exception:
                    for s in doc.spans:
                        out.append(
                            {
                                "sample_idx": seen,
                                "gold_id": int(s.gold_entity.wikidata_entity_id[1:]),
                                "pred_id": -1,
                                "mention": s.text,
                            }
                        )
                        seen += 1
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        ctx.write_jsonl(out, args.out)
        print(f"[refined] saved {len(out)} → {args.out}")

    def train_done(self, ctx, paths, args):
        out_data = args.out or paths["refined_data"]
        return os.path.join(out_data, "best_model/model.pt")

    def train_cmd(self, ctx, paths, args):
        out_data = args.out or paths["refined_data"]
        return (
            f"cd {REFINED_DIR} && CUDA_VISIBLE_DEVICES={args.gpu} {REFINED_PY} "
            f"{REFINED_TRAIN_PY} "
            f"--epochs {args.epochs} --batch-size {args.batch_size} "
            f"--yakusya_dir {paths['data']} --output_dir {out_data}"
        )

    # ---- constrained (candidate-list) prediction: wraps predict_yakusya_blink_cands_split.py ----
    def constrained_cmd(self, ctx, paths, candidates, out, args):
        return [
            REFINED_PY,
            REFINED_BLINK10_PY,
            "--refined_data",
            paths["refined_data"],
            "--data",
            paths["data"],
            "--ice",
            candidates,
            "--out",
            out,
        ]

    # ---- train_refined_pem: rebuild per-scheme PEM from train, then train + predict ----
    @staticmethod
    def _normalize_surface_form_ja(s):
        return (s or "").lower().strip().replace('"', "").replace("'", "")

    def _build_pem_from_train(self, train_jsonl):
        """Build mention->[(qcode, prob)] PEM from a train.jsonl."""
        surface_to_entities = defaultdict(lambda: defaultdict(int))
        with open(train_jsonl) as f:
            for line in f:
                m = json.loads(line)
                surface_to_entities[self._normalize_surface_form_ja(m["mention"])][
                    f"Q{m['label_id']}"
                ] += 1
        pem = {}
        for surface, ent_counts in surface_to_entities.items():
            total = sum(ent_counts.values())
            pem[surface] = [
                (q, c / total) for q, c in sorted(ent_counts.items(), key=lambda x: -x[1])
            ]
        return pem

    def _write_pem_lmdb(self, ctx, pem_dict, out_lmdb_path, log_path=None):
        """Serialize PEM dict to lmdb via REFINED_PY (which has ReFinED's lmdb_wrapper)."""
        helper = out_lmdb_path + "._writer.py"
        with open(helper, "w") as f:
            f.write(
                "import sys, json, os, shutil\n"
                f'sys.path.insert(0, "{config.WS}/ReFinED/src")\n'
                "from refined.resource_management.lmdb_wrapper import LmdbImmutableDict\n"
                "in_json, out_lmdb = sys.argv[1], sys.argv[2]\n"
                "if os.path.exists(out_lmdb):\n"
                "    shutil.rmtree(out_lmdb) if os.path.isdir(out_lmdb) else os.remove(out_lmdb)\n"
                "with open(in_json) as f: pem = json.load(f)\n"
                "LmdbImmutableDict.from_dict(pem, out_lmdb)\n"
                'print(f"Wrote {out_lmdb} with {len(pem)} keys")\n'
            )
        pem_json = out_lmdb_path + ".tmp.json"
        with open(pem_json, "w") as f:
            json.dump(pem_dict, f, ensure_ascii=False)
        rc = ctx.run([REFINED_PY, helper, pem_json, out_lmdb_path], log_path=log_path)
        os.remove(pem_json)
        os.remove(helper)
        if rc != 0:
            raise RuntimeError(f"PEM lmdb write failed (see {log_path})")

    def _setup_refined_data_with_pem(self, ctx, scheme, log_path=None):
        """Build experiments/<scheme>/refined_data/ with symlinks to ORIG_REFINED_DATA
        minus pem.lmdb, plus a fresh per-scheme pem.lmdb. Returns (dir, pem_size)."""
        import shutil

        paths = ctx.scheme_paths(scheme)
        base = paths["refined_data"]
        orig_refined_data = os.path.join(REFINED_DIR, "yakusya_data")
        os.makedirs(base, exist_ok=True)
        # Benchmarks build their OWN per-dataset KB into `base` (via
        # prepare_yakusya_data.py) before this runs. Overlaying the daime
        # 5001-entity yakusya_data on top would clobber that KB and leave two
        # qcode_to_class_tns_*.np files (the preprocessor asserts exactly one).
        # Only overlay from the daime KB when `base` has no self-built KB yet.
        base_idx = os.path.join(base, "qcode_to_idx.lmdb")
        self_built = os.path.exists(base_idx) and not os.path.islink(base_idx)
        if not self_built:
            for fn in os.listdir(orig_refined_data):
                if fn == "pem.lmdb":
                    continue
                src = os.path.join(orig_refined_data, fn)
                dst = os.path.join(base, fn)
                if os.path.lexists(dst):
                    if os.path.islink(dst) or os.path.isfile(dst):
                        os.remove(dst)
                    else:
                        shutil.rmtree(dst)
                os.symlink(src, dst)
        pem_dict = self._build_pem_from_train(paths["train"])
        pem_lmdb = os.path.join(base, "pem.lmdb")
        if os.path.exists(pem_lmdb):
            if os.path.isdir(pem_lmdb):
                shutil.rmtree(pem_lmdb)
            else:
                os.remove(pem_lmdb)
        self._write_pem_lmdb(ctx, pem_dict, pem_lmdb, log_path=log_path)
        return base, len(pem_dict)

    def register_train_pem(self, sub, ctx):
        p = sub.add_parser(
            "train_refined_pem", help="rebuild PEM from per-scheme train + train ReFinED + predict"
        )
        ctx.add_scheme(p, required=True)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--batch_size", type=int, default=1)
        p.add_argument("--out", help="default: <scheme>/models/refined_pem")
        p.add_argument("--pred_out", help="default: <scheme>/predictions/refined_pem.jsonl")
        p.add_argument("--skip_existing", action="store_true")
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.train_pem(_c, a))
        return p

    def train_pem(self, ctx, args):
        """Per-scheme ReFinED with REBUILT PEM (vs train_refined's original PEM and
        refined_blink10's BLINK10 PEM). Rebuild PEM from train -> set up refined_data
        -> train ReFinED -> predict (the two stages run run_yakusya with DATA_DIR
        monkey-patched via a generated wrapper)."""
        paths = ctx.scheme_paths(args.scheme)
        log = os.path.join(paths["logs"], "refined_pem.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        out_dir = args.out or os.path.join(paths["models"], "refined_pem")
        pred = args.pred_out or os.path.join(paths["predictions"], "refined_pem.jsonl")
        if args.skip_existing and os.path.exists(pred):
            print(f"[train_refined_pem] preds exist, skip: {pred}")
            return

        # 1+2) build PEM and set up refined_data
        print(f"[train_refined_pem/{args.scheme}] rebuilding PEM from train...")
        refined_data, pem_size = self._setup_refined_data_with_pem(ctx, args.scheme, log_path=log)
        print(f"  PEM size: {pem_size} mentions, refined_data: {refined_data}")

        # 3) train (run_yakusya.py uses module-level DATA_DIR for some paths, monkey-patch via wrapper)
        if not (
            args.skip_existing and os.path.exists(os.path.join(out_dir, "best_model/model.pt"))
        ):
            wrapper = os.path.join(paths["logs"], "_train_pem_wrapper.py")
            with open(wrapper, "w") as f:
                f.write(
                    f"import sys\n"
                    f'sys.path.insert(0, "{REFINED_DIR}")\n'
                    f'sys.path.insert(0, "{os.path.join(REFINED_DIR, "src")}")\n'
                    f"import run_yakusya\n"
                    f"run_yakusya.DATA_DIR = {refined_data!r}\n"
                    f'sys.argv = ["run_yakusya.py", "--epochs", "{args.epochs}", '
                    f'"--batch-size", "{args.batch_size}", '
                    f'"--yakusya_dir", "{paths["data"]}", '
                    f'"--output_dir", "{out_dir}"]\n'
                    f"run_yakusya.main()\n"
                )
            cmd = f"cd {REFINED_DIR} && CUDA_VISIBLE_DEVICES={args.gpu} {REFINED_PY} {wrapper}"
            if ctx.run(cmd, log_path=log) != 0:
                print(f"[train_refined_pem] train FAILED (see {log})")
                return

        # 4) predict (also needs DATA_DIR override -> wrapper)
        if not os.path.exists(pred):
            wrapper = os.path.join(paths["logs"], "_predict_pem_wrapper.py")
            model_pt = os.path.join(out_dir, "best_model/model.pt")
            with open(wrapper, "w") as f:
                f.write(
                    f"import sys\n"
                    f'sys.path.insert(0, "{REFINED_DIR}")\n'
                    f'sys.path.insert(0, "{os.path.join(REFINED_DIR, "src")}")\n'
                    f'sys.path.insert(0, "{BLINK_DIR}")\n'
                    # _SRC_DIR last -> index 0, so `import run_all`/`nel_models`/`config`
                    # resolve to THIS project, not the like-named modules in BLINK_DIR.
                    f'sys.path.insert(0, "{_SRC_DIR}")\n'
                    f"import run_yakusya\n"
                    f"run_yakusya.DATA_DIR = {refined_data!r}\n"
                    f'sys.argv = ["run_all.py", "refined", '
                    f'"--out", {pred!r}, "--model_pt", {model_pt!r}, '
                    f'"--refined_data", {refined_data!r}, '
                    f'"--test", {paths["test"]!r}]\n'
                    f"import run_all; run_all.main()\n"
                )
            cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} {REFINED_PY} {wrapper}"
            if ctx.run(cmd, log_path=log) != 0:
                print(f"[train_refined_pem] predict FAILED (see {log})")
                return
        print(f"[train_refined_pem] done; preds → {pred}")


class DyVo(TrainableNELModel):
    """DyVo learned-sparse retrieval. Two stages here:
    train_dyvo     = prep (DyVo venv) + lsr.train (subprocess) + collect trec +
                     trec->JSONL adapt (a multi-step pipeline; train() is overridden)
    dyvo_blink10   = trec ranking ∩ BLINK top-10 (pure-Python in-process adapter)."""

    name = "dyvo"
    train_help = "train + run DyVo per-split"
    constrained_help = "DyVo trec → BLINK10-restricted predictions"
    constrained_inproc = True
    url = "https://github.com/thongnt99/DyVo"
    paper_title = "DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities"

    # ---- train pipeline (in-process orchestration; overrides train/register_train) ----
    def register_train(self, sub, ctx):
        p = sub.add_parser("train_dyvo", help=self.train_help)
        ctx.add_scheme(p, required=True)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--max_steps", type=int, default=10000)
        p.add_argument("--out")
        p.add_argument("--pred_out")
        p.add_argument("--skip_existing", action="store_true")
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.train(_c, a))
        return p

    def train(self, ctx, args):
        """Train DyVo per-split: prep + lsr.train (subprocess) + collect trec + adapt."""
        import glob
        import shutil

        paths = ctx.scheme_paths(args.scheme)
        dyvo_data = paths["dyvo_data"]
        out_dir = args.out or paths["dyvo_outputs"]
        pred = args.pred_out or os.path.join(paths["predictions"], "dyvo.jsonl")
        if args.skip_existing and os.path.exists(pred):
            print(f"[train_dyvo] preds exist, skip: {pred}")
            return
        log = os.path.join(paths["logs"], "dyvo.log")
        os.makedirs(dyvo_data, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # 1) prepare DyVo format
        cmd1 = [DYVO_PY, DYVO_PREP_PY, "--src", paths["data"], "--out", dyvo_data]
        if ctx.run(cmd1, log_path=log) != 0:
            print(f"[train_dyvo] prepare failed, see {log}")
            return

        # 2) train + test
        dyvo_out_internal = os.path.join(DYVO_DIR, f"outputs/yakusya_{args.scheme}")
        os.makedirs(dyvo_out_internal, exist_ok=True)
        cmd2 = (
            f"cd {DYVO_DIR} && CUDA_VISIBLE_DEVICES={args.gpu} "
            f"DYVO_OUTPUT_DIR={dyvo_out_internal} {DYVO_PY} -m lsr.train "
            f"+experiment=yakusya "
            f"++train_dataset.data_path={dyvo_data}/triplets_train.txt "
            f"++eval_dataset.queries_path={dyvo_data}/queries_valid.tsv "
            f"++eval_dataset.docs_path={dyvo_data}/docs.tsv "
            f"++eval_dataset.qrels_path={dyvo_data}/qrels_valid.json "
            f"++test_dataset.queries_path={dyvo_data}/queries_test.tsv "
            f"++test_dataset.docs_path={dyvo_data}/docs.tsv "
            f"++test_dataset.qrels_path={dyvo_data}/qrels_test.json "
            f"++training_arguments.max_steps={args.max_steps} "
            f"++training_arguments.save_steps={args.max_steps}"
        )
        if ctx.run(cmd2, log_path=log) != 0:
            print(f"[train_dyvo] train failed, see {log}")
            return

        # 3) find generated trec, copy to per-scheme dir
        trecs = sorted(
            glob.glob(os.path.join(dyvo_out_internal, "**/test_run.trec"), recursive=True),
            key=os.path.getmtime,
            reverse=True,
        )
        if not trecs:
            print(f"[train_dyvo] no test_run.trec found in {dyvo_out_internal}")
            return
        shutil.copy(trecs[0], os.path.join(out_dir, "test_run.trec"))
        print(f"[train_dyvo] copied trec → {out_dir}/test_run.trec")

        # 4) convert to unified JSONL
        self._adapt_trec(
            ctx, os.path.join(out_dir, "test_run.trec"), paths["entities"], paths["test"], pred
        )

    def _adapt_trec(self, ctx, trec, entities_path, test_path, out_path):
        ents = load_jsonl(entities_path)
        line_to_numid = [int(e["numeric_id"]) for e in ents]
        qid_to_did = {}
        with open(trec) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                qid = int(parts[0])
                did = int(parts[2])
                if qid not in qid_to_did:
                    qid_to_did[qid] = did
        test = load_jsonl(test_path)
        out = []
        for idx, rec in enumerate(test):
            did = qid_to_did.get(idx, -1)
            if 0 <= did < len(ents):
                pred_id = line_to_numid[did]
                pred_title = ents[did].get("title", "")
            else:
                pred_id, pred_title = -1, ""
            out.append(
                {
                    "sample_idx": idx,
                    "gold_id": rec.get("label_id"),
                    "gold_title": rec.get("label_title", ""),
                    "pred_id": pred_id,
                    "pred_title": pred_title,
                    "mention": rec.get("mention"),
                    "type": rec.get("type"),
                }
            )
        ctx.write_jsonl(out, out_path)
        correct = sum(1 for r in out if r["pred_id"] == r["gold_id"])
        print(f"[adapt_dyvo] {correct}/{len(out)} = {correct / len(out) * 100:.2f}% → {out_path}")

    # ---- dyvo_blink10: trec ∩ BLINK top-10 (in-process) ----
    def add_constrained_args(self, p):
        p.add_argument("--trec")
        p.add_argument("--chatel")

    def constrained_predict_inproc(self, ctx, paths, out, args):
        """DyVo trec rerank intersected with BLINK top-10. Pure adapter, no GPU."""
        trec = args.trec or os.path.join(paths["dyvo_outputs"], "test_run.trec")
        chatel = args.chatel or paths["chatel"]
        if not os.path.exists(trec):
            print(f"[dyvo_blink10] missing trec: {trec}")
            return
        if not os.path.exists(chatel):
            print(f"[dyvo_blink10] missing chatel: {chatel}")
            return

        ents = load_jsonl(paths["entities"])
        test = load_jsonl(paths["test"])
        chatel_data = json.load(open(chatel))
        title_to_numid = {e["title"]: int(e["numeric_id"]) for e in ents}

        qid_to_ranks = {}
        with open(trec) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                qid = int(parts[0])
                did = int(parts[2])
                qid_to_ranks.setdefault(qid, []).append(did)

        out_records = []
        for idx, rec in enumerate(test):
            key = f"yakusya_test_{idx}"
            cand_titles = (
                chatel_data.get(key, {}).get("entities", {}).get("entity_candidates", [[]])[0][:10]
            )
            cand_set = set(cand_titles)
            pred_id = -1
            pred_title = ""
            for did in qid_to_ranks.get(idx, []):
                if 0 <= did < len(ents) and ents[did]["title"] in cand_set:
                    pred_id = int(ents[did]["numeric_id"])
                    pred_title = ents[did]["title"]
                    break
            if pred_id == -1 and cand_titles:
                pred_title = cand_titles[0]
                pred_id = title_to_numid.get(pred_title, -1)
            out_records.append(
                {
                    "sample_idx": idx,
                    "gold_id": rec.get("label_id"),
                    "gold_title": rec.get("label_title", ""),
                    "pred_id": pred_id,
                    "pred_title": pred_title,
                    "mention": rec.get("mention"),
                    "type": rec.get("type"),
                }
            )
        ctx.write_jsonl(out_records, out)
        correct = sum(1 for r in out_records if r["pred_id"] == r["gold_id"])
        print(
            f"[dyvo_blink10] {correct}/{len(out_records)} = "
            f"{correct / len(out_records) * 100:.2f}%  → {out}"
        )


def _t5ja_make_input(rec):
    l = rec.get("context_left", "")[-100:]
    r = rec.get("context_right", "")[:100]
    return f"{l} [START_ENT] {rec['mention']} [END_ENT] {r}".strip()


def _t5ja_make_target(rec):
    title = rec.get("label_title") or rec.get("label") or ""
    return f"[START_ENT] {rec['mention']} [END_ENT] [{title}]"


class T5ja(TrainableNELModel):
    """T5-Japanese (sonoisa/t5-base-japanese) seq2seq EL, mGENRE-style: the
    decoder generates the entity title (trie-free here, picked from generation).
    Adapted from dice-group/AugmentedEL (EL-Context_Augmentation) but without the
    LLM context-augmentation step. Fine-tune + full-pool predict run in-process
    (torch) as an override of train()."""

    name = "t5ja_ft"
    train_help = "T5-Japanese seq2seq EL (mGENRE-style) per-split"
    train_batch_size = 8
    url = "https://github.com/dice-group/AugmentedEL"
    paper_title = "Context-Augmented Entity Linking (EL-Context_Augmentation)"

    # ---- train + predict (in-process; overrides TrainableNELModel.train) ----
    def add_train_args(self, p):
        p.add_argument("--lr", type=float, default=5e-4)
        p.add_argument("--max_in", type=int, default=256)
        p.add_argument("--max_out", type=int, default=64)
        p.add_argument("--num_beams", type=int, default=5)
        p.add_argument(
            "--base_model",
            default="sonoisa/t5-base-japanese",
            help="HF base model (default: sonoisa/t5-base-japanese)",
        )
        p.add_argument("--pred_out", help="default: <scheme>/predictions/t5ja_ft.jsonl")

    def train(self, ctx, args):
        """Train + predict T5-Japanese seq2seq EL on yakusha (per-scheme)."""
        paths = ctx.scheme_paths(args.scheme)
        out_dir = args.out or os.path.join(paths["models"], "t5ja_ft")
        pred = args.pred_out or os.path.join(paths["predictions"], "t5ja_ft.jsonl")
        log = os.path.join(paths["logs"], "t5ja_ft.log")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if args.skip_existing and os.path.exists(pred):
            print(f"[t5ja_ft] preds exist, skip: {pred}")
            return

        import re

        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm
        from transformers import (
            AutoTokenizer,
            T5ForConditionalGeneration,
            get_linear_schedule_with_warmup,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = args.base_model or "sonoisa/t5-base-japanese"
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        train = load_jsonl(paths["train"])
        test = load_jsonl(paths["test"])
        entities = load_jsonl(paths["entities"])
        title_to_numids = defaultdict(list)
        for e in entities:
            title_to_numids[normalize_title(e["title"])].append(int(e["numeric_id"]))

        has_ckpt = os.path.exists(os.path.join(out_dir, "pytorch_model.bin")) or os.path.exists(
            os.path.join(out_dir, "model.safetensors")
        )
        if args.skip_existing and has_ckpt:
            print(f"[t5ja_ft] loading existing checkpoint from {out_dir}")
            model = T5ForConditionalGeneration.from_pretrained(out_dir).to(device)
        else:
            print(
                f"[t5ja_ft] training T5-Japanese from {base_model} "
                f"({len(train)} samples × {args.epochs} epochs)"
            )
            model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)

            class _DS(Dataset):
                def __init__(self, recs):
                    self.recs = recs

                def __len__(self):
                    return len(self.recs)

                def __getitem__(self, i):
                    r = self.recs[i]
                    src = _t5ja_make_input(r)
                    tgt = _t5ja_make_target(r)
                    enc = tokenizer(
                        src,
                        max_length=args.max_in,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    tgt_enc = tokenizer(
                        tgt,
                        max_length=args.max_out,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    labels = tgt_enc["input_ids"].squeeze(0)
                    labels[labels == tokenizer.pad_token_id] = -100
                    return {
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "labels": labels,
                    }

            loader = DataLoader(_DS(train), batch_size=args.batch_size, shuffle=True, num_workers=2)
            opt = AdamW(model.parameters(), lr=args.lr)
            total_steps = len(loader) * args.epochs
            sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
            model.train()
            for ep in range(args.epochs):
                losses = []
                pbar = tqdm(loader, desc=f"epoch {ep + 1}")
                for batch in pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    out.loss.backward()
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    losses.append(out.loss.item())
                    if len(losses) % 50 == 0:
                        pbar.set_postfix(loss=f"{sum(losses[-50:]) / min(50, len(losses)):.3f}")
                print(f"  epoch {ep + 1} avg loss = {sum(losses) / len(losses):.4f}")
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"[t5ja_ft] saved → {out_dir}")

        model.eval()
        out_records = []
        print(f"[t5ja_ft] predicting on {len(test)} test samples")
        pat = re.compile(r"\[([^\[\]]+)\]\s*$")
        with torch.no_grad():
            for i in tqdm(range(0, len(test), args.batch_size)):
                batch = test[i : i + args.batch_size]
                srcs = [_t5ja_make_input(r) for r in batch]
                enc = tokenizer(
                    srcs, max_length=args.max_in, truncation=True, padding=True, return_tensors="pt"
                ).to(device)
                gen = model.generate(
                    **enc, max_length=args.max_out, num_beams=args.num_beams, num_return_sequences=1
                )
                outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for j, (rec, txt) in enumerate(zip(batch, outs)):
                    m = pat.search(txt)
                    pred_title = m.group(1).strip() if m else ""
                    cand_ids = title_to_numids.get(normalize_title(pred_title), [])
                    gold_id = rec.get("label_id")
                    if gold_id in cand_ids:
                        pred_id = gold_id
                    elif cand_ids:
                        pred_id = cand_ids[0]
                    else:
                        pred_id = -1
                    out_records.append(
                        {
                            "sample_idx": i + j,
                            "gold_id": gold_id,
                            "gold_title": rec.get("label_title", ""),
                            "pred_id": pred_id,
                            "pred_title": pred_title,
                            "mention": rec.get("mention"),
                            "type": rec.get("type"),
                        }
                    )

        ctx.write_jsonl(out_records, pred)
        correct = sum(1 for r in out_records if r["pred_id"] == r["gold_id"])
        print(
            f"[t5ja_ft] {correct}/{len(out_records)} = "
            f"{correct / len(out_records) * 100:.2f}% → {pred}"
        )


class LLMAEL(TrainableNELModel):
    """LLMAEL: LLM-augmented context (context_right expanded with LLM-generated
    text) -> train a separate BLINK bi-encoder on it, predict via the `blink`
    subcommand. Augmented data is pre-generated (LLMAEL_DATA_DIR); train()
    is an in-process orchestration. Upstream: THU-KEG/LLMAEL (CIKM 2025)."""

    name = "llmael"
    train_help = "split LLMAEL data + train BLINK on it + predict (per-scheme)"
    url = "https://github.com/THU-KEG/LLMAEL"
    paper_title = "LLMAEL: Large Language Models are Good Context Augmenters for Entity Linking"
    paper_url = "https://arxiv.org/abs/2407.04020"

    def register_train(self, sub, ctx):
        p = sub.add_parser(
            "train_blink_llmael",
            help="split LLMAEL data + train BLINK on it + predict (per-scheme)",
        )
        ctx.add_scheme(p, required=True)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--out", help="default: <scheme>/models/blink_llmael")
        p.add_argument("--pred_out", help="default: <scheme>/predictions/blink_llmael.jsonl")
        p.add_argument("--skip_existing", action="store_true")
        p.set_defaults(func=lambda a, _m=self, _c=ctx: _m.train(_c, a))
        return p

    def train(self, ctx, args):
        """Per-scheme: split LLMAEL data + train BLINK on it + predict. LLMAEL is
        line-aligned with original (same label_id/mention) but context_right is
        augmented with LLM-generated text; trains a separate BLINK on it."""
        paths = ctx.scheme_paths(args.scheme)
        llmael_dir = (
            LLMAEL_DATA_DIR
            if args.scheme == "original"
            else os.path.join(EXPERIMENTS, args.scheme, "llmael_data")
        )
        model_path = args.out or (
            os.path.join(PROJ_EXP, "models_ja/yakusya_theater_model_llmael")
            if args.scheme == "original"
            else os.path.join(paths["models"], "blink_llmael")
        )
        pred = args.pred_out or os.path.join(paths["predictions"], "blink_llmael.jsonl")
        log = os.path.join(paths["logs"], "blink_llmael.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if args.skip_existing and os.path.exists(pred):
            print(f"[train_blink_llmael] preds exist, skip: {pred}")
            return

        # 1) LLMAEL split (only for non-original)
        if args.scheme != "original" and not os.path.exists(os.path.join(llmael_dir, "test.jsonl")):
            if args.scheme not in NEL_dataset.SPLITS:
                print(f"[train_blink_llmael] no split fn for {args.scheme}")
                return
            tr, va, te = NEL_dataset(LLMAEL_DATA_DIR, load_entities=False).split(args.scheme)
            ctx.write_jsonl(tr, os.path.join(llmael_dir, "train.jsonl"))
            ctx.write_jsonl(va, os.path.join(llmael_dir, "valid.jsonl"))
            ctx.write_jsonl(te, os.path.join(llmael_dir, "test.jsonl"))
            ent_link = os.path.join(llmael_dir, "entities.jsonl")
            if os.path.lexists(ent_link):
                os.remove(ent_link)
            os.symlink(os.path.join(ORIG_DATA, "entities.jsonl"), ent_link)
            print(
                f"[train_blink_llmael/{args.scheme}] split: train={len(tr)} valid={len(va)} test={len(te)}"
            )

        # 2) train BLINK on LLMAEL data
        if not (
            args.skip_existing and os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
        ):
            cmd = (
                f"cd {BLINK_DIR} && CUDA_VISIBLE_DEVICES={args.gpu} {BLINK_PY} "
                f"-m blink.biencoder.train_biencoder "
                f"--data_path {llmael_dir} --output_path {model_path} "
                f"--bert_model bert-base-multilingual-cased "
                f"--max_context_length 128 --max_cand_length 128 --max_seq_length 256 "
                f"--num_train_epochs {args.epochs} --train_batch_size {args.batch_size} "
                f"--learning_rate 2e-5 --eval_batch_size 8 --type_optimization all_encoder_layers "
                f"--save_interval 1 --print_interval 50 --eval_interval 500 "
                f"--shuffle True --zeshel True"
            )
            if (
                ctx.run(
                    cmd,
                    log_path=log,
                    env={"PYTHONPATH": BLINK_DIR, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
                )
                != 0
            ):
                print(f"[train_blink_llmael] train FAILED (see {log})")
                return

        # 3) predict using LLMAEL test
        if not os.path.exists(pred):
            cmd = [
                BLINK_PY,
                ctx.run_all_py,
                "blink",
                "--data_path",
                llmael_dir,
                "--model_path",
                model_path,
                "--bert_model",
                "bert-base-multilingual-cased",
                "--out",
                pred,
            ]
            if ctx.run(cmd, log_path=log, env={"CUDA_VISIBLE_DEVICES": str(args.gpu)}) != 0:
                print(f"[train_blink_llmael] predict FAILED (see {log})")
                return
        print(f"[train_blink_llmael] done; preds → {pred}")


# Singletons reused across the registration helpers (train / constrained / predict).
_BLINK, _MGENRE, _REFINED, _DYVO, _T5JA, _LLMAEL = (
    Blink(),
    MGenre(),
    ReFinED(),
    DyVo(),
    T5ja(),
    LLMAEL(),
)

RERANKERS = [Incel(), DeepEL(), SumMC()]
SUBPROCESS_METHODS = [
    OneNet(),
    Llmd(),
    MAD(),
    TransMAD(),
    Reflexion(),
    MapsEL(),
    Rerank(),
    ZeroShotCoT(),
    SelfConsistency(),
    PlanSolve(),
    CoVe(),
    LLMTool(),
]
TRAINABLE_METHODS = [
    _BLINK,
    _MGENRE,
    _REFINED,
    _T5JA,
    _DYVO,
    _LLMAEL,
]  # _T5JA/_DYVO/_LLMAEL.train() are overrides
CONSTRAINED_METHODS = [_MGENRE, _REFINED, _DYVO]  # candidate-list (<name>_blink10) predictors
INPROC_PREDICT_METHODS = [_BLINK, _MGENRE, _REFINED]  # in-process torch full-pool predict (<name>)
ALL_METHODS = RERANKERS + SUBPROCESS_METHODS


def register_trainers(sub, ctx):
    """Register the train_<name> subcommands for trainable methods, plus the bespoke
    extra subcommand (train_refined_pem)."""
    for m in TRAINABLE_METHODS:
        m.register_train(sub, ctx)
    _REFINED.register_train_pem(sub, ctx)  # train_refined_pem (rebuilt-PEM variant)


def register_constrained_methods(sub, ctx):
    """Register the <name>_blink10 candidate-list predictors (mgenre/refined)."""
    for m in CONSTRAINED_METHODS:
        m.register_constrained(sub, ctx)


def register_inproc_predictors(sub, ctx):
    """Register the in-process full-pool predict subcommands (blink/mgenre/refined)
    plus Blink's gen_blink_cands candidate generator."""
    for m in INPROC_PREDICT_METHODS:
        m.register_predict(sub, ctx)
    _BLINK.register_gen_cands(sub, ctx)


def register_methods(sub, ctx):
    """Register every enabled NELModel-based method as a run_all subcommand.
    Enablement is controlled centrally by ``config.METHOD_ENABLED`` (single source of
    truth): a method whose switch is False is not registered, so it neither shows in
    ``--help`` nor runs, while its code stays in place. Flip the switch to bring it
    back — no code change needed."""
    for m in ALL_METHODS:
        if not config.method_enabled(m.name):
            continue
        m.register(sub, ctx)
