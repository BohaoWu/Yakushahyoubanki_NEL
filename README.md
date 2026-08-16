# Yakushahyoubanki NEL

**An Evidence-Extraction Agent (EEA) for generation disambiguation and historical
entity linking on the early modern Japanese _Yakusha Hyōbanki_ (歌舞伎役者評判記).**

Given a kabuki-actor mention, the task is to link it to the **correct generation**
(代目) in the knowledge base. The EEA automatically extracts the deciding evidence
(a mention year plus the actor's name-change table) and hands it to a swappable
reasoning module. Methods and experiments are described in the accompanying paper.

![python](https://img.shields.io/badge/python-3.10-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements.txt
cp .env.sample .env                                    # then set OPENAI_API_KEY in .env
```

`run_all.py` loads `.env` on startup (see [`.env.sample`](.env.sample) for the keys).
The EEA and the LLM disambiguators (`mad`/`maps`/`reflexion`/`cove`/`selfcon`/`zscot`/
`plansolve`/`rerank`) run with only the above plus an API key and the dataset. The
trained and reranker baselines (`blink`/`mgenre`/`refined`/`dyvo`/`onenet`, and the
`In_Context_EL`/`DeepEL`/`SumMC` rerankers) each require their own upstream repository,
pretrained weights, and a GPU; toggle them in `src/config.py` (`METHOD_ENABLED`).

```bash
./run.sh --help
```

## Usage

```bash
./run.sh <subcommand> --scheme original ...   # single launcher; default dataset daime_eval_full
./run.sh --help                               # list all subcommands
```

Switch the evaluation dataset with `YAKUSYA_DATASET`:

```bash
export YAKUSYA_DATASET=daime_eval        # frozen 1479-mention subset (published numbers)
export YAKUSYA_DATASET=relinked_yr_v1    # training / methods that need a train split
export YAKUSYA_DATASET=hipe2020          # a public HIPE-2022 benchmark (see Datasets)
```

`daime_eval_full` is the 2214-mention 代目 evaluation set (eval-only, `test.jsonl`).
For trained models (mGENRE/BLINK) only rows marked `heldout=true` are leak-free; see
the dataset-variant notes in [`src/config.py`](src/config.py).

## Datasets

- **Yakusha Hyōbanki** (default: `daime_eval_full` / `daime_eval`) — **not redistributed
  in this repository.** The knowledge-base records are drawn from the digital portal
  databases of the **Art Research Center (ARC), Ritsumeikan University**
  (<https://www.dh-jac.net/db/shumei>). **Access requires a separate application** to
  the ARC / the authors; the code here expects the dataset to be provided under
  `data/`.
- **HIPE-2022** (AJMC, HIPE-2020, NewsEye, SoNAR, TopRes19th) — public, from the CLEF
  HIPE-2022 shared task: <https://github.com/hipe-eval/HIPE-2022-data>. Fetch and
  convert them with `python src/build_dataset.py hipe`.
- **Mahānāma** (Sanskrit entity discovery & linking over the _Mahābhārata_) — Sarkar
  et al., EMNLP 2025: <https://arxiv.org/abs/2509.19844>.

## Citation

If this code or data helps your research, please cite the accompanying paper (BibTeX to
follow on publication).

## License

Code is released under the [MIT](LICENSE) license. Dataset licenses are held by their
respective sources (see **Datasets**); the Yakusha data in particular is not covered by
this license and must be obtained separately.
