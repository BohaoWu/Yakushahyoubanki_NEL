"""Disk cache for OpenAI chat.completions, shared by the LLM disambiguator suite.

`wrap_client(client)` transparently caches `client.chat.completions.create` keyed
by the full request (model + messages + sampling params). Identical requests are
given sequential *occurrence* slots, so temperature>0 sampling keeps its multiset
of distinct samples (selfcon / rerank) and still reuses them verbatim on a re-run
— a crash, an added method, or a metric re-compute never re-hits the API for work
already done.

Env:
  LLM_CACHE=0        disable caching entirely
  LLM_CACHE_DIR=...  cache location (default: <repo>/experiment/llm_cache)
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
import hashlib
import json
import os
import threading

_LOCK = threading.Lock()
_OCC = {}  # base_key -> next occurrence index (per process)

_KEY_FIELDS = (
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "n",
    "tools",
    "tool_choice",
    "response_format",
    "stop",
    "seed",
)


def _cache_dir(explicit=None):
    d = (
        explicit
        or os.environ.get("LLM_CACHE_DIR")
        or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "experiment",
            "llm_cache",
        )
    )
    os.makedirs(d, exist_ok=True)
    return d


def _base_key(kwargs):
    keep = {k: kwargs[k] for k in _KEY_FIELDS if k in kwargs}
    blob = json.dumps(keep, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def wrap_client(client, cache_dir=None):
    """Monkey-patch client.chat.completions.create with a disk cache. Idempotent
    and a no-op when LLM_CACHE=0. Returns the same client."""
    if os.environ.get("LLM_CACHE", "1") == "0":
        return client
    if getattr(client.chat.completions.create, "_llm_cached", False):
        return client
    cache_dir = _cache_dir(cache_dir)
    orig = client.chat.completions.create
    from openai.types.chat import ChatCompletion

    def cached_create(**kwargs):
        if kwargs.get("stream"):  # never cache streaming
            return orig(**kwargs)
        base = _base_key(kwargs)
        with _LOCK:
            occ = _OCC.get(base, 0)
            _OCC[base] = occ + 1
        path = os.path.join(cache_dir, f"{base}-{occ}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return ChatCompletion.model_validate(json.load(f))
            except Exception:
                pass  # corrupt entry -> refetch
        resp = orig(**kwargs)
        try:
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(resp.model_dump(mode="json"), f, ensure_ascii=False)
            os.replace(tmp, path)  # atomic
        except Exception:
            pass
        return resp

    cached_create._llm_cached = True
    client.chat.completions.create = cached_create
    return client
