"""Single entry point for LLM chat requests.

Consolidates what every disambiguator used to repeat inline: build an OpenAI client
(base_url / api_key from CLI args, else the environment), wrap it with the on-disk
response cache (llm_cache.wrap_client), and issue a chat completion with retries.

    client = make_client(base_url=args.base_url, api_key=args.api_key)
    text = ask(client, model, messages)

`ask` is also re-exported as `_ask` from llm_debate_disambiguator for the reasoning
methods that import it there.
"""

import os
import time

from llm_cache import wrap_client
from openai import OpenAI


def make_client(base_url=None, api_key=None, cache=True):
    """Build an OpenAI client, optionally wrapped with the on-disk response cache.

    base_url / api_key fall back to the OPENAI_BASE_URL / OPENAI_API_KEY environment
    variables. A base_url with no key anywhere gets a dummy key, since a local server
    usually still requires one to be set.
    """
    kwargs = {}
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        kwargs["api_key"] = key
    elif base_url:
        kwargs["api_key"] = "EMPTY"  # local server usually needs a dummy key
    client = OpenAI(**kwargs)
    return wrap_client(client) if cache else client


def ask(client, model, messages, temperature=0, max_retries=6):
    """One chat completion with exponential backoff on transient errors; returns the
    message content (empty string if the model returned none)."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)
