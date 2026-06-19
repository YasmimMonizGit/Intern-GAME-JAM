"""
gemma.py — turn a task list into a natural instruction for the "intern".

Primary path: call a local Ollama model (default gemma4:e4b-it-qat).
Fallback path: a simple templated phrasing, used automatically if Ollama is
unreachable, so the pipeline always runs (and is testable without a GPU).

The instruction always describes the jobs IN THE GIVEN ORDER, so the order the
network must learn lives in the text, not in any task-list input.
"""

import json
import random
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b-it-qat"

_PROMPT = """You are a person casually telling a new kitchen intern what to do.
In ONE or TWO natural sentences, tell the intern to carry out the jobs below IN \
THIS EXACT ORDER.
Rules: keep the exact order, mention every job once, do NOT number them, do NOT \
invent new jobs, do NOT add commentary. Just the instruction.

Jobs in order:
{jobs}

Instruction:"""


def gemma_instruction(task_texts, model=DEFAULT_MODEL, host=OLLAMA_HOST,
                      timeout=60, temperature=0.8):
    """Ask Ollama for a natural instruction. Raises on connection failure."""
    jobs = "\n".join(f"{i+1}. {t}" for i, t in enumerate(task_texts))
    body = json.dumps({
        "model": model,
        "prompt": _PROMPT.format(jobs=jobs),
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(host + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    text = data.get("response", "").strip()
    return " ".join(text.split())            # collapse whitespace/newlines


def ollama_available(model=DEFAULT_MODEL, host=OLLAMA_HOST):
    try:
        gemma_instruction(["Wipe the counter"], model=model, host=host, timeout=20)
        return True
    except Exception:
        return False


# ---- templated fallback (no LLM needed) ----------------------------------- #
_OPENERS = ["Hey, could you", "When you get a sec,", "Alright, please",
            "Quick one:", "Whenever you're ready,"]


def template_instruction(task_texts, rng=None):
    rng = rng or random
    jobs = [t[0].lower() + t[1:] for t in task_texts]
    parts = []
    n = len(jobs)
    for i, t in enumerate(jobs):
        if i == 0:
            parts.append(f"first {t}")
        elif i == n - 1:
            parts.append(f"and finally {t}")
        else:
            parts.append(f"then {t}")
    return f"{rng.choice(_OPENERS)} " + ", ".join(parts) + "."
