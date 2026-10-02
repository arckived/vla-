"""Build paraphrased versions of the LIBERO task instructions.

Step 1 - extract the exact instruction strings the simulator uses:
    python -m vla_study.paraphrase extract --suite libero_object --out data/tasks_libero_object.json

Step 2 - generate paraphrase sets (rule-based levels are deterministic; `llm` is optional):
    python -m vla_study.paraphrase generate --tasks data/tasks_libero_object.json \
        --out data/paraphrases_libero_object.json [--llm Qwen/Qwen2.5-1.5B-Instruct]

Step 3 - READ the output file and fix any paraphrase that changes the meaning.
    A paraphrase test is only valid if every rewrite means the same thing.

Levels (roughly increasing distance from the training wording):
    original : identity mapping, a sanity check that the patching changes nothing
    synonym  : verb swaps    ("pick up" -> "grab", "place" -> "put")
    restruct : new sentence structure ("put the X into the basket")
    verbose  : polite wrapper + extra words ("Could you please ... for me?")
    llm      : free-form rewrite by a small instruction-tuned LLM
    swapped  : CONTROL - each task gets ANOTHER task's instruction. If success barely
               drops, the policy is not really using language (it recognizes the task
               from the scene). Without this control, "robust to paraphrases" is ambiguous.
"""
from __future__ import annotations

import argparse
import json
import os
import re

SYNONYMS = {
    "pick up": "grab",
    "place": "put",
    "put": "place",
    "open": "pull open",
    "close": "shut",
    "turn on": "switch on",
    "push": "shove",
    "on top of": "onto",
}
# One combined regex = all swaps happen in a single pass, so "place"->"put" is never
# turned back into "place" by the "put"->"place" rule. Longest phrases first.
_SYN_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SYNONYMS), key=len, reverse=True)) + r")\b")


def synonym(t: str) -> str:
    return _SYN_RE.sub(lambda m: SYNONYMS[m.group(1)], t)


def restruct(t: str) -> str:
    m = re.match(r"pick up the (.+?) and place it in(?:to)? the (.+)$", t)
    if m:
        return f"put the {m.group(1)} into the {m.group(2)}"
    m = re.match(r"put the (.+?) on(?: top of)? the (.+)$", t)
    if m:
        return f"the {m.group(1)} should go on the {m.group(2)}"
    return f"your task: {t}"


def verbose(t: str) -> str:
    return f"could you please {t} for me?"


def extract(suite: str, out: str) -> None:
    try:
        from libero.libero import benchmark
    except ImportError as e:
        raise SystemExit(f"LIBERO not importable ({e}). Install with: pip install -e '.[libero]' inside lerobot/")
    task_suite = benchmark.get_benchmark_dict()[suite]()
    tasks = [task_suite.get_task(i).language for i in range(task_suite.n_tasks)]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"suite": suite, "tasks": tasks}, f, indent=2)
    print(f"wrote {len(tasks)} instructions to {out}")
    for t in tasks:
        print("  -", t)


def llm_paraphrases(tasks: list[str], model_id: str) -> dict[str, str]:
    from transformers import pipeline

    gen = pipeline("text-generation", model=model_id, device_map="auto")
    out = {}
    for t in tasks:
        msgs = [
            {"role": "system", "content": "You rewrite robot instructions. Keep the exact same meaning and the "
                                          "same objects. Use different wording. Reply with the instruction only."},
            {"role": "user", "content": t},
        ]
        reply = gen(msgs, max_new_tokens=40, do_sample=False)[0]["generated_text"][-1]["content"]
        out[t] = reply.strip().strip('"').rstrip(".").lower()
        print(f"  {t!r} -> {out[t]!r}")
    return out


def generate(tasks_file: str, out: str, llm: str | None) -> None:
    with open(tasks_file) as f:
        tasks = json.load(f)["tasks"]
    sets = {
        "original": {t: t for t in tasks},
        "synonym": {t: synonym(t) for t in tasks},
        "restruct": {t: restruct(t) for t in tasks},
        "verbose": {t: verbose(t) for t in tasks},
        "swapped": {t: tasks[(i + 1) % len(tasks)] for i, t in enumerate(tasks)},
    }
    if llm:
        sets["llm"] = llm_paraphrases(tasks, llm)
    unchanged = [t for t in tasks if sets["synonym"][t] == t]
    if unchanged:
        print(f"note: synonym level left {len(unchanged)} instruction(s) unchanged: {unchanged}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(sets, f, indent=2)
    print(f"wrote levels {list(sets)} to {out}. Now open it and check every line by hand.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--suite", default="libero_object")
    e.add_argument("--out", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--tasks", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--llm", default=None, help="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.suite, a.out)
    else:
        generate(a.tasks, a.out, a.llm)


if __name__ == "__main__":
    main()
