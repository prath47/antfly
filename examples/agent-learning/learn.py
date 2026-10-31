# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Store agent traces in Antfly, then search, score, compare and export them.

python learn.py demo            # the whole loop, start to finish
python learn.py --help          # individual steps
"""

import argparse
import json
import sys
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path

from agent import WORKFLOWS
from antfly_client import Antfly, AntflyError
from evaluate import score_run

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load_json(name):
    return json.loads((DATA / name).read_text())


POLICY_KEYWORDS = ("id", "kind", "status", "applies_to")
TRACE_KEYWORDS = (
    "run_id",
    "tag",
    "workflow",
    "task_id",
    "split",
    "answer_source",
    "outcome",
    "failure_reason",
    "retrieval_hit",
)


class Store:
    """Trace storage on top of two Antfly tables."""

    def __init__(self, client, prefix="agent_learning"):
        self.db = client
        self.policies = f"{prefix}_policies"
        self.traces = f"{prefix}_traces"

    def setup(self, policies, reset=False):
        tables = [
            (self.policies, {"keyword_fields": POLICY_KEYWORDS}),
            (
                self.traces,
                {"keyword_fields": TRACE_KEYWORDS, "keyword_lists": ("retrieved_ids",)},
            ),
        ]
        for table, schema in tables:
            if reset:
                self.db.drop_table(table)
            try:
                self.db.create_table(table, **schema)
            except AntflyError as err:
                if err.status != 409:  # 409: table already exists
                    raise
        self.db.put(self.policies, {p["id"]: dict(p) for p in policies})

    def search_policies(self, query, limit):
        return self.db.search(self.policies, query, limit=limit)

    def record(self, run, tag):
        run_id = f"{run['workflow']}.{run['task_id']}.{tag}"
        doc = {k: v for k, v in run.items() if k != "steps"}
        doc.update(
            run_id=run_id,
            tag=tag,
            steps_json=json.dumps(run["steps"]),
            steps_text=" | ".join(_step_text(s) for s in run["steps"]),
            outcome="unscored",
            recorded_at=int(time.time()),
        )
        self.db.put(self.traces, {run_id: doc})
        return run_id

    def update(self, docs):
        self.db.put(self.traces, {d["run_id"]: _strip(d) for d in docs})

    def traces_matching(self, query):
        return [_load(d) for d in self.db.scan(self.traces, query)]

    def trace(self, run_id):
        doc = self.db.get(self.traces, run_id)
        return None if doc is None else _load(doc)


def _step_text(step):
    out = step.get("output")
    if isinstance(out, list):
        out = " ".join(out)
    return " ".join(
        str(x) for x in (step["type"], step.get("input", ""), out or "") if x
    )


def _strip(doc):
    doc = {k: v for k, v in doc.items() if not k.startswith("_")}
    doc.pop("steps", None)
    return doc


def _load(doc):
    doc = dict(doc)
    doc["retrieved_ids"] = list(doc.get("retrieved_ids") or [])
    doc["steps"] = json.loads(doc.get("steps_json") or "[]")
    return doc


def run_workflow(store, workflow, split, tag, max_tool_calls):
    fn = WORKFLOWS[workflow]
    tasks = [t for t in load_json("tasks.json") if t["split"] == split]
    for task in tasks:
        run = fn(task, store.search_policies, max_tool_calls=max_tool_calls)
        store.record(run, tag)
    print(
        f"ran {workflow} on {len(tasks)} {split} tasks (budget {max_tool_calls} tool calls)"
    )


def evaluate_unscored(store):
    tasks = {t["id"]: t for t in load_json("tasks.json")}
    corpus = {p["id"]: p for p in load_json("policies.json")}
    runs = store.traces_matching("outcome:unscored")
    for run in runs:
        run.update(score_run(run, tasks[run["task_id"]], corpus))
    store.update(runs)
    print(f"scored {len(runs)} runs")
    return runs


def show_failures(store, split, tag="r1"):
    query = f"+tag:{tag} +split:{split} +outcome:fail"
    failed = store.traces_matching(query)
    print(f"\nfailed runs ({query}): {len(failed)}")
    for (wf, reason), n in sorted(
        Counter((r["workflow"], r["failure_reason"]) for r in failed).items()
    ):
        print(f"  {wf:<17} {reason:<24} {n}")

    query = f"+tag:{tag} +split:{split} +outcome:fail +retrieval_hit:yes"
    found = store.traces_matching(query)
    print(
        f"\nfailed although retrieval found the right document ({query}): {len(found)}"
    )
    for r in sorted(found, key=lambda r: r["run_id"]):
        print(f"  {r['run_id']:<28} {r['failure_reason']:<20} {r['question']}")
    return failed


def compare(store, split, tag="r1", workflows=None):
    runs = store.traces_matching(f"+tag:{tag} +split:{split} -outcome:unscored")
    by_wf = defaultdict(dict)
    for r in runs:
        if workflows and r["workflow"] not in workflows:
            continue
        by_wf[r["workflow"]][r["task_id"]] = r
    names = sorted(by_wf)
    shared = sorted(set.intersection(*(set(by_wf[w]) for w in names))) if names else []
    if not shared:
        print(f"no tasks in {split} were run by every workflow")
        return {}

    caps = {r["max_tool_calls"] for wf in names for r in by_wf[wf].values()}
    if len(caps) > 1:
        print(
            f"warning: runs used different tool-call caps {sorted(caps)}; not a fair comparison"
        )

    print(f"\n{split}: {len(shared)} tasks run by {', '.join(names)}")
    summary = {}
    for wf in names:
        rows = [by_wf[wf][t] for t in shared]
        passed = sum(r["outcome"] == "pass" for r in rows)
        calls = sum(r["tool_calls"] for r in rows) / len(rows)
        cap = max(r["max_tool_calls"] for r in rows)
        summary[wf] = {"passed": passed, "total": len(rows), "mean_tool_calls": calls}
        print(
            f"  {wf:<17} {passed}/{len(rows)} passed, {calls:.1f} tool calls per task (cap {cap})"
        )

    print("\n  task  " + "".join(f"{w:<32}" for w in names))
    for t in shared:
        cells = []
        for wf in names:
            r = by_wf[wf][t]
            cells.append(
                r["outcome"]
                if r["outcome"] == "pass"
                else f"fail ({r['failure_reason']})"
            )
        print(f"  {t:<5} " + "".join(f"{c:<32}" for c in cells))
    return summary


def correct(store, run_id, answer):
    run = store.trace(run_id)
    if run is None:
        sys.exit(f"no run with id {run_id}")
    if run.get("outcome") != "fail":
        sys.exit(f"{run_id} is {run.get('outcome')}; only failed runs take corrections")
    run["correction"] = answer
    store.update([run])
    print(f"stored correction on {run_id}")


def build_dataset(runs):
    """Turn scored runs into supervised examples and preference pairs.

    Pure function so it can be tested without a server.
    """
    examples, pairs = [], []
    by_task = defaultdict(list)
    for r in sorted(runs, key=lambda r: r["run_id"]):
        by_task[r["task_id"]].append(r)

    for task_id, task_runs in sorted(by_task.items()):
        good = [r for r in task_runs if r["outcome"] == "pass"]
        bad = [r for r in task_runs if r["outcome"] == "fail"]
        for r in good:
            examples.append(_example(r, r["answer"], "passed"))
        for r in bad:
            if r.get("correction"):
                examples.append(_example(r, r["correction"], "human_correction"))
        chosen = good[0]["answer"] if good else None
        for r in bad:
            preferred = r.get("correction") or chosen
            if preferred and preferred != r["answer"]:
                pairs.append(
                    {
                        "task_id": task_id,
                        "prompt": r["question"],
                        "chosen": preferred,
                        "rejected": r["answer"],
                        "rejected_run": r["run_id"],
                        "rejected_reason": r["failure_reason"],
                    }
                )
    return examples, pairs


def _example(run, completion, label):
    return {
        "task_id": run["task_id"],
        "run_id": run["run_id"],
        "workflow": run["workflow"],
        "label": label,
        "prompt": run["question"],
        "completion": completion,
        "trajectory": run["steps"],
    }


def export(store, split, out_dir, tag="r1", allow_heldout=False):
    if split == "heldout" and not allow_heldout:
        sys.exit(
            "refusing to export held-out runs; they are for measurement (use --allow-heldout)"
        )
    runs = store.traces_matching(f"+tag:{tag} +split:{split} -outcome:unscored")
    examples, pairs = build_dataset(runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "sft.jsonl", examples)
    _write_jsonl(out / "preferences.jsonl", pairs)
    manifest = {
        "split": split,
        "tag": tag,
        "runs": len(runs),
        "examples": len(examples),
        "preference_pairs": len(pairs),
        "traces_table": store.traces,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"\nexported {len(examples)} examples and {len(pairs)} preference pairs to {out}/"
    )
    return manifest


def _write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def demo(store, args):
    print("== 1. load the policy corpus")
    store.setup(load_json("policies.json"), reset=True)

    print("\n== 2. run the baseline agent on the training tasks and store its traces")
    run_workflow(store, "baseline", "train", args.tag, args.max_tool_calls)
    evaluate_unscored(store)

    print("\n== 3. search the traces for failures")
    show_failures(store, "train", args.tag)

    print(
        "\n== 4. try a revised workflow: current policies only, exceptions, grounding check"
    )
    run_workflow(store, "grounded", "train", args.tag, args.max_tool_calls)
    evaluate_unscored(store)
    compare(store, "train", args.tag)

    print("\n== 5. measure both workflows on held-out tasks with the same budget")
    for wf in ("baseline", "grounded"):
        run_workflow(store, wf, "heldout", args.tag, args.max_tool_calls)
    evaluate_unscored(store)
    compare(store, "heldout", args.tag)
    show_failures(store, "heldout", args.tag)

    print("\n== 6. export training data from the training split only")
    export(store, "train", args.out, args.tag)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--url", help="Antfly URL (default: $ANTFLY_URL or http://127.0.0.1:8080)"
    )
    common.add_argument("--prefix", default="agent_learning", help="table name prefix")
    common.add_argument("--tag", default="r1", help="label for this batch of runs")
    common.add_argument("--max-tool-calls", type=int, default=3)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def command(name, help):
        return sub.add_parser(name, help=help, parents=[common])

    p = command("demo", "run the whole loop")
    p.add_argument("--out", default="out")
    p = command("setup", "create tables and load the policy corpus")
    p.add_argument("--reset", action="store_true", help="drop existing tables first")
    p = command("run", "run one workflow and store its traces")
    p.add_argument("--workflow", choices=sorted(WORKFLOWS), required=True)
    p.add_argument("--split", choices=["train", "heldout"], default="train")
    command("evaluate", "score runs that have no outcome yet")
    p = command("failures", "search failed runs")
    p.add_argument("--split", choices=["train", "heldout"], default="train")
    p = command("compare", "compare workflows on the same tasks")
    p.add_argument("--split", choices=["train", "heldout"], default="heldout")
    p = command("correct", "attach a human-corrected answer to a failed run")
    p.add_argument("run_id")
    p.add_argument("answer")
    p = command("export", "write sft.jsonl and preferences.jsonl")
    p.add_argument("--split", choices=["train", "heldout"], default="train")
    p.add_argument("--out", default="out")
    p.add_argument("--allow-heldout", action="store_true")
    args = parser.parse_args(argv)

    store = Store(Antfly(args.url), args.prefix)
    try:
        dispatch(store, args)
    except AntflyError as err:
        if err.status == 404 and args.cmd != "setup":
            sys.exit(f"{err}\nhave you run `python learn.py setup`?")
        sys.exit(str(err))
    except urllib.error.URLError as err:
        sys.exit(f"can't reach Antfly at {store.db.url}: {err.reason}")


def dispatch(store, args):
    if args.cmd == "demo":
        demo(store, args)
    elif args.cmd == "setup":
        store.setup(load_json("policies.json"), reset=args.reset)
        print(f"loaded policies into {store.policies}")
    elif args.cmd == "run":
        run_workflow(store, args.workflow, args.split, args.tag, args.max_tool_calls)
    elif args.cmd == "evaluate":
        evaluate_unscored(store)
    elif args.cmd == "failures":
        show_failures(store, args.split, args.tag)
    elif args.cmd == "compare":
        compare(store, args.split, args.tag)
    elif args.cmd == "correct":
        correct(store, args.run_id, args.answer)
    elif args.cmd == "export":
        export(store, args.split, args.out, args.tag, args.allow_heldout)


if __name__ == "__main__":
    main()
