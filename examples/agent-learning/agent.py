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

"""A small, deterministic support agent with two workflows.

This stands in for your own agent. It is rule-based on purpose so the
example runs without a model and gives the same results every time. The
only thing the rest of the example relies on is the trace each run returns:
which tools were called, with what input, what came back, and the final
answer.
"""

import re

ABSTAIN_ANSWER = (
    "I couldn't find a support policy that covers this, so I'm passing it to a person."
)

STOPWORDS = frozenset(
    """a about after an and any are as at be been can could do does for from
    get has have how i if in is it its long me my no not now of on or our
    please still so that the their them then there this to us was we what
    when where which who why will with would yesterday you your""".split()
)


def terms(text):
    """Lowercase content words with a crude suffix strip (refunds, refunded -> refund)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


class Budget:
    """Caps tool calls so workflows are compared on equal terms."""

    def __init__(self, max_tool_calls):
        self.max_tool_calls = max_tool_calls
        self.used = 0

    def take(self):
        if self.used >= self.max_tool_calls:
            return False
        self.used += 1
        return True


class Run:
    def __init__(self, workflow, task, max_tool_calls):
        self.workflow = workflow
        self.task = task
        self.budget = Budget(max_tool_calls)
        self.steps = []
        self.retrieved = []

    def search(self, search_fn, query, limit=3):
        if not self.budget.take():
            self.steps.append({"type": "budget_exhausted", "input": query})
            return []
        hits = search_fn(query, limit)
        ids = [h["_id"] for h in hits]
        for doc_id in ids:
            if doc_id not in self.retrieved:
                self.retrieved.append(doc_id)
        self.steps.append({"type": "search", "input": query, "output": ids})
        return hits

    def note(self, kind, text):
        self.steps.append({"type": kind, "output": text})

    def finish(self, answer, source):
        self.steps.append({"type": "answer", "output": answer})
        return {
            "workflow": self.workflow,
            "task_id": self.task["id"],
            "split": self.task["split"],
            "question": self.task["question"],
            "answer": answer,
            "answer_source": source or "",
            "abstained": source is None,
            "retrieved_ids": list(self.retrieved),
            "tool_calls": self.budget.used,
            "max_tool_calls": self.budget.max_tool_calls,
            "steps": self.steps,
        }


def keyword_query(question):
    # Antfly's query string joins bare terms with AND; any term may match here.
    return " OR ".join(dict.fromkeys(terms(question)))


def baseline(task, search_fn, max_tool_calls=3):
    """Search once and answer from the best-scoring document."""
    run = Run("baseline", task, max_tool_calls)
    query = keyword_query(task["question"])
    hits = run.search(search_fn, query) if query else []
    if not hits:
        return run.finish(ABSTAIN_ANSWER, None)
    top = hits[0]
    return run.finish(top["answer"], top["_id"])


def grounded(task, search_fn, max_tool_calls=3, min_overlap=2):
    """The revised workflow, written after reading the baseline's failures.

    1. Search current policies only, so superseded versions never answer.
    2. Look up the policy's exceptions and use one if the question triggers it.
    3. Abstain unless the question and the chosen text share enough terms.
    """
    run = Run("grounded", task, max_tool_calls)
    query = keyword_query(task["question"])
    if not query:
        return run.finish(ABSTAIN_ANSWER, None)

    hits = run.search(search_fn, f"+kind:policy +status:current +({query})")
    if not hits:
        return run.finish(ABSTAIN_ANSWER, None)
    chosen = policy = hits[0]

    asked = set(terms(task["question"]))
    exceptions = run.search(
        search_fn,
        f"+kind:exception +status:current +applies_to:{policy['_id']}",
        limit=10,
    )
    for exc in exceptions:
        matched = set(terms(" ".join(exc.get("triggers") or []))) & asked
        if matched:
            run.note("exception", f"{exc['_id']} applies because of {sorted(matched)}")
            chosen = exc
            break

    text = " ".join(
        d["title"] + " " + d["body"]
        for d in {policy["_id"]: policy, chosen["_id"]: chosen}.values()
    )
    covered = asked & set(terms(text))
    run.note("grounding", f"{chosen['_id']} shares {sorted(covered)}")
    if len(covered) < min_overlap:
        return run.finish(ABSTAIN_ANSWER, None)
    return run.finish(chosen["answer"], chosen["_id"])


WORKFLOWS = {"baseline": baseline, "grounded": grounded}
