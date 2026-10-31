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

import re

from agent import ABSTAIN_ANSWER, baseline, grounded, keyword_query, terms
from learn import load_json

POLICIES = load_json("policies.json")


def fake_search(ranking=None):
    """In-memory stand-in for the two query shapes the agent sends.

    Handles `+field:value` filters and an optional `+(a OR b)` / `a OR b`
    term group. Documents are ranked by matched terms, or by `ranking`
    when a test needs a specific order.
    """
    calls = []

    def search(query, limit):
        calls.append(query)
        filters = dict(re.findall(r"\+(\w+):(\w+)", query))
        group = re.sub(r"\+\w+:\w+", "", query).strip(" +()")
        wanted = set(terms(group.replace(" OR ", " "))) if group else set()
        hits = []
        for doc in POLICIES:
            if any(str(doc.get(k)) != v for k, v in filters.items()):
                continue
            text = set(terms(" ".join([doc["title"], doc["body"]])))
            score = len(wanted & text) if wanted else 1
            if score:
                hits.append(dict(doc, _id=doc["id"], _score=score))
        order = ranking or []
        hits.sort(
            key=lambda d: (
                d["id"] not in order,
                order.index(d["id"]) if d["id"] in order else 0,
                -d["_score"],
            )
        )
        return hits[:limit]

    search.calls = calls
    return search


def task(question, split="train"):
    return {"id": "x", "split": split, "question": question}


def test_keyword_query_drops_stopwords_and_ors_terms():
    assert (
        keyword_query("How do I get a refund on my subscriptions?")
        == "refund OR subscription"
    )


def test_baseline_answers_from_top_hit_even_when_superseded():
    search = fake_search(ranking=["refunds_v2"])
    run = baseline(task("How long do I have to get a refund?"), search)
    assert run["answer_source"] == "refunds_v2"
    assert run["tool_calls"] == 1


def test_grounded_never_reads_superseded_policies():
    search = fake_search(ranking=["refunds_v2"])
    run = grounded(
        task("How long do I have to get a refund on my subscription?"), search
    )
    assert run["answer_source"] == "refunds"
    assert "+status:current" in search.calls[0]


def test_grounded_applies_triggered_exception():
    run = grounded(
        task("Our school is on the education plan. Can we reassign a seat?"),
        fake_search(),
    )
    assert run["answer_source"] == "seats_education"
    assert any(s["type"] == "exception" for s in run["steps"])


def test_grounded_abstains_when_policy_does_not_cover_question():
    run = grounded(
        task("Do you offer a discount for nonprofit subscriptions?"), fake_search()
    )
    assert run["abstained"]
    assert run["answer"] == ABSTAIN_ANSWER


def test_budget_caps_tool_calls():
    search = fake_search()
    run = grounded(
        task("When will my international hardware order ship?"),
        search,
        max_tool_calls=1,
    )
    assert run["tool_calls"] == 1
    assert len(search.calls) == 1
    assert run["steps"][1]["type"] == "budget_exhausted"


def test_trace_records_steps_and_retrieved_ids():
    run = grounded(task("When will my hardware order ship?"), fake_search())
    assert [s["type"] for s in run["steps"]] == [
        "search",
        "search",
        "grounding",
        "answer",
    ]
    assert "shipping" in run["retrieved_ids"]
