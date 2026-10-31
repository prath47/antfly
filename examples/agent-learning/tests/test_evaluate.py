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

from evaluate import score_run
from learn import build_dataset, load_json

CORPUS = {p["id"]: p for p in load_json("policies.json")}
ANNUAL = {
    "id": "t02",
    "expected_source": "refunds_annual",
    "must_include": ["pro rata", "60 days"],
}
NONE = {"id": "t10", "expected_source": None, "must_include": []}


def run(source, answer=None, retrieved=(), abstained=False):
    return {
        "answer_source": source,
        "answer": answer
        if answer is not None
        else CORPUS.get(source, {}).get("answer", ""),
        "retrieved_ids": list(retrieved),
        "abstained": abstained,
    }


def test_pass_requires_every_expected_fact():
    assert (
        score_run(run("refunds_annual", retrieved=["refunds_annual"]), ANNUAL, CORPUS)[
            "outcome"
        ]
        == "pass"
    )
    result = score_run(run("refunds_annual", answer="Pro rata refund."), ANNUAL, CORPUS)
    assert result["outcome"] == "fail"
    assert result["failure_reason"] == "missing_fact:60 days"


def test_failure_reasons():
    assert score_run(run("refunds", retrieved=["refunds_annual"]), ANNUAL, CORPUS) == {
        "outcome": "fail",
        "score": 0.0,
        "failure_reason": "missed_exception",
        "retrieval_hit": "yes",
    }
    assert (
        score_run(run("refunds_v2"), ANNUAL, CORPUS)["failure_reason"]
        == "used_superseded_policy"
    )
    assert score_run(run("seats"), ANNUAL, CORPUS)["failure_reason"] == "wrong_policy"
    assert (
        score_run(run(None, answer="", abstained=True), ANNUAL, CORPUS)[
            "failure_reason"
        ]
        == "abstained"
    )


def test_unanswerable_questions_must_be_abstained():
    assert (
        score_run(run(None, answer="", abstained=True), NONE, CORPUS)["outcome"]
        == "pass"
    )
    assert (
        score_run(run("refunds"), NONE, CORPUS)["failure_reason"]
        == "answered_unanswerable"
    )


def scored(run_id, task_id, outcome, answer, **extra):
    return dict(
        run_id=run_id,
        task_id=task_id,
        workflow=run_id.split(".")[0],
        question=f"question {task_id}",
        outcome=outcome,
        answer=answer,
        failure_reason="" if outcome == "pass" else "wrong_policy",
        steps=[],
        **extra,
    )


def test_dataset_pairs_failures_with_passing_run_or_correction():
    runs = [
        scored("baseline.t1.r1", "t1", "fail", "bad"),
        scored("grounded.t1.r1", "t1", "pass", "good"),
        scored("baseline.t2.r1", "t2", "fail", "bad2", correction="fixed2"),
        scored("baseline.t3.r1", "t3", "fail", "bad3"),
    ]
    examples, pairs = build_dataset(runs)
    assert [(e["run_id"], e["label"], e["completion"]) for e in examples] == [
        ("grounded.t1.r1", "passed", "good"),
        ("baseline.t2.r1", "human_correction", "fixed2"),
    ]
    assert [(p["task_id"], p["chosen"], p["rejected"]) for p in pairs] == [
        ("t1", "good", "bad"),
        ("t2", "fixed2", "bad2"),
    ]


def test_every_task_points_at_a_real_document():
    for t in load_json("tasks.json"):
        assert t["split"] in ("train", "heldout")
        if t["expected_source"] is not None:
            doc = CORPUS[t["expected_source"]]
            assert doc["status"] == "current"
            for fact in t["must_include"]:
                assert fact.lower() in doc["answer"].lower(), (t["id"], fact)
