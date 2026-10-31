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

"""Independent checks for a completed run.

The checks only use the task's expected answer facts and the corpus
metadata. They never ask the agent to grade itself.
"""


def score_run(run, task, corpus):
    """Return outcome fields to store on the run's trace.

    `corpus` maps document id -> policy document.
    """
    expected = task.get("expected_source")
    source = run.get("answer_source") or None
    answer = run.get("answer", "").lower()

    if expected is None:
        passed = bool(run.get("abstained"))
        reason = "" if passed else "answered_unanswerable"
        retrieval_hit = "n/a"
    else:
        missing = [f for f in task.get("must_include", []) if f.lower() not in answer]
        passed = not run.get("abstained") and not missing
        retrieval_hit = "yes" if expected in run.get("retrieved_ids", []) else "no"
        reason = "" if passed else _failure_reason(run, task, corpus, source, missing)

    return {
        "outcome": "pass" if passed else "fail",
        "score": 1.0 if passed else 0.0,
        "failure_reason": reason,
        "retrieval_hit": retrieval_hit,
    }


def _failure_reason(run, task, corpus, source, missing):
    expected = task["expected_source"]
    if run.get("abstained"):
        return "abstained"
    used = corpus.get(source, {})
    if used.get("status") == "superseded":
        return "used_superseded_policy"
    if corpus.get(expected, {}).get("applies_to") == source:
        return "missed_exception"
    if source != expected:
        return "wrong_policy"
    return "missing_fact:" + ",".join(missing)
