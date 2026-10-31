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

"""Runs the full demo against a real server. Set ANTFLY_URL to enable."""

import json
import os
import uuid

import pytest

from antfly_client import Antfly
from learn import Store, compare, main

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTFLY_URL"), reason="ANTFLY_URL not set"
)


@pytest.fixture
def prefix():
    name = "agent_learning_test_" + uuid.uuid4().hex[:8]
    yield name
    client = Antfly()
    for suffix in ("policies", "traces"):
        client.drop_table(f"{name}_{suffix}")


def test_demo_end_to_end(prefix, tmp_path, capsys):
    main(["demo", "--prefix", prefix, "--out", str(tmp_path)])
    capsys.readouterr()

    store = Store(Antfly(), prefix)
    heldout = compare(store, "heldout")
    assert heldout["grounded"]["passed"] > heldout["baseline"]["passed"]
    assert heldout["grounded"]["mean_tool_calls"] <= 3

    stored = store.traces_matching("+split:heldout +workflow:baseline")
    assert len(stored) == 7
    assert all(r["outcome"] in ("pass", "fail") for r in stored)
    assert all(r["steps"] and r["steps"][-1]["type"] == "answer" for r in stored)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["split"] == "train"
    assert manifest["preference_pairs"] >= 1
    exported = [
        json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()
    ]
    assert exported and all(e["task_id"].startswith("t") for e in exported)

    with pytest.raises(SystemExit, match="only failed runs"):
        main(["correct", "--prefix", prefix, "grounded.t01.r1", "nope"])
    main(
        [
            "correct",
            "--prefix",
            prefix,
            "baseline.t01.r1",
            "Refunds are available within 30 days.",
        ]
    )
    run = store.trace("baseline.t01.r1")
    assert run["correction"] == "Refunds are available within 30 days."
    assert run["outcome"] == "fail"
    assert store.trace("baseline.nope.r1") is None
