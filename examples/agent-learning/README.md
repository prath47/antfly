# Improve an Agent with Its Own Traces

Store every run of your agent in Antfly, find the runs that went wrong,
try a revised workflow, measure it on questions it hasn't seen, and export
the good and bad runs as training data.

The agent here is a tiny support bot that answers questions from a set of
help-desk policies. It's rule-based so the example needs no model or API key
and prints the same numbers every time. Bring your own agent by replacing
`agent.py`: the rest of the example only needs the trace each run returns.

## What it shows

1. **Trace storage.** Each run becomes one document in an Antfly table: the
   question, every tool call with its input and output, the documents it
   retrieved, the answer, and how much of its tool-call budget it used.
2. **Search.** Failed runs are found with ordinary full-text queries, for
   example "failed runs where retrieval did find the right policy":
   `+outcome:fail +retrieval_hit:yes`.
3. **Evaluation.** A separate check scores each run against facts listed with
   the task (`data/tasks.json`). The agent never grades itself. Scores and
   failure reasons are written back onto the run, and a person can attach a
   corrected answer with `learn.py correct`.
4. **Workflow comparison.** A revised workflow runs on held-out questions
   with the same tool-call cap as the baseline, and the two are compared
   task by task.
5. **Dataset export.** Passing runs and human corrections become supervised
   examples (`sft.jsonl`). A failed run paired with a better answer to the
   same question becomes a preference pair (`preferences.jsonl`). Only the
   training split is exported by default, so held-out results stay honest.

## Prerequisites

- A running Antfly server (`antfly standalone`). No models are needed: the
  example only uses full-text search.
- Python 3.11+. There are no third-party dependencies;
  [uv](https://docs.astral.sh/uv/) is only used to run the tests.

## Run it

```bash
python learn.py demo
```

Set `ANTFLY_URL` (and `ANTFLY_TOKEN` if auth is on) to point at another
server. The demo creates `agent_learning_policies` and
`agent_learning_traces`, dropping them first if they exist; use
`python learn.py demo --prefix my_run` to pick other names.

The interesting part of the output is the held-out comparison:

```text
heldout: 7 tasks run by baseline, grounded
  baseline          4/7 passed, 1.0 tool calls per task (cap 3)
  grounded          6/7 passed, 1.9 tool calls per task (cap 3)

  task  baseline                        grounded
  h01   fail (used_superseded_policy)   pass
  h02   pass                            pass
  h03   pass                            pass
  h04   pass                            pass
  h05   pass                            pass
  h06   fail (answered_unanswerable)    pass
  h07   fail (answered_unanswerable)    fail (answered_unanswerable)
```

The baseline searches once and answers from the best hit. On the training
questions its failures come in two kinds: it quoted a superseded version of
the refund policy, and it answered a question no policy covers. The
`grounded` workflow was written in response. It searches current policies
only, checks the policy's exceptions, and passes the question to a person
when the policy and the question have little in common.

h07 still fails. The question ("I returned an international hardware order.
When will the refund arrive?") shares enough words with the shipping policy
to pass the grounding check, but no policy covers returns. Failures like this
are what the next round of trace review is for. Because h07 is held out, fix
it by adding a similar question to the training split rather than training on
h07 itself.

On training runs, a person can attach the answer they wanted to a failed run.
The next export turns it into a supervised example and a preference pair:

```bash
python learn.py correct baseline.t10.r1 \
  "We don't offer nonprofit discounts that I can find, so I'm passing this to a person."
python learn.py export --out out
```

## Steps one at a time

```bash
python learn.py setup --reset
python learn.py run --workflow baseline --split train
python learn.py evaluate
python learn.py failures --split train
python learn.py run --workflow grounded --split train
python learn.py run --workflow baseline --split heldout
python learn.py run --workflow grounded --split heldout
python learn.py evaluate
python learn.py compare --split heldout
python learn.py export --split train --out out
```

Runs are keyed `<workflow>.<task>.<tag>`. Rerunning with the same `--tag`
replaces the earlier runs, including any corrections on them. Use a new tag
(`run --workflow grounded --tag r2`) to keep both. `failures`, `compare` and
`export` read one tag at a time (`r1` unless you pass `--tag`), so batches
never mix. `compare` warns if the workflows ran under different tool-call
caps.

## Searching traces yourself

Every field is searchable, so the CLI works too:

```bash
# Which runs quoted the old refund policy?
antfly query --table agent_learning_traces \
  --full-text-search 'answer_source:refunds_v2' \
  --fields 'run_id,question,outcome'

# Failed baseline runs that mention shipping anywhere in their steps
antfly query --table agent_learning_traces \
  --full-text-search '+outcome:fail +workflow:baseline steps_text:shipping' \
  --fields 'run_id,question,failure_reason'
```

Identifier fields (`run_id`, `workflow`, `outcome`, `answer_source`,
`retrieved_ids` and so on) are declared as `keyword` when `setup` creates the
table. Without that they would be analyzed as text, and a filter on
`refunds_v2` would also match runs that used `refunds`. Everything else,
including `question`, `answer` and `steps_text`, stays ordinary full text.

The trace document looks like this (`steps_json` is truncated):

```json
{
  "run_id": "baseline.t01.r1",
  "workflow": "baseline",
  "task_id": "t01",
  "split": "train",
  "question": "How long do I have to get a refund on my subscription?",
  "answer": "You can get a refund within 14 days of the charge.",
  "answer_source": "refunds_v2",
  "retrieved_ids": ["refunds_v2", "refunds", "refunds_annual"],
  "tool_calls": 1,
  "max_tool_calls": 3,
  "steps_json": "[{\"type\": \"search\", \"input\": \"refund OR subscription\", ...}]",
  "steps_text": "search refund OR subscription refunds_v2 refunds refunds_annual | answer ...",
  "outcome": "fail",
  "failure_reason": "used_superseded_policy",
  "retrieval_hit": "yes"
}
```

## Using it with a real agent

- Record the same fields from your framework's callbacks or tracing hooks.
  Steps are stored as a JSON string plus a flattened `steps_text` so they can
  be searched without fixing a schema for every tool.
- Keep the checks independent of the agent. Fixed facts, unit tests on tool
  output, or human review all work. An LLM judge can be added, but keep held-out
  questions that it doesn't see while you tune workflows.
- Compare workflows under the same budget. A workflow that wins by making
  twice as many tool calls may not be the one you want.
- The exported files are plain JSONL. Fine-tuning on them is a later step and
  isn't covered here.

The loop of comparing workflows on finished runs and then training on the
successful ones is loosely based on Sakana AI's
[MASS](https://pub.sakana.ai/mass/). This example covers the trace and data
side only, with no model training.

## Tests

```bash
uv run pytest                                          # no server needed
ANTFLY_URL=http://127.0.0.1:8080 uv run pytest         # also runs the demo end to end
```

The live test uses tables with a random prefix and drops them afterwards.
