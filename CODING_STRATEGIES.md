# Coding strategies for a small local model

How to get useful work on a large code base out of Xenon (RWKV-7 G1l 2.9B, ~23 tok/s writing,
~900 tok/s reading, recurrent state that can be cached and copied). Collected 2026-11-09 from the
sources linked below; numbers are theirs unless marked as ours.

The short version: a small model does well when it is asked one narrow question at a time, sees
only the few lines that matter, writes short output in a fixed shape, and has its work checked by
running code. Training helps its habits, not its knowledge of the repo.

## Ranked, with status

| # | Technique | Evidence | Status here |
|---|---|---|---|
| 1 | **Snap paths to real files.** Map a written path to the one repo file ending with it, or constrain decoding to real paths. | Ours: 87/100 answers name the right file, 37/100 the exact path. Grammar-constrained output didn't cost accuracy ([JSONSchemaBench](https://arxiv.org/html/2501.10868v3)). | Done for the folder agent (`snap_path`, `snap_to_list`). |
| 2 | **Read twice.** Recurrent models read once, front to back: state the question before the material, and again after it. | +11 points on average across 16 recurrent models, RWKV among them ([Just Read Twice](https://arxiv.org/abs/2407.05483)); [RWKV prompt guide](https://wiki.rwkv.com/RWKV-Prompts/prompt-guidelines.html). | Done for code search results. The cached repo-map preamble comes before any question, so it is only for orientation. |
| 3 | **Short edits, applied leniently.** SEARCH/REPLACE blocks instead of whole-file rewrites; apply with whitespace-tolerant matching. | Strict patch application gave 9x more edit errors ([aider](https://aider.chat/2023/12/21/unified-diffs.html)). Small models gain little accuracy from any format ([Diff-XYZ](https://arxiv.org/abs/2510.12487v2)), so the win is speed: a 300-line rewrite is ~130 s here, an edit block ~7 s. | Short edits yes, SEARCH/REPLACE no: Xenon couldn't fill in both halves. What works for it is in "How Xenon edits" below. |
| 4 | **Sample several, let tests choose.** N candidate edits from copies of one state; prefer those that pass regression and reproduction tests. | 15.9% → 56% on SWE-bench Lite from 1 to 250 samples ([Large Language Monkeys](https://arxiv.org/abs/2407.21787)); test selection +6.3 points in Agentless; a 3B model beat GPT-4o-mini with sampling plus test-based selection ([S*](https://arxiv.org/abs/2502.14382)). RWKV state copies make each extra sample cost only its own tokens. | Next. |
| 5 | **Fixed pipeline, not an open agent.** Localize files (map + search), then functions (signature skeletons), then lines (±10 lines), then edit, then validate. | Agentless beat the agent systems of its time at $0.70/issue; skeletons beat whole files for localization, 58.3% vs 53.7% ([Agentless](https://arxiv.org/html/2407.01489v2)). | Partly: search localizes files. Skeletons to do. |
| 5b | **Don't let the model grade its own diff.** | Ours: asked whether its diff did the goal, Xenon approved an edit that deleted a line and rejected a correct one, inventing a requirement. | Removed. The edit tasks' hidden checks do this job. |
| 6 | **Interface rules.** Reject edits that don't parse, with a one-line reason; show code in ~100-line windows; give search results as one list; feed back only the last few failures. | Without the edit linter, 18.0% → 15.0%; full-file view 12.7% vs 100-line window 18.0%; full history 15.0% vs last five 18.0%; one-at-a-time search 12.0%, worse than none ([SWE-agent](https://arxiv.org/html/2405.15793v3)). | Failures are already cut to the last 2,000 characters. |
| 7 | **Keyword search, plus definitions and references.** Keep BM25-style search; add a symbol index; re-search with the model's own draft. | BM25 matched the best embeddings on repo-level completion (93.2 vs 94.3 NDCG@10); LLM reranking hurt (27.4 → 9.1) ([CodeRAG-Bench](https://arxiv.org/html/2406.14497)); re-search with a draft gave >10% ([RepoCoder](https://arxiv.org/abs/2303.12570)). | Keyword search with definition boost done (`code_search.rs`). |
| 8 | **Few-shot from this repo in the cached preamble.** 2–3 worked examples in the exact output format. | Removing the demonstration cost SWE-agent 1.7 points; the RWKV guide recommends few-shot. | The example bank does this per question; a fixed set is to do. |
| 9 | **Fine-tune for format and behavior, not facts.** State-tune or QLoRA on question → exact path and diff → edit pairs. | Fine-tuning facts in loses to retrieval ([Ovadia et al.](https://arxiv.org/abs/2312.05934)); training on agent trajectories gave up to +19 points ([SWE-Gym](https://arxiv.org/abs/2412.21139)). | Data ready (`training/make_self_data.py --sft`); run after 3 and 4. |
| 10 | **Think freely, constrain only the edit.** No grammar on the plan; grammar only from the edit marker on. | Strict formats hurt reasoning ([Let Me Speak Freely](https://arxiv.org/abs/2408.02442)). | With 3. |

## How Xenon edits (measured on its own repo and `training/edit_tasks`)

The folder agent's steps, each a fix for a failure seen on Xenon2 itself:

1. **Find the files** with code search on the goal. If Xenon later names a real file search
   missed, that's accepted (unless the caller listed the files).
2. **Pick the file**: the reply is started as `FILE:`, so Xenon can only write a path. Without
   that, G1 (trained on tool use) answered a long code prompt with a `<tool_call>` to grep.
3. **Pick the lines** in code, not by asking: the best line for the goal, weighted like code
   search (rare words and identifiers count, common words don't), plus its neighbours, or a
   definition's first lines. Blank lines stay off the edges; Xenon dropped them.
4. **Write the edit** in a short prompt of its own (goal plus three lines of context either side,
   no file listing), with the reply started as "Lines a-b of `file` are: … With the goal done,
   they become:". Three things each broke this step:
   - Too much in front of it. With three large files read first, Xenon dropped entries from a
     set it was asked to add one to. With only the lines in front of it, it got the same edit right.
   - The lines to change in the question instead of the reply's start: 2 of 7 tasks passed, not 7 of 7.
   - SEARCH/REPLACE. Xenon misremembered a long line five times running, copied the prompt's
     example, or wrote the same lines into both halves.
5. **Check, and undo on failure**: run the tests; a failure undoes the edit and goes back to
   Xenon with its diff and the output, so a run out of budget never leaves a broken file.

`training/run_edit_tasks.py` scores this on small tasks, each with a hidden check of the goal and
of everything around it (tests passing isn't enough: one edit fixed `average` and deleted the line
above it, and its tests passed). On Xenon2 itself, "add models to SKIP_DIRS in the extension's
context.ts" took 36 s from the goal alone (commit `b15c9c1`).

## Nested specialists (2026-11-10)

The steps above handle one change in one place. Goals that need more are handled by narrow
specialists under an orchestrator, each a fix for a failure seen in the edit tasks:

| Specialist | Job | Why it exists |
|---|---|---|
| Orchestrator | One edit at a time first; plans first only when the goal says "both", "everywhere", "each" or "all of" | Planning every goal was a regression (7/7 to 2/9): a small model over-decomposes |
| Planner | Numbered steps, one change in one place each; a step that can't be done is planned again into parts, two levels deep | Steps were dropped when the Locator found no place for a vague one |
| Locator | A goal or step naming one file goes straight to it; lines picked by search; never lines another step of the round rewrote | Asked, Xenon named shop.py for "update the import in report.py"; overlapping steps duplicated lines |
| Editor | Rewrites the picked lines (as above), told what earlier steps changed | Not told, the fetch step added a constant step 1 had already added |
| Test-writer | When the goal says its test changes, rewrites the failing assertion; never drops one | A behavior change failed the old test, and the code edit was undone |
| Renamer | "Rename X to Y": whole-word replace in every file using X, checked once, kept or undone together | One-place steps renamed an import and left the call beside it |
| Checker | Tests; no new compiler warnings; a "remove the warning" goal must remove one; no name defined twice | Weak tests passed a re-indent that removed nothing and a constant defined twice |

Edit tasks (ten, hidden checks): 7/10 before this (all multi-place tasks failed), 10/10 after,
with two-places-one-file's check tightened to one definition. Every specialist is unit-tested
end to end with a scripted stand-in for the model (`Model` trait), and mutation testing covers the
loop (see PLAN.md).

## Known not to work for small models

- Search that returns one result at a time (worse than no search), whole files or full history in
  the prompt, line-number edit formats, exact-match patch application.
- Reranking retrieved chunks with the model; embedding-only retrieval.
- Majority vote without running anything: it plateaus. Tests are what make sampling pay.
- Training on raw repo code to teach it the code base.
- Open-ended agent loops: every free choice is another place for a 2.9B model to go wrong.

## Measurements here

"Which file defines X", 100 held-out questions from `training/make_self_data.py`:

| Setup | exact path | right file |
|---|---|---|
| no search | 0 | 1 |
| search, results before the question | 37 | 87 |
| search, question before and after the results, definitions must start their line | 83 | 89 |
