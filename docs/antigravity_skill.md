# Antigravity Skills Research and Master Prompt (Lexsi Track 3)

## Part A. Research summary

### A.1 How skills work in Google Antigravity

- A skill is a **folder** with a required `SKILL.md` and optional `scripts/`, `references/`, `assets/`.
- `SKILL.md` has YAML frontmatter (`name`, `description`) plus a markdown body (Goal, Instructions, Examples, Constraints).
- Only the **description** is used to route: the agent matches your prompt against all skill descriptions, then loads the full body on demand. So write descriptions that say *what it does* and *when to use it*.
- **Install locations**
  - Project scope: `<project-root>/.agent/skills/` (some docs show `.agents/skills/`; if one is not picked up, try the other)
  - Global scope: `~/.gemini/antigravity/skills/`
- **Security:** review `SKILL.md` and `scripts/` of any third-party skill before installing, especially ones that run terminal commands.
- Skills can be invoked by naming them in the prompt ("Use the X skill to ...") or picked automatically from the description.

### A.2 Existing skills worth installing

| Skill | Source | Use for | Install |
|---|---|---|---|
| `hugging-face-model-trainer` (also published as `huggingface-llm-trainer`) | github.com/huggingface/skills | SFT, DPO, GRPO and reward-model training with TRL on HF Jobs; hardware selection, cost estimation, GGUF export | `npx skills add huggingface/skills --skill hugging-face-model-trainer` |
| `trl-training` | github.com/huggingface/trl | TRL CLI for SFT, DPO, GRPO, KTO, RLOO, reward models | `npx skills add https://github.com/huggingface/trl --skill trl-training` |
| `hugging-face-evaluation` | github.com/huggingface/skills | Eval tables in model cards, vLLM/lighteval runs | `npx skills add huggingface/skills --skill hugging-face-evaluation` |
| `hugging-face-trackio` | github.com/huggingface/skills | Experiment tracking dashboards | `npx skills add huggingface/skills --skill hugging-face-trackio` |
| `hugging-face-jobs` | github.com/huggingface/skills | Running compute jobs on HF infrastructure | `npx skills add huggingface/skills --skill hugging-face-jobs` |
| `hugging-face-cli` | github.com/huggingface/skills | Hub operations (upload model, datasets) | `npx skills add huggingface/skills --skill hugging-face-cli` |
| `huggingface-papers` | github.com/huggingface/skills | Paper lookup and linking | `npx skills add huggingface/skills --skill huggingface-papers` |

Other references: Antigravity skills codelab (codelabs.developers.google.com/getting-started-with-antigravity-skills) and the example repo github.com/rominirani/antigravity-skills.

> Verify exact skill names and install commands in each repo's README; names and CLIs change.

### A.3 Skills you must write yourself

No public skills exist for AgentTune or AuditKIT, so write four custom ones. Templates are in Part C.

| Custom skill | Purpose |
|---|---|
| `agenttune-trainer` | How to run AgentTune post-training for this project |
| `auditkit-evaluator` | How to run AuditKIT baselines, evals, failure diagnosis |
| `reward-design-guard` | Reward function design plus anti-reward-hacking checks |
| `hackathon-submission` | README, results tables, demo, submission checklist |

### A.4 Recommended skill stack by phase

| Phase | Skills |
|---|---|
| 0. Setup | `hugging-face-cli`, `agenttune-trainer`, `auditkit-evaluator` |
| 1. Baseline | `auditkit-evaluator`, `hugging-face-trackio` |
| 2. Data | `hugging-face-cli` (datasets) |
| 3. SFT | `hugging-face-model-trainer` / `trl-training`, `agenttune-trainer`, `hugging-face-jobs` |
| 4. RL | `agenttune-trainer`, `reward-design-guard`, `trl-training` (GRPO) |
| 5. Evaluate | `auditkit-evaluator`, `hugging-face-evaluation` |
| 6. Package | `hackathon-submission`, `hugging-face-cli`, `huggingface-papers` |

---

## Part B. Master prompt (paste into Antigravity)

Fill the `[BRACKETS]`. Add or remove skills in the SKILLS block.

```
ROLE
You are a senior ML engineer and agentic post-training specialist. You are building my project for the Lexsi AI Safety & Alignment Hackathon, Track 3: Agentic Fine-tuning. Work autonomously but show a plan first and checkpoint at the end of every phase.

PROJECT GOAL
Post-train a small open-source LLM agent with AgentTune (https://github.com/Lexsi-Labs/AgentTune, docs https://agenttune.lexsi.ai) so it picks the correct tool, writes well-formed arguments, calls tools in the right order, and finishes tasks in fewer steps. Prove the gain with AuditKIT (https://github.com/Lexsi-Labs/AuditKIT, docs https://auditkit.lexsi.ai).
Primary problem statements: #1 Reliable tool calling and #2 Efficient trajectories.
Optional bonus: [#9 reward-hacking defences / #8 self-healing / #3 distillation / none].

SKILLS TO USE (load the matching skill before each task; if one is missing, tell me and continue from official docs)
- agenttune-trainer: all AgentTune training runs
- auditkit-evaluator: all baselines, evaluations, failure diagnosis
- reward-design-guard: any reward function or RL change
- hackathon-submission: README, results, demo, submission checks
- hugging-face-model-trainer / trl-training: TRL SFT and GRPO fallbacks
- hugging-face-trackio: experiment tracking
- hugging-face-jobs: remote compute
- hugging-face-evaluation: model card results
- hugging-face-cli: Hub uploads
- [ADD ANY OTHER SKILL HERE]

CONSTRAINTS
- Compute: [GPU model and memory / credits available]
- Time: [hours remaining] hours, deadline [date/time]
- Team: [size and skills]
- Base model: [model name, or "recommend one that AgentTune supports"]
- Do not invent AgentTune or AuditKIT APIs. Read the repos and docs first; cite the file or page you relied on.
- Do not fabricate results. Every number in any report must come from a logged run.
- Keep everything reproducible: pinned versions, fixed seeds, saved configs.
- Respect dataset licenses; record them in a data card.
- Keep train and eval data strictly separate.

PHASES (stop at each CHECKPOINT and summarise before continuing)
0. SETUP: inspect both repos and docs; list supported models, algorithms, benchmarks, metrics; set up env and repo layout (configs/, data/, src/, results/, notebooks/); confirm hello-world runs for both libraries. CHECKPOINT: tell me what is supported and any blockers.
1. BASELINE: run AuditKIT on the untouched base model; record tool-call accuracy, argument validity, order correctness, task success rate, steps per task, redundant-call rate; categorise failures. CHECKPOINT: baseline table and top failure modes.
2. DATA: choose public function-calling data (check licences) plus on-policy rollouts; filter to correct, efficient trajectories; write a data card. CHECKPOINT: dataset stats and splits.
3. SFT: train for format and syntax with AgentTune (TRL as fallback); track in Trackio. CHECKPOINT: SFT results vs baseline.
4. RL: implement reward = format + tool selection + arguments + order + verified outcome, minus an efficiency penalty applied ONLY to successful runs; train with AgentTune; monitor KL, reward curves, sample outputs. Use reward-design-guard to test for hacking. CHECKPOINT: RL results vs SFT and baseline.
5. EVALUATE: full AuditKIT run with at least [3] seeds if budget allows; ablations (base, SFT, SFT+RL, with/without efficiency reward); failure analysis with examples. [Optional bonus work here.]
6. PACKAGE: README, model card, results tables and plots, short demo script (before vs after), pitch outline, reproducibility check from a fresh clone. CHECKPOINT: submission checklist status.
7. SUBMIT: final verification, then list what to upload where.

FALLBACKS
- If RL is unstable or compute runs short: fall back to the best SFT checkpoint plus rejection sampling, and say so honestly in the report.
- If a library feature is missing: tell me, propose the closest supported alternative, and continue.

OUTPUT RULES
- Start by confirming which skills you found, then give the Phase 0 plan.
- Be concise. Use tables for results. Flag assumptions and unknowns explicitly.
- At every checkpoint, end with: DONE / NEXT / RISKS.
```

---

## Part C. Custom SKILL.md templates

Save each as `.agent/skills/<name>/SKILL.md` (project) or in `~/.gemini/antigravity/skills/<name>/SKILL.md` (global).
Replace `[TBD]` items after reading the AgentTune and AuditKIT docs. Do not invent commands.

### C.1 `agenttune-trainer`

```markdown
---
name: agenttune-trainer
description: Runs and configures AgentTune post-training (SFT and RL) for reliable, efficient tool-calling agents. Use when the user wants to train, fine-tune, or post-train an agent, or change training configs for the Lexsi Track 3 project.
---

# AgentTune Trainer

## Goal
Post-train a small open model with AgentTune to improve tool selection, argument validity, call ordering, and step efficiency.

## Instructions
1. Read the AgentTune README and https://agenttune.lexsi.ai. Record supported models, algorithms, dataset formats, and CLI/API entry points in references/agenttune_notes.md. [TBD]
2. Validate the dataset format against the docs before training; fail early on schema errors.
3. Stage A: SFT for format and syntax. Stage B: RL with verifiable rewards (see reward-design-guard).
4. Save every run's config, seed, library versions, and git commit in results/runs/<run_id>/.
5. Log metrics with Trackio or wandb. Save checkpoints at fixed intervals.
6. After each run, hand off to auditkit-evaluator.

## Examples
- Input: "Train the SFT stage on dataset X." Output: validated config, run command, expected outputs, checkpoint path.

## Constraints
- Do not invent flags or APIs; cite the doc or source file used.
- Do not train on evaluation data.
- Do not delete checkpoints without asking.
- Start with small smoke-test runs before full runs.
```

### C.2 `auditkit-evaluator`

```markdown
---
name: auditkit-evaluator
description: Runs AuditKIT benchmarks, metrics, and failure diagnosis for tool-calling agents. Use for baselines, post-training evaluation, before/after comparisons, and failure analysis in the Lexsi Track 3 project.
---

# AuditKIT Evaluator

## Goal
Produce trustworthy, reproducible measurements of tool-call accuracy, argument validity, order correctness, task success, steps per task, and redundant-call rate.

## Instructions
1. Read the AuditKIT README and https://auditkit.lexsi.ai. Record available benchmarks, metrics, and export formats in references/auditkit_notes.md. [TBD]
2. Always evaluate the base model first and store it as the baseline.
3. Use identical settings, data, and seeds for baseline and trained models.
4. Run multiple seeds when budget allows; report mean and spread.
5. Categorise failures (wrong tool, bad arguments, wrong order, loops, redundant calls, premature stop) with 3-5 concrete examples each.
6. Write results to results/ as CSV/JSON plus a markdown table.

## Examples
- Input: "Compare base vs SFT+RL." Output: table of metrics, deltas, failure-category shifts, caveats.

## Constraints
- Never report numbers that were not produced by a logged run.
- Never evaluate on training data.
- State sample sizes and any limitations next to every claim.
```

### C.3 `reward-design-guard`

```markdown
---
name: reward-design-guard
description: Designs and audits RL reward functions for tool-calling agents and tests them for reward hacking. Use whenever creating, changing, or debugging a reward, judge, or penalty.
---

# Reward Design Guard

## Goal
Raise real task success without letting the agent game the reward.

## Instructions
1. Decompose the reward: format, tool selection, arguments, order, verified outcome, efficiency penalty.
2. Apply the efficiency penalty only to runs that succeed, otherwise "do nothing" wins.
3. Prefer verifiable outcome checks over LLM judges; if a judge is used, calibrate it on labelled examples.
4. Before full training, run a hacking audit: try degenerate policies (empty call, repeated call, copy expected output, skip tools) and confirm each scores lower than a correct policy.
5. During training, sample and read outputs at fixed intervals; watch reward vs real success diverging.
6. Compare outcome-only vs process rewards when time allows; document findings.

## Examples
- Input: "Add a step penalty." Output: reward code, unit tests with degenerate policies, expected score ordering.

## Constraints
- Never ship a reward without unit tests.
- Keep held-out evaluation independent of the reward function.
- Report any discovered hacking honestly.
```

### C.4 `hackathon-submission`

```markdown
---
name: hackathon-submission
description: Prepares the final hackathon submission: README, results tables, model card, demo script, pitch, and reproducibility check. Use in the packaging and submission phases.
---

# Hackathon Submission

## Goal
Make the project easy to understand, reproduce, and judge in under two minutes.

## Instructions
1. Re-read the official submission requirements on hackathon.lexsi.ai and list them in references/requirements.md. [TBD]
2. README sections: problem, approach, setup, reproduce, results (before/after table), ablations, failure analysis, reward-hacking check, limitations, licences.
3. Model card with intended use, training data, metrics, and limitations.
4. Demo script: same tasks shown with base model vs trained model.
5. Pitch outline: problem, method, results, safety angle, limitations, next steps.
6. Reproducibility check: fresh clone, follow README, confirm the headline number reproduces.
7. Output a final checklist with DONE / MISSING for each requirement.

## Constraints
- No claims beyond the evidence.
- No secrets or tokens in the repo.
- Respect dataset and model licences.
```
