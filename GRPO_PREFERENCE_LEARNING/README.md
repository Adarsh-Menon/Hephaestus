# GRPO in Practice

A hands-on notebook for **Group Relative Policy Optimization** — the RL algorithm behind DeepSeek-R1 — running end-to-end on a free Colab T4.

Two complete training runs: a from-scratch walkthrough on a toy dataset, then the real pattern on GSM8K math problems with verifiable rewards.

---

## What GRPO actually is

Classic RLHF (PPO) needs a **value/critic network** roughly the size of the policy — expensive to train, expensive to hold in memory. GRPO deletes it.

For each prompt it samples a **group** of `G` completions, scores them all, and uses the group's own statistics as the baseline:

```
A_i = (r_i − mean(r_1 … r_G)) / std(r_1 … r_G)
```

Completions that beat their siblings get pushed up; ones that lose get pushed down. A KL penalty against the frozen reference model stops the policy drifting into gibberish.

**The consequence that matters:** in GRPO the *reward function is the training data*. You aren't labelling outputs — you're writing code that grades them. Most of this notebook is reward design.

---

## What's in the notebook

| | **Part 1** | **Part 2** |
|---|---|---|
| Model | SmolLM-135M-Instruct | Qwen2.5-0.5B-Instruct (4-bit) |
| Stack | TRL + PEFT | Unsloth + vLLM |
| Data | 8 hand-written Q&A pairs | GSM8K (8.5K math word problems) |
| Reward | similarity to a reference | **verifiable** — is the number right? |
| Rollouts | HF `generate` | vLLM (5–10× faster) |
| Steps | 10 (smoke test) | 100 (demo; 250+ for real) |
| Runtime | ~2 min | ~20–40 min on a T4 |

**Part 1** is deliberately a toy. Similarity-to-reference rewards teach imitation, not reasoning — that's a bad idea in production. It exists so every moving part is visible: reward function signatures, `GRPOConfig`, LoRA, the training loop.

**Part 2** is the pattern you'd actually ship. Format rewards bootstrap the model into producing structured output, then a verifiable correctness reward takes over.

---

## Setup

**Requires a GPU.** In Colab: Runtime → Change runtime type → T4 GPU.

```bash
# Part 1
pip install -U datasets transformers trl peft accelerate bitsandbytes
pip uninstall -y torchao        # conflicts with the TRL/PEFT stack

# Part 2
pip install -U unsloth vllm
```

Run cells top to bottom. **Restart the runtime after each install cell** — Unsloth in particular patches Transformers at import time and won't take effect on an already-imported module.

---

## Reward function reference

Every reward function has the same contract:

```python
def my_reward(prompts, completions, **kwargs) -> list[float]
```

Any extra dataset column arrives as a keyword argument named after the column, holding a list aligned with `completions`. **Always accept `**kwargs`** — TRL passes things you may not use, and a strict signature raises `TypeError`.

### Part 1 — three graders, summed

| Function | Range | Grades |
|---|---|---|
| `correctness_reward` | 0 → 5 | Blend of sequence similarity and keyword recall against the reference |
| `helpfulness_reward` | −2.5 → 1.5 | Length window, sentence-like punctuation, hedge-phrase penalty |
| `clarity_reward` | −1.5 → 1.0 | Readability guardrail; catches degenerate output |

Correctness is worth ~3× the others, so the model prioritizes it. Reward weighting is your main design lever — there's no separate loss to tune.

**Helpers:** `normalize_text` (lowercase + collapse whitespace), `get_completion_text` (unwraps str or chat-format completions), `similarity_score` (order-sensitive), `keyword_overlap_score` (order-insensitive recall).

### Part 2 — a difficulty ladder

| Function | Max | Grades |
|---|---|---|
| `xmlcount_reward_func` | ~0.5 | Fractional credit (0.125) per correctly-placed tag |
| `soft_format_reward_func` | 0.5 | Tags present in the right order, whitespace-tolerant |
| `strict_format_reward_func` | 0.5 | Exact whitespace-perfect layout |
| `int_reward_func` | 0.5 | The answer field contains a bare integer |
| `correctness_reward_func` | **2.0** | The number is **right** |

**Why five instead of one?** A lone correctness reward is *sparse*. Early on a 0.5B model gets nearly every problem wrong, every completion in the group scores 0.0, group std is 0, advantages are 0, and **no gradient flows**. Training never starts.

The format rewards are dense — they fire on almost every rollout. The model first learns to emit well-formed XML, *then* to put a number in it, *then* to get the number right. It's curriculum learning encoded into the reward function.

---

## Hyperparameters that actually matter

| Parameter | Why it matters |
|---|---|
| **`num_generations`** | The `G` in "group" — this *is* the algorithm. With `G=1` there's no group to compare against and the advantage is undefined. 4–8 typical, 8–16 for hard tasks. Cost scales linearly. |
| **`per_device_train_batch_size`** | **Must be divisible by `num_generations`.** The trainer builds batches out of whole groups; a partial group has no valid baseline. |
| **`beta`** | KL penalty weight. Too low → the model drifts and forgets language. Too high → it can't move. Recent TRL defaults to `0.0`; a small non-zero value is safer while learning. |
| **`temperature`** (rollouts) | Keep it **higher than inference** (0.9 vs 0.7). GRPO needs diverse completions. Low temperature → all `G` samples near-identical → group std collapses → advantages go to zero. |
| **`max_grad_norm`** | Set aggressively (0.1) in Part 2. A single high-advantage rollout can produce a huge gradient and wreck the policy. Standard GRPO stability fix. |
| **`gpu_memory_utilization`** | Unsloth/vLLM only. Caps the KV-cache pool. **Lower to 0.5 on OOM**, raise toward 0.7 for faster rollouts. The single most useful knob when things won't fit. |
| **`learning_rate`** | 1e-6 to 1e-5. RL is unstable; this is not the place for 2e-4. |

---

## Reading the training logs

TRL logs each reward function separately as `rewards/<function_name>`. In Part 2 the order in which they rise tells the whole story:

1. `xmlcount_reward_func` climbs — learning tags
2. `soft_format` / `strict_format` — layout locking in
3. `int_reward_func` — putting numbers in the answer block
4. `correctness_reward_func` — actually solving problems (slowest)

**Two diagnostics:**

- **`reward_std` → 0** means all completions in each group have become identical. Advantages vanish, learning stops. Fix with more rollout diversity (higher temperature) or more generations per group.
- **Formats saturated but correctness flat** means the model learned to *look* right without reasoning. Train longer, raise the correctness weight, or use a bigger base model.

Per-function logging is also how you catch **reward hacking**. If `helpfulness_reward` climbs while `correctness_reward` flatlines, the model found the length bonus and stopped trying to be right.

---

## Fixes applied to the original notebook

Four genuine breakages were found and corrected. Three of them would have stopped execution.

| # | Where | Problem |
|---|---|---|
| 1 | `SYSTEM_PROMPT`, `XML_COT_FORMAT`, `extract_xml_answer`, `soft_format_reward_func`, `strict_format_reward_func`, `count_xml` | **All `<reasoning>` / `<answer>` XML tags were missing** — stripped from every string and regex in Part 2. Classic symptom of pasting code out of a rendered HTML page, which eats anything in angle brackets. Left as-is, `extract_xml_answer` splits on the empty string, format rewards match everything unconditionally, and correctness can never fire. |
| 2 | `correctness_reward_func` | Built the `rewards` list and **never returned it** → returned `None` → `trainer.train()` crashes on step 1. |
| 3 | `soft_format_reward_func` | Final list comprehension **missing its closing bracket** → `SyntaxError`. |
| 4 | `xmlcount_reward_func` | Same **missing closing bracket** → `SyntaxError`. |

A debug print of one rollout per batch was also added to `correctness_reward_func` — reading real generations during training is how you notice reward hacking before the curves tell you.

---

## Where to go next

**Make the run real**
- `max_steps` → 250–500. 100 is a smoke test.
- `num_generations` → 8. Lower-variance advantages, linearly more compute.
- Swap in Qwen2.5-1.5B or 3B-Instruct on an A100.
- Evaluate on the GSM8K **test** split — training reward is not accuracy.

**Ablations that teach the algorithm**
- Set `num_generations=2` and watch training destabilize — that's the group baseline doing its job.
- Set `beta=0.0` and watch how fast the policy drifts.
- Delete `xmlcount_reward_func` and see whether training ever gets started.

**Write your own rewards**

The transferable skill is reward design. The pattern that works: **one verifiable signal** (tests pass, schema validates, answer matches) plus **dense format shaping** to bootstrap. Good targets — code that must pass unit tests, tool calls that must match a JSON schema, SQL that must return the right rows, extraction that must hit an exact field.

**Papers**
- DeepSeekMath — arXiv 2402.03300 — introduced GRPO
- DeepSeek-R1 — arXiv 2501.12948 — GRPO at scale, emergent reasoning

---

## Files

```
GRPO_PRACTICAL.ipynb    # the notebook — 18 lessons, Part 1 + Part 2
README.md               # this file
```

**Artifacts produced by a run:**

```
grpo_output/            # Part 1 checkpoints
grpo_output_final/      # Part 1 LoRA adapter + tokenizer
grpo_outputs/           # Part 2 checkpoints
grpo_saved_lora/        # Part 2 LoRA adapter
grpo_merged_model/      # Part 2 standalone 16-bit model
```

Adapters are a few MB and require the base model to load. Merged models are full-size (~1GB for 0.5B in fp16) but load with plain `AutoModelForCausalLM` and serve directly under vLLM/TGI. Merge for deployment; keep adapters for iteration.
