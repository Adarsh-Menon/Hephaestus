# LoRA & QLoRA — Concept + Colab NoteBook Explanation

A reference for the `LoRA_QLoRA_Practical_Implementation` notebook: the theory in plain terms, then a step-by-step of what each part of the Colab actually does.

*Building it from 0 to 100 · @Adarsh-Menon*

---

## 1. The problem LoRA solves

A 1.1B–70B parameter model has billions of weights. Full fine-tuning means updating **all** of them, which needs:

- enough GPU memory to hold the weights **plus** their gradients **plus** optimizer state (roughly 3–4× the model size), and
- a fresh full-size copy of the model saved per task.

For most people that's not feasible on a single consumer/Colab GPU. LoRA makes it feasible.

---

## 2. What LoRA is

**LoRA (Low-Rank Adaptation)** freezes the original model weights and trains a small pair of "side" matrices next to chosen layers instead.

The intuition: the *change* a fine-tune makes to a big weight matrix `W` (call it `ΔW`) doesn't need to be full-rank. It can be approximated by two skinny matrices multiplied together:

```
ΔW ≈ B · A        where  A is (r × d),  B is (d × r),  and r is tiny (e.g. 16)
```

During training:

- `W` (the original weight) stays **frozen**.
- Only `A` and `B` are trained.
- At inference the layer computes `W·x + (B·A)·x`.

Because `r` is small, `A` and `B` together have a *fraction* of a percent of the original parameter count. You train far fewer weights, use far less memory, and the result — the **adapter** — is just a few MB.

### Key hyperparameters

| Param | What it controls | In this notebook |
|---|---|---|
| `r` (rank) | Size of the bottleneck. Higher = more capacity + more params. | 16 |
| `lora_alpha` | Scaling applied to the update (`alpha/r`). Higher = stronger adapter influence. | 32 |
| `lora_dropout` | Dropout on the adapter path; light regularization. | 0.05 |
| `target_modules` | Which layers get adapters. Usually the attention projections. | `q_proj, k_proj, v_proj, o_proj` |

A common rule of thumb: set `alpha = 2 × r`. That's exactly what's used here (32 = 2×16).

---

## 3. What QLoRA adds

**QLoRA = Quantization + LoRA.** It's LoRA, but the frozen base model is loaded in **4-bit** instead of 16-bit. That roughly quarters the memory the base model occupies, so a model that wouldn't fit now does.

The pieces:

- **4-bit NF4 quantization** — weights stored in a 4-bit "NormalFloat" format designed for normally-distributed weights.
- **Double quantization** — the quantization constants are themselves quantized, saving a bit more.
- **fp16 compute dtype** — math is still done in 16-bit; only *storage* is 4-bit.
- The LoRA adapters on top are trained in normal precision.

Net effect: you fine-tune a model that's too big for the GPU at full precision, by storing it compressed and only training the small adapters.

```
QLoRA = [ frozen base in 4-bit ]  +  [ trainable LoRA adapters ]
```

---

## 4. What the Colab does, step by step

### Setup
- Installs `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, `trl`.
- Imports the quantization config, LoRA tooling, and the `SFTTrainer`.
- Selects the base model: **TinyLlama-1.1B-Chat**.

### Data preparation
- Loads the `Abirate/english_quotes` dataset.
- `format_example` turns each row into one training string: a user turn asking for tags, then the model turn containing the real tags — wrapped in chat tokens.
- Slices to the **first 100 examples** for a fast demo run.
- Maps the formatter across the rows and prints a couple to eyeball them.

### Model loading (the "Q")
- Loads the tokenizer; sets `pad_token = eos_token` so batching works.
- Builds the `BitsAndBytesConfig`: load in **4-bit**, **NF4**, fp16 compute, **double quant**.
- Loads the base model in 4-bit with `device_map="auto"`.

### LoRA setup
- Defines `LoraConfig` (r=16, alpha=32, dropout=0.05, attention projections).
- Wraps the base model with `get_peft_model`.
- `print_trainable_parameters()` confirms only the tiny adapter is trainable — the billions of base weights are frozen.

### Training
- `TrainingArguments`: batch size 1, gradient accumulation 4 (so effective batch = 4), LR 2e-4, 1 epoch.
- Builds the `SFTTrainer` over the model + dataset and runs `trainer.train()`.

### Save the adapter
- Saves the adapter and tokenizer to `./TinyLlama/adapter`. This is the few-MB artifact — not the whole model.

### Inference
- Reloads the adapter on top of the base model with `PeftModel.from_pretrained`.
- Builds a test prompt, tokenizes it, and generates with sampling (`temperature=0.7`, `top_p=0.9`, 50 new tokens).
- Decodes and prints the generated tags.

### Merge & export
- `merge_and_unload()` folds the adapter weights into the base weights, producing a single standalone model.
- Saves the merged model + tokenizer to `./merged_full_model`.

### Push to Hugging Face Hub
- Authenticate with a write token (Colab Secrets → `HF_TOKEN`).
- **Option A (recommended):** push just the adapter — `model.push_to_hub(repo)`.
- **Option B:** push the full merged model.
- Load back from the Hub: reload the 4-bit base and attach the adapter with `PeftModel.from_pretrained(base, repo)`, then generate.

---

## 5. The mental model end to end

```
base model (frozen)
      │
      ├── load in 4-bit  ............... QLoRA: shrinks memory
      │
      ├── attach LoRA adapters  ........ only these train
      │
      ├── SFTTrainer on formatted data . the fine-tune
      │
      ├── save adapter (few MB)  ....... portable artifact
      │
      ├── merge_and_unload()  .......... fold adapter into weights
      │
      └── push_to_hub  ................. share / serve
```

---

## 6. Gotchas worth knowing

- **Prompt format is part of the model.** This run uses a **Gemma-style** template (`<bos>`, `<start_of_turn>`) on TinyLlama (which natively uses a Zephyr-style `<|user|>` format). It trains fine because SFT only sees raw text — but inference *must* use the same format the adapter was trained on, or output degrades.
- **`skip_special_tokens=True`** when decoding, or the output keeps the `<bos>` / `<end_of_turn>` markers. (Easy to mistype.)
- **f-string indentation** leaks into training text — keep the content of a triple-quoted template left-aligned so you don't bake leading spaces into every example.
- **Merging on a 4-bit base** can shift generations slightly due to rounding. For a pristine merged checkpoint, reload the base in fp16 (no bnb config), attach the adapter, then merge.
- **Demo scale.** 100 examples × 1 epoch is for learning the mechanics, not quality. Scale data and epochs for anything real.

---

## 7. Quick glossary

- **PEFT** — Parameter-Efficient Fine-Tuning; the family LoRA belongs to.
- **Adapter** — the small trained LoRA weights (`A`, `B`); what you save and share.
- **NF4** — 4-bit NormalFloat, the quantization format QLoRA uses.
- **SFT** — Supervised Fine-Tuning; training on input→output pairs.
- **`merge_and_unload`** — combine adapter + base into one standalone model.
- **rank (`r`)** — the size of the low-rank bottleneck; the main capacity knob.
