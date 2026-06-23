<div align="center">

# 🔨 Hephaestus

**A fine-tuning & post-training framework — forge any base model into a specialist.**

Config-driven SFT, LoRA/QLoRA, and preference alignment in one consistent interface.

<sub>Part of the *Building it from 0 to 100* series · [@adarsh-menon28](https://www.linkedin.com/in/adarsh-menon28)</sub>

</div>

---

## What it is

Hephaestus is a lightweight framework for the **post-training** stage of the LLM lifecycle — everything that happens after pretraining to turn a base model into something useful: supervised fine-tuning, parameter-efficient adaptation, and preference alignment.

The goal is one consistent, config-driven workflow across techniques, so switching from a full SFT run to QLoRA to a DPO pass is a config change, not a rewrite.

> **Status:** early / build-in-public. APIs will move. Watch the repo to follow along.

---

## Focus: fine-tuning & post-training techniques

| Stage | Technique | Status |
|---|---|---|
| Supervised fine-tuning | Full SFT | 🔨 building |
| | LoRA | 🔨 building |
| | QLoRA (4-bit NF4) | 🔨 building |
| Preference alignment | DPO | 🗺️ planned |
| | ORPO | 🗺️ planned |
| | GRPO | 🗺️ planned |
| Quantization | 4-bit / 8-bit (bitsandbytes) | 🔨 building |
| | Adapter merge → fp16 export | 🗺️ planned |

The through-line is the post-training pipeline: **SFT → preference alignment → quantize → merge → export**, each step reproducible from a single config.

---

## Why

Most fine-tuning code is copy-pasted notebooks that drift apart — different prompt formatting, different trainer args, different save logic per project. Hephaestus standardizes that:

- **One config, one command.** Declare model, dataset, technique, and hyperparameters in YAML; run the same CLI regardless of method.
- **Technique-agnostic.** SFT, LoRA, QLoRA, and alignment methods share the same data and training interface.
- **Reproducible.** Every run is fully described by its config — no hidden state in a notebook cell.
- **Lean.** Thin layer over `transformers`, `peft`, `trl`, and `bitsandbytes` — not a reinvention.

---

## Install

```bash
git clone https://github.com/your-username/hephaestus.git
cd hephaestus
pip install -e .
```

Requires Python 3.10+ and a CUDA GPU for 4-bit training.

---

## Quickstart

Define a run in a config file:

```yaml
# configs/sft_lora.yaml
model:
  name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  load_in_4bit: true            # QLoRA
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: float16

technique:
  method: lora                  # full | lora | qlora | dpo | orpo | grpo
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

data:
  dataset: Abirate/english_quotes
  split: "train[:100]"
  prompt_template: gemma        # gemma | zephyr | chatml | custom

train:
  epochs: 1
  batch_size: 1
  grad_accum: 4
  lr: 2.0e-4
  output_dir: ./outputs/quote-tagger

export:
  push_to_hub: false
  hub_repo: your-username/tinyllama-quote-tagger
  merge_and_save: true
```

Then forge:

```bash
hephaestus train --config configs/sft_lora.yaml
```

Run inference against a trained adapter:

```bash
hephaestus infer --adapter ./outputs/quote-tagger --prompt "Tag this quote: ..."
```

---

## Project structure



---

## Roadmap

- [ ] SFT + LoRA/QLoRA training loop
- [ ] Prompt-template registry (gemma, zephyr, chatml, custom)
- [ ] Push-to-Hub + auto model card generation
- [ ] DPO / ORPO preference alignment
- [ ] GRPO
- [ ] fp16 merge & export path (clean, non-4-bit merge)
- [ ] Eval hooks (per-technique metrics)

---

## Contributing

Build-in-public — issues, ideas, and PRs welcome. Open an issue to discuss a technique or recipe you'd like to see.

## License

MIT.
