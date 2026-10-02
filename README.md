# How Small Can a Robot Brain Get? Fine-tuning, Quantizing and Stress-testing SmolVLA on LIBERO

*Archita · Northeastern University · [Fill in month/year]*

> **Status:** [in progress / complete]. All numbers below come from my own runs. Placeholders like `X%` mean that run hasn't finished yet.

Vision-Language-Action (VLA) models read a camera image and a natural-language instruction, then output robot motor commands. This project studies a compact open VLA, **SmolVLA** (~450M parameters), on the **LIBERO** simulated manipulation benchmark, and asks three questions:

1. **Fine-tuning:** Is it enough to train only the small *action expert* on top of a frozen vision-language backbone, or does also adapting the language layers help?
2. **Compression:** How much closed-loop task success survives 8-bit and 4-bit weight quantization, and which part of the model (vision-language backbone vs. action expert) is most sensitive?
3. **Language grounding:** Does the policy still succeed when instructions are reworded? And, as a control, does it fail when given the *wrong* instruction (i.e., is it actually using language)?

## Results

<!-- Replace with results/results.md and results/success_rates.png after running scripts/07_results.sh -->

| Experiment | Success % (95% CI) | Notes |
|---|---|---|
| Pretrained community checkpoint (sanity check) | X [lo, hi] | validates the eval pipeline |
| A: train action expert only | X [lo, hi] | GPU memory: X GB, train time: X h |
| B: action expert + language layers | X [lo, hi] | GPU memory: X GB, train time: X h |
| A + 8-bit weights (all Linear) | X [lo, hi] | est. weights X MiB → Y MiB |
| A + 4-bit weights (all Linear, g=128) | X [lo, hi] | |
| A + 4-bit backbone only | X [lo, hi] | |
| A + 4-bit action expert only | X [lo, hi] | |
| A, reworded instructions (synonym / restructured / verbose) | X / X / X | |
| A, **swapped** instructions (control) | X [lo, hi] | low = policy uses language |

Suite: `libero_object` (10 tasks), N episodes per task. Training: 20k steps, batch 32, seed 1000, on all LIBERO suites.

**Key findings:** *(write 3 sentences after you see the numbers, e.g. "4-bit quantization of the backbone cost only X points, while quantizing the action expert cost Y, suggesting ...")*

<!-- ![success rates](results/success_rates.png) -->
<!-- ![rollout](results/gifs/....gif) -->

## Method in one paragraph

SmolVLA combines a pretrained vision-language model (SmolVLM2) with a small transformer "action expert" that generates chunks of future actions using flow matching. I trained it by behavior cloning on LIBERO's human demonstrations using Hugging Face LeRobot, and evaluated it closed-loop in the MuJoCo simulator. For quantization, I used simulated (fake) weight-only quantization: symmetric, per-channel for 8-bit and group-wise (g=128) for 4-bit. This isolates the accuracy effect of low-bit weights; memory figures are estimates of N-bit storage, not measured kernel speed-ups. For language robustness, I patched the evaluation so each instruction is replaced by a paraphrase just before tokenization, and verified that every rewrite was applied.

## Reproduce

```bash
bash scripts/00_setup.sh                 # environment (records the LeRobot commit used)
bash scripts/01_eval_baseline.sh         # sanity check: pretrained checkpoint must score well
bash scripts/02_train_expert_only.sh     # experiment A
bash scripts/03_train_expert_plus_vlm.sh # experiment B
bash scripts/04_eval_trained.sh
python -m vla_study.quantize --list outputs/train/expert_only/checkpoints/last/pretrained_model
bash scripts/05_quant_sweep.sh           # set VLM_RE / EXPERT_RE from the listing above
bash scripts/06_paraphrase_sweep.sh      # run twice: once to generate, once (after manual review) to evaluate
bash scripts/07_results.sh
python -m pytest tests -q                # unit tests, no GPU needed
```

On Colab Pro or Kaggle, use `vla_study_colab.ipynb` / `vla_study_kaggle.ipynb` instead of the shell steps above (it runs the same scripts and backs up checkpoints to a private Hugging Face repo).

LeRobot version: see `LEROBOT_VERSION.txt` (or `LEROBOT_COMMIT.txt` in the backup repo). Requires Python 3.12+. Hardware: [GPU model].

## Repository layout

```
scripts/            one shell script per experiment stage (00 → 07)
vla_study/
  quantize.py       fake weight quantization + module listing
  patched_eval.py   lerobot-eval wrapper: quantize after load / rewrite instructions
  paraphrase.py     extract LIBERO instructions, generate paraphrase + control sets
  offline_bench.py  fast simulator-free action-drift benchmark
  results.py        aggregate eval_info.json → table with Wilson CIs + chart
  hub_sync.py       back up / restore runs via a private Hugging Face repo
  lerobot_compat.py all LeRobot imports in one place
tests/              unit tests
```

## Limitations

Simulation only; one benchmark suite; a single training seed per configuration (differences inside the confidence intervals should not be over-interpreted); fake quantization measures accuracy, not real latency.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot), [SmolVLA](https://arxiv.org/abs/2506.01844) and [LIBERO](https://libero-project.github.io/).
