# Learning Guide: How This Project Works and Why It Is Built This Way

This guide is for you, not for GitHub. Read it before you run anything, then again while you run each stage. By the end you should be able to explain every line of this project in an interview without notes. That is the real goal: the code is a scaffold, but the understanding and the numbers have to be yours.

A note on honesty before anything else. I wrote this code without being able to run it (I had no GPU or simulator), so treat it as a careful first draft. LeRobot changes often, and some flag or function name may differ in the version you install. When something breaks, fixing it is part of the project, and "I hit X, diagnosed it as Y, fixed it with Z" is one of the best stories you can tell in an interview. Only put numbers on your resume that you produced yourself.

---

## Part 1. The concepts you need

### 1.1 What a VLA is

A Vision-Language-Action model is a policy: a function from what the robot sees and what it is told to what it does next.

- **Vision:** camera images (in LIBERO, a third-person "agentview" camera and a wrist camera).
- **Language:** an instruction such as "pick up the alphabet soup and place it in the basket".
- **Action:** a vector of motor commands. In LIBERO this is 7 numbers: 6 for how the end-effector should move (position and rotation deltas) and 1 for the gripper (open/close).

The key idea behind VLAs is reuse. Instead of learning vision and language from scratch on a few thousand robot demonstrations, you start from a vision-language model (VLM) that already learned from huge amounts of internet image-text data, and attach something that outputs actions.

### 1.2 SmolVLA's architecture, in plain terms

SmolVLA has two parts:

1. **A VLM backbone (SmolVLM2).** A vision encoder turns each image into tokens; a small language model processes those image tokens together with the instruction tokens and the robot's current state (joint positions, etc.). SmolVLA keeps this cheap by using fewer visual tokens per image and by using only the lower layers of the language model, which the authors found to be enough for control.
2. **An action expert.** A smaller transformer that reads the VLM's internal features and produces a *chunk* of future actions (many timesteps at once, not just the next one).

When you run `python -m vla_study.quantize --list <checkpoint>`, you will literally see these two parts as separate groups of module names. That listing is how you choose the regexes for the quantization experiment.

### 1.3 Two ideas inside the action expert

**Action chunking.** Predicting a sequence of future actions and executing several of them before re-planning makes motion smoother and reduces a classic imitation-learning failure, *compounding error*: when a policy predicts one step at a time, small mistakes push the robot into states the demonstrations never covered, and errors snowball.

**Flow matching.** Rather than predicting the action directly (which averages out when demonstrations disagree, e.g. some people grasp from the left and some from the right), the expert learns to *transform random noise into a realistic action chunk* over a few refinement steps. This is closely related to diffusion models. Consequence for you: **SmolVLA's output is random.** Run it twice on the same input and you get slightly different actions. That is why `offline_bench.py` reseeds the random number generator before every prediction; otherwise you'd be measuring noise, not quantization error. Expect an interviewer to ask about this.

### 1.4 Behavior cloning (imitation learning)

Training is supervised learning: the dataset contains human demonstrations (image, instruction, state → action), and the model learns to reproduce the demonstrator's actions. In LIBERO each task has dozens of demonstrations. The loss going down tells you the model imitates the data well; it does **not** tell you the robot succeeds. Success is only measured by **closed-loop evaluation**: run the policy in the simulator, let its own actions change the scene, and check whether the task gets done. This gap between training loss and closed-loop success is the single most important thing to understand in robot learning.

### 1.5 LIBERO

LIBERO is a benchmark built on the MuJoCo physics simulator with a simulated Franka robot arm. It has several suites, each testing a different kind of generalization:

- `libero_spatial`: same objects, different layouts.
- `libero_object`: different target objects, the same pick-and-place skill. **This is your suite**: 10 tasks, so with `N_EP=10` you run 100 episodes per evaluation.
- `libero_goal`: same scene, different goals.
- `libero_10` (LIBERO-Long): long multi-step tasks.

We train on the combined LeRobot-format LIBERO dataset (all suites, as in the LeRobot docs) and evaluate on `libero_object`. Say this clearly in your write-up: it's a multi-task policy evaluated on one suite.

### 1.6 Quantization

A trained network stores each weight as a 32-bit or 16-bit float. Quantization stores it with fewer bits. For N-bit symmetric quantization of one row of a weight matrix:

```
qmax  = 2^(N-1) - 1                 # 127 for 8-bit, 7 for 4-bit
scale = max(|w|) / qmax
q     = round(w / scale), clipped to [-qmax-1, qmax]   # an integer
w_hat = q * scale                   # the value the model effectively uses
```

With 4 bits there are only 16 possible values per row, so a single large weight (an outlier) makes `scale` big and crushes all the small weights to zero. **Group-wise** quantization fixes this by using a separate scale for every 128 consecutive weights. That's why the scripts use `--quant-group-size 128` for 4-bit.

**Why "fake" quantization?** Real low-bit inference requires special GPU kernels (bitsandbytes, torchao, TensorRT). Those affect speed, but they also mix several effects together. Fake quantization rounds the weights and stores them back at full precision, so the model's math is exactly "the original model, but with N-bit weights". This isolates the *accuracy* cost, which is the scientific question. Memory savings are computed analytically (N bits per weight plus a 16-bit scale per group). **Never claim measured speed-ups from fake quantization.** If an interviewer asks how you'd get real speed-ups, the answer is: export with a real low-bit kernel library and benchmark on the target hardware; the accuracy result should carry over closely.

**Why test the backbone and the expert separately?** The two parts do different jobs. The backbone holds most of the parameters, so it's where memory savings come from. The action expert outputs fine-grained continuous control, which may be more sensitive to small errors. Finding out which part tolerates compression is an actual research result, and it matters in practice: you'd quantize the robust part aggressively and keep the sensitive part at higher precision ("mixed precision").

### 1.7 Language robustness and why you need a control

LIBERO instructions follow fixed templates. A model trained only on those exact sentences might fail on a natural rewording, which matters for any system where humans give instructions.

Here's the subtle part, and probably the most interesting thing in your project. Suppose success stays the same when you reword instructions. That could mean the model understands language robustly, **or** that it ignores language entirely and recognizes the task from the scene. These two explanations predict different results for the **swapped** control, where each task receives a *different* task's instruction:

- Success drops a lot under `swapped` → the policy uses language, so stable paraphrase results mean real robustness.
- Success stays high under `swapped` → the policy mostly ignores language, and the paraphrase results say little about language understanding.

Either outcome is a legitimate, reportable finding. Designing a control like this is what turns "I ran some evals" into research.

### 1.8 Statistics: don't over-read small differences

A success rate from 100 episodes is an estimate. The 95% Wilson interval for a measured 80% runs from about 71% to 87%. So 80% vs. 76% is **not** a meaningful difference at this sample size. `results.py` computes these intervals automatically. Use them in your write-up, and say "within noise" when intervals overlap heavily. Also note you train each configuration with one seed; mention it as a limitation.

---

## Part 2. What each file does, and why

**`scripts/common.sh`.** Shared settings, so every experiment uses the same suite, episode count and device. `MUJOCO_GL=egl` makes the simulator render on the GPU without a display, which servers need.

**`scripts/01_eval_baseline.sh`.** Runs *before* training. If a known-good community checkpoint scores near 0% in your setup, the problem is your environment (rendering, camera names, action space), not your training. People who skip this step can waste days training models they can't evaluate. Check on huggingface.co that the `BASELINE` repo id exists; community checkpoints get renamed.

**`scripts/02` and `03` (experiments A and B).** Identical except for one flag, `--policy.train_expert_only`. This is the experimental-design principle you should name in interviews: **change one variable at a time** and keep the seed, steps and batch size fixed.

- `--policy.type=smolvla` builds SmolVLA from its config using the dataset's camera and action layout. That avoids input mismatches you'd get by loading a checkpoint pretrained with different camera names.
- `--policy.load_vlm_weights=true` loads the pretrained SmolVLM2 backbone instead of random weights. Without it, you'd be training a VLA with no pretrained knowledge.
- `--policy.freeze_vision_encoder=true` keeps the image encoder fixed in both experiments.

If LeRobot rejects any flag, run `lerobot-train --help` (or open `src/lerobot/policies/smolvla/configuration_smolvla.py`) and look for the current name.

**`vla_study/lerobot_compat.py`.** All imports from LeRobot in one file, so version changes require editing one place. This is a good software habit worth mentioning.

**`vla_study/quantize.py`.** Implements the math from section 1.6, plus `--list`, which prints how many parameters live under each module prefix. It refuses to run if your regex matches nothing, because a silent no-op would produce a fake "quantization doesn't hurt" result.

**`vla_study/patched_eval.py`.** The cleverest file, so make sure you understand it. Instead of rewriting LeRobot's evaluation loop (and risking subtle differences), it **monkey-patches** two functions inside LeRobot's eval module before running it:

- `make_policy` → wrapped so the policy is quantized right after it loads.
- `make_pre_post_processors` → wrapped so the preprocessor swaps the instruction text before it gets tokenized.

Because LeRobot's own rollout, success counting and video saving are untouched, quantized and paraphrased runs are directly comparable to normal runs. It also **verifies itself**: at the end it prints how many instructions it rewrote, and warns loudly if the number is zero. Always read that line. If it says 0, the run was not a paraphrase test.

**`vla_study/paraphrase.py`.** Extracts the exact instruction strings from LIBERO (so keys match what the simulator sends), then builds rule-based levels and optional LLM rewrites. The script stops after generating so you can **read every paraphrase by hand**. An LLM might turn "place it in the basket" into "place it next to the basket", which changes the task and invalidates the test.

**`vla_study/offline_bench.py`.** Each closed-loop evaluation takes a long time. This script compares quantized and full-precision action predictions on 200 dataset frames in minutes, so you can rule out hopeless configurations before spending simulator time on them. Its numbers are in the model's normalized action space, which is fine for comparing configurations against each other. Present it as a screening tool, not the headline metric.

**`vla_study/results.py`.** Finds every `eval_info.json`, pulls out the success rate, adds Wilson intervals, and produces a markdown table and bar chart. It searches for the `pc_success` field instead of assuming a fixed layout, because that layout has changed between LeRobot versions.

**`vla_study/hub_sync.py`.** Cloud notebooks wipe their disk when a session ends. This script uploads the latest complete checkpoint and all results to a private Hugging Face repo, restores them at the start of the next session, and deletes old checkpoints so the disk doesn't fill. It only uploads a checkpoint once LeRobot has finished writing it (both the model and the optimizer state exist), so a backup can never capture a half-saved file.

**`tests/test_study.py`.** Unit tests for the parts that can be tested without a GPU: quantization error shrinks as bits increase, regex filtering works, paraphrase rewriting handles single strings and batches, and the confidence-interval math is correct. Having tests at all puts you ahead of most student projects.

---

## Part 3. Running it: a realistic two-week plan

Compute: Northeastern's Explorer cluster, or Kaggle's free T4 GPUs via `vla_study_kaggle.ipynb`. On a T4, start with `BS=8` and run the speed test before committing to a step count. On a smaller GPU, lower `BS` to 16. Wall-clock times vary a lot by hardware; time your first runs and adjust.

| Days | Do this | Done when |
|---|---|---|
| 1–2 | `00_setup.sh`, then `01_eval_baseline.sh` with `N_EP=2` first, then full | Baseline success is clearly non-zero and you've watched a rollout video |
| 3–5 | Start A (`02`). While it trains, read the SmolVLA paper (sections on architecture and training) | Loss curve flattens; checkpoint saved |
| 5–7 | Start B (`03`). Run `04_eval_trained.sh` on A | You have A's success rate |
| 7–9 | `quantize --list`, set regexes, `05_quant_sweep.sh` | Quant table filled in |
| 9–11 | `06_paraphrase_sweep.sh` (generate → review → evaluate) | Paraphrase + swapped results |
| 12–14 | `07_results.sh`, fill in README, write key findings, record GIFs | Repo public and pinned on your GitHub |

If time runs out, cut in this order: experiment B first, then the LLM paraphrase level, then `w4_expert`. Keep the baseline, A, the quantization sweep and the swapped control. Those carry the story. **Apply as soon as A is evaluated** and list the project as in progress.

---

## Part 4. When things break

- **0% success but training loss looks fine.** The most common failure. Check that the camera keys, image size and state/action dimensions the policy expects match what the environment provides. Watch a rollout video: a robot that doesn't move points to an action-scale or normalization problem; a robot moving toward the wrong place points to a camera mix-up or flipped image.
- **`unrecognized arguments` / unknown flag.** LeRobot renamed something. Use `--help` and check the config dataclass in the source.
- **MuJoCo / EGL / rendering errors.** Confirm `MUJOCO_GL=egl` is set and you are on a GPU node. On some clusters you need `MUJOCO_GL=osmesa` instead (slower, CPU rendering).
- **CUDA out of memory.** Lower `BS`. Experiment B uses more memory than A, since gradients flow through more layers. Record both numbers, because that difference is itself a result.
- **`patched_eval` says "no make_policy".** LeRobot's eval script imports it differently in your version. Open the eval module (`python -c "import lerobot.scripts.lerobot_eval as m; print(m.__file__)"`), find where the policy and processors are created, and patch that name instead.
- **Paraphrase run reports 0 rewrites.** The task strings from the environment don't match your file's keys. Print the unmatched strings it lists, and re-run `paraphrase extract`.

---

## Part 5. Interview preparation

Practice answering these out loud, in two minutes or less each.

**"Walk me through your VLA project."** Structure: the problem (VLAs are big; deployment needs small models that still follow instructions), what you did (fine-tuned SmolVLA on LIBERO, compared two fine-tuning strategies, studied quantization by component, tested language robustness with a control), the most interesting result, and one limitation.

**"Why freeze the backbone?"** Fewer trainable parameters means less memory, faster training, and less risk of destroying the VLM's pretrained knowledge with a small robot dataset. The cost: the backbone's features can't adapt to the robot domain. Experiment B measures that trade-off directly.

**"Why is your model's output random, and how did you handle it?"** Flow matching starts from noise. For fair comparisons you fix the noise by reseeding before each prediction; for closed-loop evaluation, you average over many episodes and report confidence intervals.

**"Your 4-bit result was X; is that significant?"** Answer with the confidence interval and the number of episodes. Being precise about uncertainty is exactly what research teams look for.

**"What would you do next?"** Real low-bit kernels to measure latency on edge hardware; more seeds; harder suites like `libero_10`; quantization-aware fine-tuning to recover lost accuracy; testing on real robot data. For Intuitive specifically: applying the same evaluation thinking (closed-loop validation, controls, uncertainty) to surgical video and instrument kinematics.

**"How does this relate to surgical robotics?"** Same building blocks: video plus language context to actions, learned from expert demonstrations, validated in simulation before hardware. Compactness matters in an operating room, where models need to run on fixed hardware with tight latency limits, and safety-critical settings demand that you know when a model is relying on the wrong signal, which is what your swapped-instruction control tests.

---

## Part 6. Resume bullets (fill in only with your own numbers)

> **Efficient Vision-Language-Action Policies for Robot Manipulation** | PyTorch, LeRobot, Hugging Face, MuJoCo
> • Fine-tuned SmolVLA (450M) on LIBERO via behavior cloning, reaching X% closed-loop success on LIBERO-Object; compared action-expert-only vs. expert + language-layer training (X% vs. Y%, Z GB less GPU memory).
> • Built a component-wise weight-quantization study (8/4-bit, group-wise): 4-bit backbone cost X points of success for an estimated Y% smaller model, while the action expert proved [more/less] sensitive.
> • Designed instruction-robustness evals with a swapped-instruction control, showing the policy [does / largely does not] ground its behavior in language; results reported with 95% confidence intervals.

Pick the two strongest bullets if space is tight. Make sure every claim is something you can defend line by line.
