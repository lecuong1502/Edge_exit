# EdgeExit

**Repository name:** `edgeexit`

**Repository description (GitHub "About"):**
> Does early exit survive the edge? A benchmark and toolkit for evaluating early-exit methods for reasoning LLMs on consumer/edge GPUs under quantization, with probe-overhead-aware latency, per-query energy measurement, and Bounded-Entropy Probe (BEP) for cheap, certified entropy-based exit decisions.

**Suggested topics:** `efficient-nlp` `early-exit` `reasoning-models` `overthinking` `quantization` `llm-inference` `energy-efficiency` `edge-ai` `cuda` `emnlp`

> **Language policy.** All documents, code, code comments, commit messages, issues, pull requests, experiment logs, and paper drafts in this repository are written in **English**. Discussion between the author and the AI assistant (chat) is conducted in **Vietnamese**; any content produced in chat that enters the repository must be written or translated into English.

---

# Does Early Exit Survive the Edge?
## Re-evaluating early exit for reasoning models under quantization, probe overhead, and measured energy

**Research report & paper plan. Target venue: EMNLP 2027 (Efficient NLP track)**
Author: Lê Kiên Cường · Version: 2026-11-22 · Status: **proposal, no experiments run yet**

> **Important note on "experimental results".** This report was written *before* any experiments were run. Section 9 presents **hypotheses, acceptance criteria, and result table templates** (empty cells to be filled in), not real data. Every specific number in this report is either (a) a number already published in another paper, with a citation, or (b) a rough estimate by the author, explicitly labeled as an estimate. No fabricated numbers may enter the paper.

---

## Table of Contents
0. Executive Summary
1. Background and Motivation
2. Related Work Survey
3. Research Gaps
4. Research Questions and Hypotheses
5. Expected Contributions
6. Method
7. System Build Plan
8. Experimental Setup
9. Experimental Results: Hypotheses, Criteria, and Table Templates
10. Analysis and Ablation Plan
11. ACL Paper Outline (8 pages) + Draft Abstract
12. Positioning Against Closest Work, Risks, and Contingency Plans
13. Limitations & Ethics (required by ACL)
14. Timeline to Deadline
15. References

---

## 0. Executive Summary

**Problem.** Reasoning models (R1-style, with a "thinking" mode) tend to overthink: they keep generating verification steps after they have already reached the correct answer. A wave of *early-exit* methods appeared in 2025–2026 to truncate the reasoning chain (DEER, Dynasor-CoT, EAT, hidden-state probes, ROM, TERMINATOR, DTSR, BLADE, SpecExit, …). However, these methods are almost always evaluated on full-precision models running on datacenter GPUs, and most of them report only **the number of tokens saved**.

**Observation.** When reasoning models are deployed on edge hardware (a 6GB laptop GPU, an edge device), three conditions change at once:
1. The model **must be quantized**. There is already evidence that aggressive quantization increases overthinking [Lotfi et al., 2026; Alimaskina, Rudas et al., 2026].
2. **Probe cost** (generating trial answers, computing entropy, running a probe head) becomes a larger share relative to decoding, which has itself become cheaper through quantization.
3. **The LM head with a ~150k vocabulary** is often kept in BF16 while the body is quantized to 4-bit, making it relatively expensive in every decode step. Entropy-based exit signals depend on exactly this component.

**Proposal.** The paper consists of three connected parts, with **energy (J/query)** and **measured latency** as cross-cutting metrics:
- **(D) EdgeExit-Bench:** a unified benchmark for 6–8 early-exit methods, measured by real latency including probe overhead, plus energy.
- **(F) Quantization × early-exit interaction analysis:** do stopping signals stay calibrated, do thresholds transfer from BF16 to INT4/FP8, and do the savings compound with quantization?
- **(G) Bounded-Entropy Probe (BEP):** a way to compute entropy signals for early exit without running the full LM head. It uses certified upper/lower bounds based on vocabulary clustering, a fused CUDA kernel, and falls back to the full head only when the bounds are inconclusive.

**Why it is likely to be accepted.** This is a "systematic re-evaluation, with findings and a small method" paper. The contributions are independent of one another: even if BEP yields only a modest improvement, the benchmark and analysis can stand on their own as a Findings paper.

---

## 1. Background and Motivation

### 1.1. Overthinking in reasoning models
The survey by Sui et al. defines "overthinking" as long, redundant reasoning chains that incur large computational cost, and groups mitigation approaches into three categories: model-based (retraining), output-based (inference-time intervention), and prompt-based [1]. Early exit belongs to the second category and is attractive because it does not require retraining the backbone.

The strongest empirical motivation comes from DEER: the authors' analysis shows that about 75% of samples contain at least one point where stopping early still yields the correct answer, and more than one third of samples need less than half of the original reasoning chain [2].

### 1.2. Edge hardware changes the problem
- **Quantization is mandatory.** A 4B model in BF16 needs about 8GB for weights alone, exceeding the 6GB VRAM of an RTX 4050 Laptop GPU.
- **Quantization interacts with overthinking.** Liu et al. (2025) find that at W4G128 or W8A8KV8, output length stays nearly unchanged relative to BF16, but at more aggressive settings (W3G128, W4A4KV4) models generate longer outputs [3]. Lotfi et al. (2026, NeurIPS oral) go further: up to 52% of the errors of quantized models occur because the model *found the correct answer midway but failed to commit to it* [4]. Alimaskina, Rudas et al. (2026) show that at 2-bit, loops and "commitment failures" increase total token count to the point that per-token speed gains no longer hold end-to-end [5].
- **Consequence:** on edge hardware, early exit is *even more necessary*. Yet no one has checked whether existing early-exit methods still work correctly once the model is quantized.

### 1.3. Measuring in tokens is not enough
- The closest unified benchmark is EffiReason-Bench. It evaluates only instruct models (no R1-style reasoning models), reports only output token counts, and does not measure latency, energy, or probe overhead [6].
- OckBench measures accuracy and token count jointly and introduces the notion of an "Overthinking Tax", but still uses token count as the primary efficiency measure [7].
- SpecExit notes that probe-based methods such as DEER incur detection overhead that reduces their real latency benefit [8]. This shows that *tokens ≠ time*, and that probe overhead is a variable that must be measured directly.

---

## 2. Related Work Survey

### 2.1. Early-exit / anti-overthinking methods

We categorize methods by **the source of the stopping signal**, because the signal source determines *the type of overhead*. This is also the main analysis axis of the paper.

| Method | Stopping signal | Training? | Main overhead source | What the original paper measures | Notes |
|---|---|---|---|---|---|
| **DEER** [2] (ICLR 2026) | At transition points (e.g., the "Wait" token), force the model to produce a trial answer and compute its confidence | No | Extra trial-answer generation (many tokens) + logits | Length, accuracy; proposes branch-parallel decoding and dynamic KV management | Widely used baseline |
| **Dynasor-CoT / Certaindex** [9][10] | Probe-in-the-middle: periodically insert a prompt to extract an answer; stop when the answer is stable across probes | No | Periodic probes (prefill + a few tokens) | Tokens; Certaindex includes a serving system and reports compute and throughput savings | DEER notes Dynasor stops late due to its conservative condition [2] |
| **EAT** [11] | Entropy of the next-token distribution after appending `</think>`; stop when the (EMA) variance of the entropy trajectory is small enough | No | 1 forward + **softmax over the full vocabulary** per probe | Tokens (12–22% reduction on MATH500, AIME2025); no latency reported | **Directly depends on the LM head**; the target of Part G |
| **Hidden-state probe** (Zhang et al.) [12] | Linear/MLP probe on hidden states predicting whether the intermediate answer is correct | Yes (small probe) | Near zero (reads existing hidden states) | Tokens (~24% reduction) | Shows that hidden states encode correctness information |
| **LYNX** [13] | Hidden-state probe at cues ("hmm", "wait") + conformal calibration | Yes (probe) | Low | Tokens, accuracy; 1.5B–32B | Probe trained once, reused across tasks |
| **ROM** [14] | Detection head on last-layer hidden states, running in streaming fashion per token | Yes (head) | Low | Tokens, accuracy, **reports reduced wall-clock latency** | Qwen3-8B, R1-Distill-32B |
| **BLADE** [15] | Multi-level checkpoints (sentence, self-doubt, paragraph) + adaptive selection of layers to probe | Yes | Low–medium | Tokens, accuracy; 2 Qwen3 models | |
| **TERMINATOR** [16] | Predicts the position where the final answer first appears | Yes | Low | 14–55% length reduction; reports **>2× latency reduction** | |
| **DTSR** [17] | Detects reflection signals, then checks "is this sufficient to answer?" | No (self-assessment) | Sufficiency check (cost unclear) | 28.9–34.9% length reduction; no latency reported | Qwen3 |
| **SpecExit** [8] | Draft model (EAGLE-style) additionally predicts exit signals from hidden states | Yes (draft) | Tied to speculative decoding | 66% length reduction, **2.5× latency** vs. speculative-decoding baseline | Requires a draft model; not a fair comparison at the edge |
| **NoWait** [18] | Suppresses reflection tokens ("Wait", "Hmm", "Alternatively") | No | ~0 | 27–51% length reduction | *Steering* family, not a pure exit method |
| **Logit penalty** (Lotfi et al.) [4] | Subtracts a penalty λ from the logits of 50 overthinking markers | No | ~0 | 12–23% CoT reduction on quantized models | Steering, designed specifically for quantized models |
| **Budget forcing / NoThinking** [19][20] | Hard cut at a budget / skip the thinking phase | No | 0 | Tokens, accuracy | Mandatory "naive" baselines |

**Summary observations:**
1. There are two method families with **fundamentally different cost structures**: (i) methods that *use logits or generate extra tokens* (DEER, Dynasor, EAT, DTSR), whose overhead scales with LM-head cost and the number of probe tokens; (ii) methods that *use hidden-state probes* (Zhang, LYNX, ROM, BLADE, TERMINATOR), whose overhead is negligible but which require training and may be sensitive to shifts in the hidden-state distribution under quantization.
2. **None of the methods in the table has been evaluated on a quantized model.** This is the central gap for Part F (see Section 3).
3. Only a few (ROM, TERMINATOR, SpecExit) report latency, and all measure on datacenter hardware with BF16 models.

### 2.2. Quantization for reasoning models
- **Liu et al. (2025), "Quantization Hurts Reasoning?"** [3]: the first systematic study, on R1-Distill-Qwen/LLaMA 1.5B–70B and QwQ-32B, covering weight, KV, and activation quantization. W8A8 and W4A16 are considered nearly lossless. Harder tasks and RL-trained models are affected more. Output length increases only at very low bit-widths.
- **Lotfi et al. (2026), "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not"** [4]: five models (1.5B–32B), GPTQ/AWQ at 3–4 bits, FlatQuant W4A4KV4/W8A8KV8. A notable example: Qwen-1.5B with 3-bit AWQ on MATH-500 drops from 85.6% to 47.0% accuracy, while CoT length grows from 5.2K to 23.4K tokens. The KL divergence between the quantized and full-precision models concentrates at high-entropy positions, where "wait"-type markers are sampled more often. The proposed method is a logit penalty. **From the portion of the paper we read, it mentions early exit only in related work, does not evaluate any early-exit method, and does not report latency, energy, or hardware** [4].
- **Alimaskina, Rudas et al. (2026), "Extreme Low-Bit Inference in Reasoning Models"** [5]: Qwen3-8B/32B, GPTQ W4A16 and W2A16. They separate two failure modes: failing to find a solution path, and failing to commit to an answer. They propose two interventions: *FP16 planning* and *loop rescue* (detect repeated n-grams, commit to an existing answer or rerun in FP16). End-to-end speed is *derived* from throughput times token count rather than measured directly, and no energy numbers are reported. In the portion we read, the paper does not compare directly against DEER or Dynasor (the appendix should be re-checked to confirm) [5].
- **Quantization foundations** (for choosing configurations): GPTQ [21], AWQ [22], SmoothQuant [23], FP8 formats [24], KV-cache quantization (e.g., KIVI [25]).

### 2.3. LM-head cost and large vocabularies
- **FR-Spec** (ACL 2025) [26] shows that the LM head is an overlooked bottleneck in speculative sampling with large vocabularies. The authors restrict drafting to a high-frequency token subset, cutting LM-head computation by 75% and achieving an average 1.12× speedup over EAGLE-2.
- **VocabTrim** (Qualcomm, ICML 2025 workshop) [27] highlights that the LM head accounts for a large share of size and latency, especially in small models with large vocabularies. For example, in a 314M drafter using the Llama 3 vocabulary (128K), the LM head accounts for more than 30% of parameters.
- Follow-up work, DynaSpec [28] and MicroSpec [29], builds dynamic or context-dependent shortlists. Ben Shoham [30] analyzes the trade-off between coverage and draft latency.
- **FlashHead** (2026) [31] is the closest work to Part G and must be read carefully. The authors report that the LM head can account for up to 60% of parameters and 50% of inference compute in small models. FlashHead clusters embeddings (spherical k-means) and then performs two-stage IVF-style retrieval, achieving up to 1.75× model-level speedup (Llama-3.2-1B, INT4, batch 1, RTX 3500 Ada). **Key limitation:** FlashHead does not provide a closed-form distribution over the full vocabulary; the authors must estimate probabilities via Monte Carlo for tasks that need likelihoods [31]. Therefore FlashHead *cannot directly compute entropy*, which is exactly what EAT and confidence-based signals need.
- **Top-k entropy approximation** has been used in several settings: decoding control [32], hallucination detection with tail bounds [33], accuracy monitoring [34], and uncertainty estimation [35]. However, these works **assume the full logits are already available** (the full LM head has been run) or only have top-k from an API. To the best of our search, *no prior work estimates entropy with certified bounds without computing the full LM head, for the purpose of early-exit decisions*. This is the gap for Part G (arXiv must be re-scanned before submission).

### 2.4. Measuring inference energy
- **Samsi et al. (2023), From Words to Watts** [36]: one of the early studies of the energy cost of LLaMA inference on V100/A100.
- **ML.ENERGY Benchmark** (NeurIPS 2025) [37]: derives design principles for energy benchmarks. The authors point out that measuring *energy per token* ignores differences in output length. This is the crux for reasoning models: early exit reduces *the number of tokens*, so we must measure **energy per query**.
- **Jin et al. (2025), The Energy Cost of Reasoning** [38]: analyzes the accuracy–energy trade-off of test-time compute and reiterates that prefill is compute-bound while decode is memory-bandwidth-bound.
- **Yang et al., "Part-time Power Measurements"** [39]: warns that `nvidia-smi` samples power during only part of the runtime (e.g., about 25% on A100/H100), which can cause large estimation errors. Our energy protocol must address this (Section 6.5).

### 2.5. Critiques of efficiency measurement
- Dehghani et al., "The Efficiency Misnomer" [40]: any single cost indicator (FLOPs, parameter count, throughput) fails to fully capture efficiency, and indicators can contradict each other. This is the conceptual basis for reporting tokens, latency, VRAM, and energy together.

---

## 3. Research Gaps

| # | Gap | Evidence | Addressed by |
|---|---|---|---|
| **G1** | No unified evaluation of early-exit methods *for reasoning models* with **real latency including probe overhead** and **energy** | EffiReason-Bench uses only instruct models and measures tokens [6]; OckBench measures tokens [7]; most original papers do not report latency (EAT [11], DTSR [17]) | D |
| **G2** | No one has evaluated **early-exit methods on quantized models** | Lotfi et al. do not evaluate early exit [4]; Alimaskina, Rudas et al. only have their own loop rescue, with no comparison to DEER/Dynasor found [5]; all early-exit papers use BF16 | F |
| **G3** | Unclear whether **stopping signals remain calibrated** under quantization: entropy, confidence, and hidden states are all affected by quantization noise | Lotfi et al. show divergence concentrates at high-entropy positions [4], exactly the region read by entropy-based signals (EAT) | F |
| **G4** | Entropy-based signals need the **full LM head**, which becomes relatively expensive once the body is quantized; approximate LM heads (FlashHead) do not provide the full distribution | FlashHead [31]; FR-Spec/VocabTrim apply only to drafting [26][27] | G |
| **G5** | "Lower energy" claims are based on token counts and are not measured directly on edge hardware | ML.ENERGY stresses per-query rather than per-token measurement [37]; `nvidia-smi` under-samples [39] | I (cross-cutting) |

**One-sentence gap statement (for the Introduction):**
> Early-exit methods are designed and evaluated for full-precision models on datacenter GPUs using token counts, whereas the most realistic deployment setting for small reasoning models (edge hardware) requires quantization, incurs non-trivial probe costs, and cares about energy. Whether these methods remain effective under such conditions is unknown.

---

## 4. Research Questions and Hypotheses

### RQ1 (D): How much is actually saved?
*When measured by end-to-end latency (including probe overhead) and energy per query, how much of the token savings reported in the original papers do early-exit methods retain? Does the ranking among methods change?*

- **H1a:** The ratio (latency savings / token savings) is < 1 for logit/probe-based methods (DEER, Dynasor, EAT), and ≈ 1 for hidden-state-based methods (ROM, LYNX, linear probe).
- **H1b:** The ranking by tokens differs from the ranking by energy (Kendall τ < 0.8).
- **H1c:** At least one simple baseline (fixed budget, or stopping when the answer is stable) lies on the accuracy–energy Pareto front in some configurations.

### RQ2 (F): What does quantization do to early exit?
*How does quantization (INT8, FP8, INT4 weight-only, W3) change CoT length, overthinking rate, and the reliability of stopping signals? Do thresholds tuned at BF16 transfer to quantized models? Do savings from early exit and from quantization compound?*

- **H2a (calibration drift):** For entropy/confidence-based signals, the AUROC for predicting "stopping here is still correct" decreases as bit-width decreases, and thresholds tuned at BF16 cause systematically *premature* or *late* exits on quantized models.
- **H2b (complementarity):** At aggressive quantization (W3, W4A4), because overthinking increases [4][5], the *relative* savings of early exit are **larger** than at BF16. That is, early exit and quantization are complementary, not merely additive.
- **H2c (probe transfer):** Hidden-state probes trained at BF16 lose effectiveness when applied to INT4. Re-calibration (threshold adjustment only, no retraining) recovers most of the loss.

### RQ3 (G): Can entropy signals be made cheaper while staying correct?
*Can entropy-based exit decisions be made without running the full LM head, using certified upper/lower bounds and falling back only when the bounds are inconclusive? How much overhead is saved, and do the decisions match those made with the full head?*

- **H3a:** With a reasonable vocabulary clustering, most probes (hypothesis: > 80%) can be decided from the bounds alone, without fallback.
- **H3b:** Decision agreement with the full head is ≥ 99%, and final accuracy shows no statistically significant difference.
- **H3c:** EAT's probe overhead decreases substantially, visible in end-to-end latency and energy.

---

## 5. Expected Contributions

1. **EdgeExit-Bench:** a framework and protocol for evaluating early exit for reasoning models on edge hardware, comprising (i) a *cost model* that separates probe overhead from decoding, (ii) an exact *offline replay protocol* for pure stopping rules, and (iii) a per-query energy measurement protocol that corrects for sampling error. Code and traces released.
2. **The first empirical study of quantization × early-exit interaction:** 6–8 methods × 4–5 precision levels × 3 models × 4 benchmarks, with findings on calibration drift and complementarity.
3. **Bounded-Entropy Probe (BEP):** a method for computing entropy signals for early exit with certified bounds, using vocabulary clustering, with fallback and a fused CUDA kernel.
4. **Practical recommendations** for deploying reasoning models at the edge: which precision × exit-method combination to choose for each energy budget.

---

## 6. Method

### 6.1. Cost model of early exit

For a query $q$ with a full reasoning chain of $L$ tokens, an exit method stops at $\tau \le L$ and performs $K$ probes:

$$
T_{\text{total}}(q) = T_{\text{prefill}}(|q|) + \sum_{t=1}^{\tau} T_{\text{dec}}(t) + \sum_{k=1}^{K} T_{\text{probe},k} + T_{\text{answer}}
$$

where:
- $T_{\text{dec}}(t)$: the cost of decoding one token at context length $t$. On edge GPUs, this is dominated by the number of weight and KV bytes read, so it depends strongly on precision.
- $T_{\text{probe},k}$: the cost of the $k$-th probe. We classify probes as:
  - **(P1) Generation-based probes** (DEER, Dynasor, DTSR): prefill of the inserted text + trial-answer generation.
  - **(P2) Logit-based probes** (EAT): 1 forward for `</think>` + softmax/entropy over the vocabulary $|V|$.
  - **(P3) Hidden-state probes** (ROM, LYNX, linear probe): a small MLP.

Energy has the same structure: $E_{\text{total}} = \int P(t)\,dt$ over the entire query-processing window, measured directly (Section 6.5).

**Primary metrics:**
- **Retention ratio** $\rho = \dfrac{\Delta T_{\text{latency}}}{\Delta T_{\text{token}}}$: the fraction of token savings retained when converted into real time. $\rho = 1$ means probing is free; $\rho < 1$ means probe overhead has eaten part of the savings.
- **Accuracy–Energy Pareto front**, and the area under the Pareto curve (provisionally named AE-AUC) to compare methods with a single number.

### 6.2. Offline replay protocol

**Problem:** 8 methods × 5 precisions × 3 models × 4 benchmarks is too many to run online on a single laptop GPU.

**Solution:** split methods into two groups.
- **Pure stopping rules:** do not alter the tokens generated before the stopping point. This includes DEER (branch-parallel variant, with trial answers on a separate branch), Dynasor (probes do not affect the main generation, as described by the authors [9]), EAT, hidden-state probes, TERMINATOR, and budget forcing.
  → With the same seed and sampling parameters, the method's output = **a prefix of the full trace** + the answer generated after stopping. Therefore we only need to **generate the full trace once** per (model, precision, query, seed), then *replay* the stopping rule over the checkpoints. At each checkpoint, run the probe on that prefix (with the prefix's KV cached) and generate the final answer.
- **Steering methods:** alter the generation distribution (NoWait [18], logit penalty [4]). These run online, but only a few configurations are needed since they serve as comparison baselines.

**Replay correctness check:** on a subset of about 50 queries × 2 configurations, run real online inference and compare against replay results. Requirement: ≥ 99% answer agreement, and exact stopping-position agreement under deterministic sampling.

> Technical note: GPU kernels may be non-deterministic (floating-point summation order), so regenerated traces may diverge. Mitigation: replay always uses **stored traces** and never regenerates. Online runs are used only to measure latency and energy (Section 6.5) and to confirm that result distributions are equivalent.

### 6.3. Precision configurations

| Code | Description | Role |
|---|---|---|
| BF16 | Reference | Runnable only for models ≤ ~2B on 6GB; 4B models run BF16 on a rented GPU (subset of data) |
| W8A16 / FP8 (W8A8-FP8) | Nearly lossless per [3] | Practical reference for 4B; Ada (sm_89) has FP8 tensor cores |
| W4A16-AWQ, W4A16-GPTQ | Most common edge configuration | Main axis |
| W3A16 | Aggressive quantization, increased overthinking [3][4] | Tests H2b |
| (optional) KV8 / KV4 | KV-cache quantization | If time permits |

**An easily overlooked detail (linking F and G):** in many quantization toolchains, **the LM head is often kept at high precision** and not quantized together with the body. This must be verified against the default configuration of AutoAWQ/GPTQ or whichever tool is used. If so, once the body drops to 4-bit, the LM head's *share* of bytes read per decode step increases sharply.

*Rough estimate (to be measured):* for a ~1.5B model with hidden size 1536 and a ~152K vocabulary (the Qwen3 tokenizer has 151,669 tokens [49]; the LM-head matrix in `config.json` is often padded slightly larger, so take the exact size from the config), the LM head ≈ 233M parameters. In BF16, the LM head ≈ 0.47GB, while the ~1.3B-parameter body at 4-bit ≈ 0.65GB. The LM head could therefore account for **~40% of weight bytes per decode step**. If confirmed by measurement, this would be a strong motivation figure for Part G.

### 6.4. Proposed method: Bounded-Entropy Probe (BEP)

**Goal:** decide *"is the entropy $H$ at the probe position below threshold $\theta$?"* (or decide on an EMA statistic of entropy, as in EAT) without computing the full $W h$, where $W \in \mathbb{R}^{|V|\times d}$.

**Idea:**
1. **Offline:** cluster the rows of $W$ (output embeddings) into $C$ clusters, e.g., with spherical k-means as in FlashHead [31]. Each cluster $c$ stores its centroid $\mu_c$, radius $r_c = \max_{i\in c}\lVert w_i - \mu_c\rVert$, and size $n_c$.
2. **Online, at each probe:**
   - Compute centroid logits: $s_c = \mu_c^\top h$ (cheap, since $C \ll |V|$).
   - By Cauchy–Schwarz, every token $i$ in cluster $c$ has logit $z_i \in [s_c - r_c\lVert h\rVert,\; s_c + r_c\lVert h\rVert]$.
   - Select a set of "hot" clusters $\mathcal{A}$ (top by upper bound) and **compute exact** logits for the tokens in $\mathcal{A}$.
   - For the remaining clusters, use the interval bounds to obtain **upper and lower bounds on the log-partition** $\log Z$, and from these derive **bounds on the tail probability mass** and **bounds $[H_{\min}, H_{\max}]$** on the entropy. The upper bound on tail entropy uses a "uniform over the remainder" argument, similar to [33], but applied here to *tokens that were never computed*.
   - **Decision:** if $H_{\max} < \theta$, exit; if $H_{\min} > \theta$, continue; if the bounds are inconclusive, **expand $\mathcal{A}$** or **fall back to the full head**.
3. **Fused CUDA kernel:** a single kernel performs gather of the rows in $\mathcal{A}$ → GEMV → online log-sum-exp (the same idea as the online softmax already implemented in the author's NanoInfer engine) → entropy accumulation → bound computation, without writing logits to global memory.

**Properties to prove or verify:**
- *Soundness:* whenever BEP makes a decision (without fallback), that decision **matches the full-head decision** (true by construction if the bounds are correct; numerical error under FP16/FP32 accumulation must be accounted for).
- *Cost:* $O(Cd + |\mathcal{A}|\,d)$ versus $O(|V|\,d)$, plus the fallback cost at frequency $p_{fb}$. A break-even point in terms of $p_{fb}$ can be computed.

**Differences from prior work:**
- Versus FlashHead [31]: BEP does not replace the LM head for sampling; it serves only the *decision signal*, and has **certified bounds with fallback**, so it never changes the output.
- Versus FR-Spec/VocabTrim [26][27]: those methods apply to drafting; BEP applies to early-exit probes.
- Versus top-k entropy [32]–[35]: those methods need full logits to select the top-k; BEP does not.

**Extension (if time permits):** apply the same bounding technique to DEER's *confidence* signal (the probability of the answer tokens), which only requires bounds on a few specific probabilities and is even easier than entropy.

### 6.5. Energy and latency measurement protocol

1. **Energy counter:** prefer reading NVML's cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`) before and after each query, instead of integrating sampled power. Check whether the laptop GPU supports it. If not, sample power at the highest possible frequency, run sufficiently long workloads, and repeat many times to reduce the sampling error described by Yang et al. [39]. The Zeus/ML.ENERGY library can serve as a reference [37].
2. **Baseline correction:** measure idle power and either subtract it or report both (total and incremental).
3. **Fixed laptop conditions:** plugged in, fixed power mode, GPU clocks locked if possible (`nvidia-smi -lgc`), temperature logged. Laptops throttle easily, so each configuration is run ≥ 3 times with interleaved ordering, reporting the median with confidence intervals.
4. **CPU:** if probing or answer parsing runs on the CPU (e.g., regex extraction of `\boxed{}`), additionally measure CPU energy via RAPL if readable, or state explicitly that it is excluded.
5. **Latency:** batch size 1 (edge scenario), using CUDA events for each phase (prefill, decode, probe, answer) to separate overhead exactly as in the cost model of Section 6.1.
6. **Online measurement set:** 100 queries (stratified by difficulty) for each selected combination. The full combination matrix is not run online.

---

## 7. System Build Plan

### 7.1. Proposed repository structure
```
edgeexit/
├── engines/            # trace generation & measurement backends
│   ├── hf_backend.py   # HF transformers: reference implementation, hidden-state hooks
│   ├── vllm_backend.py # fast batched trace generation (if quant formats are supported)
│   └── nanoinfer/      # custom kernels: INT8 GEMM, online softmax, BEP kernel
├── quant/              # scripts to build AWQ/GPTQ/FP8/W3 checkpoints + perplexity checks
├── traces/             # generated traces (jsonl + token ids + logprobs at checkpoints)
├── methods/            # each exit method = one class with a shared interface
│   ├── base.py         # interface: checkpoints(trace) -> decide(state) -> exit?
│   ├── budget.py, nothinking.py, answer_consistency.py
│   ├── deer.py, dynasor.py, eat.py, dtsr.py
│   ├── probe_linear.py, lynx.py, rom.py
│   └── steering/ nowait.py, logit_penalty.py   # run online
├── bep/                # Bounded-Entropy Probe: clustering, bounds, fallback
├── replay/             # offline replay engine + replay-vs-online checks
├── measure/            # NVML energy, CUDA events, thermal logging
├── eval/               # answer grading (math-verify / boxed matching), pass@1
└── analysis/           # Pareto, AE-AUC, Kendall tau, calibration (AUROC, ECE)
```

### 7.2. Build steps in order

| Step | Task | Verifiable output |
|---|---|---|
| B1 | Set up one model (R1-Distill-Qwen-1.5B), generate BF16 traces on MATH-500 | Accuracy close to published numbers [3][4] (small deviations due to seed/config differences) |
| B2 | Build W4-AWQ, W4-GPTQ, W3, FP8 checkpoints; measure perplexity and accuracy | Accuracy degradation trend consistent with [3][4] |
| B3 | Implement `methods/base.py` + 3 simple baselines + EAT; offline replay | Accuracy-vs-token curves |
| B4 | Replay-vs-online check on 50 queries | ≥ 99% answer agreement |
| B5 | Implement DEER, Dynasor (prefer original code [2][10]), linear probe [12], LYNX/ROM (if code is public [14]) | Reproduce the trends of the original papers at BF16 |
| B6 | Energy measurement module + CUDA events; run one full configuration | Run-to-run variation < ~5% (target) |
| B7 | Scale to 3 models × 5 precisions × 4 benchmarks (batched trace generation) | RQ1 and RQ2 tables |
| B8 | BEP: clustering, bounds, fallback (Python/PyTorch first); measure decision rates | No-fallback rate, decision agreement rate |
| B9 | Fused CUDA kernel for BEP + comparison against full head (cuBLAS GEMV + softmax) | Probe latency, ncu profile |
| B10 | End-to-end online measurement for selected combinations | RQ3 table + final Pareto |

### 7.3. Compute budget estimate (rough, to be confirmed by a pilot)
- RTX 4050 Laptop: 6GB, spec memory bandwidth about 192 GB/s (96-bit bus). The theoretical upper bound on batch-1 decode speed ≈ bandwidth / bytes read per token.
- Lotfi et al. report an average CoT of ~5.2K tokens for Qwen-1.5B BF16 on MATH-500 [4], so one MATH-500 pass ≈ 2.6M generated tokens.
- With batched generation (a few concurrent sequences, limited by KV cache on 6GB), if aggregate throughput reaches a few hundred tokens/s, each (model, precision) pass on MATH-500 takes on the order of a few hours. The full matrix (3 models × 5 precisions × 4 benchmarks, plus AIME with multiple seeds) is estimated at **about 80–150 GPU-hours**, which is feasible within 2–3 months running overnight.
- **A rented GPU is needed** for BF16 runs of the 4B model and for cross-checking on a second GPU (to show the conclusions are not specific to one machine). Estimated at about 20–40 GPU-hours on an L4/A10/A100-class GPU.

---

## 8. Experimental Setup

### 8.1. Models (proposed, finalized after the pilot)
| Model | Rationale |
|---|---|
| **DeepSeek-R1-Distill-Qwen-1.5B** | Used in both [3] and [4], enabling direct comparison; distilled reasoning model |
| **Qwen3-4B-Thinking-2507** (or Qwen3-1.7B thinking) | Used in SpecExit [8]; RL-trained, which [3] shows is more sensitive to quantization |
| **LFM2.5-1.2B-Thinking** (Liquid AI, 01/2026) | Reasoning model targeting edge devices [41]; architecture differs from the Qwen family, testing generality |

*(If newer and more popular small reasoning models are available by December 2026, e.g., small Qwen3.5 or Gemma 4 variants, the third model should be replaced. Check the license and the availability of a thinking variant.)*

### 8.2. Compared methods
- **Baselines:** Full CoT; budget forcing (multiple budget levels) [19]; NoThinking [20]; stop when the answer is stable (simple Dynasor-style).
- **Early exit (pure stopping):** DEER [2], Dynasor-CoT [9][10], EAT [11], linear probe [12], LYNX [13], ROM [14]. Add TERMINATOR [16] and BLADE [15] if code is available.
- **Steering (online, few configurations):** NoWait [18], logit penalty [4].
- **Excluded from the main comparison:** SpecExit [8] (requires a separately trained draft model, a different axis). Discussed in the Discussion section.

### 8.3. Benchmarks
- **GSM8K** [42]: easy; where overthinking is most visible.
- **MATH-500** [43][44]: the common standard in nearly all related papers.
- **AIME 2025 (and AIME 2026 if available)**: hard; run multiple seeds since each year has only 30 problems.
- **GPQA-Diamond** [45]: science, outside the math domain.
- *(optional)* **LiveCodeBench** [46]: code, testing generality beyond math.

### 8.4. Metrics
- **Quality:** pass@1 (averaged over seeds), plus the "overthinking error" rate as defined in [4] (the correct answer appears midway but the final answer is wrong).
- **Efficiency:** token count (thinking + answer), probe token count, end-to-end latency, **energy (J/query)**, peak VRAM.
- **Stopping-signal quality:** AUROC and ECE of the "is stopping here correct?" signal, and *exit regret* (excess tokens relative to the oracle-optimal stopping point).
- **Aggregate:** retention ratio ρ, Pareto front, AE-AUC.
- **Statistics:** bootstrap 95% CIs; paired tests (e.g., McNemar) for accuracy between method pairs; Kendall τ for ranking changes.

### 8.5. Generation settings
- Use each model's recommended sampling settings (e.g., temperature 0.6, top-p 0.95 for the R1-distill family), with a 32K length limit. Record all parameters.
- Fixed seeds; AIME uses ≥ 4 seeds.
- Tune each method's threshold on a **held-out dev set** (e.g., part of the MATH training split), never on the test set.

---

## 9. Experimental Results: Hypotheses, Criteria, and Table Templates

> **No data yet.** The tables below are *templates* to be filled in after the experiments are run. The "Expected" column contains hypotheses only. If the data refute a hypothesis, report it honestly; that is a finding too.

### Table 1: RQ1, tokens vs. latency vs. energy (model X, BF16 or W8, MATH-500)
| Method | Acc (%) | Tokens ↓ | Probe tokens | Latency (s) ↓ | Energy (J) ↓ | ρ | Expected |
|---|---|---|---|---|---|---|---|
| Full CoT | | | 0 | | | — | reference |
| Budget (best) | | | 0 | | | ≈1 | surprisingly strong baseline? (H1c) |
| DEER | | | | | | | ρ < 1 (H1a) |
| Dynasor-CoT | | | | | | | ρ < 1 |
| EAT | | | | | | | ρ < 1, LM-head dependent |
| Linear probe | | | ~0 | | | | ρ ≈ 1 |
| LYNX / ROM | | | ~0 | | | | ρ ≈ 1 |

**Criterion supporting H1a:** the 95% CI of ρ for the logit/generation-based family lies entirely below 1, while that of the hidden-state family contains 1.

### Table 2: RQ2, quantization × early exit (each cell: Δacc / Δenergy relative to Full CoT at the same precision)
| Method \ Precision | BF16 | FP8 | W4-AWQ | W4-GPTQ | W3 |
|---|---|---|---|---|---|
| Full CoT (acc, tokens, J) | | | | | |
| DEER | | | | | |
| EAT | | | | | |
| Linear probe (trained at BF16) | | | | | |
| Linear probe (re-calibrated) | | | | | |
| NoWait | | | | | |
| Logit penalty [4] | | | | | |

**Criterion for H2a:** signal AUROC decreases monotonically with bit-width, and BF16-tuned thresholds yield significantly higher exit regret or false-early-exit rates at W3/W4.
**Criterion for H2b:** the % energy savings of early exit at W3 exceed those at BF16, with non-overlapping CIs.

### Table 3: RQ3, BEP vs. full head for the EAT signal
| Configuration | No-fallback decision rate | Agreement with full head (%) | Probe latency (µs) | Δ E2E latency | Δ E2E energy | Acc |
|---|---|---|---|---|---|---|
| Full head (cuBLAS + softmax) | — | 100 | | 0 | 0 | |
| Top-k after full head (reference [32]) | — | | | | | |
| BEP, C = 256 | | | | | | |
| BEP, C = 1024 | | | | | | |
| BEP + fused kernel | | | | | | |

**Criterion for H3:** no-fallback rate ≥ 80%, decision agreement ≥ 99%, no significant accuracy difference (McNemar p > 0.05), and a clear reduction in probe latency.

### Planned figures
1. **Figure 1 (teaser):** accuracy–energy Pareto, comparing "by tokens" vs. "by measured energy". Rankings change.
2. **Figure 2:** time breakdown (prefill / decode / probe / answer) as stacked bars per method.
3. **Figure 3:** LM-head share of bytes per decode step as a function of body precision (motivation for Part G).
4. **Figure 4:** calibration (reliability diagram / AUROC) of stopping signals by precision.
5. **Figure 5:** tightness of BEP bounds: distribution of $H_{\max} - H_{\min}$ and fallback rate as a function of C.

---

## 10. Analysis and Ablation Plan

- **By difficulty:** split queries by MATH level or by BF16 CoT length. Check whether early exit helps mainly on easy queries (overthinking) or hard ones.
- **By overhead source:** separate analysis of P1/P2/P3, and vary probe frequency (every N tokens vs. only at cues).
- **Threshold transfer:** BF16-tuned thresholds → applied to W4/W3; per-precision tuned thresholds; tuning with 50 dev queries.
- **Overthinking markers:** re-check the findings of [4] in our setup (frequency of "wait" by precision) as a sanity check.
- **BEP:** vary the number of clusters C, the strategy for expanding $\mathcal{A}$, and FP16 vs. FP32 accumulation (which affects bound soundness).
- **Hardware:** repeat a subset on a different (rented) GPU to show the ρ conclusions are not specific to one machine.
- **Batching:** add batch sizes 4–8 for a few configurations, since ML.ENERGY stresses that batch size strongly affects energy [37]. This is kept separate from the batch-1 edge scenario.

---

## 11. ACL Paper Outline (long paper, 8 content pages)

| Section | Pages | Content |
|---|---|---|
| Abstract | — | ~200 words |
| 1. Introduction | 1.0 | Overthinking → early exit → edge deployment changes 3 conditions → research questions → contributions → Figure 1 |
| 2. Background & Related Work | 0.9 | Early exit (categorized by signal source), quantized reasoning ([3][4][5]), LM head ([26][27][31]), energy measurement ([37][39]) |
| 3. EdgeExit-Bench | 1.3 | Cost model, overhead taxonomy, offline replay and its correctness check, energy protocol |
| 4. Experimental Setup | 0.7 | Models, precisions, methods, benchmarks, metrics |
| 5. RQ1: Do token savings translate? | 1.0 | Table 1, Figure 2 |
| 6. RQ2: Early exit under quantization | 1.3 | Table 2, Figure 4, calibration drift and complementarity findings |
| 7. RQ3: Bounded-Entropy Probe | 1.3 | Method, bounds, kernel, Table 3, Figures 3 and 5 |
| 8. Discussion & Recommendations | 0.5 | Guidance on choosing combinations by budget |
| Limitations | (not counted) | Section 13 |
| Ethics | (not counted) | Section 13 |
| Appendix | unlimited | Per-method details, hyperparameters, full results, replay checks, bound proofs |

### Draft abstract (with placeholders for numbers)
> Early-exit methods promise to curb overthinking in reasoning language models, yet they are designed and evaluated on full-precision models in datacenter settings, and mostly report token savings. We ask whether these savings survive *on the edge*, where models must be quantized, probing costs are non-negligible, and energy matters. We introduce **EdgeExit-Bench**, a framework that separates probing overhead from decoding cost, enables exact offline replay of stopping rules, and measures per-query energy on a consumer laptop GPU. Evaluating [N] early-exit methods across [M] precisions, three reasoning models and four benchmarks, we find that [only X% of token savings translate into energy savings for probe-based methods], that [quantization shifts the calibration of stopping signals, making thresholds tuned at BF16 exit too early/late], and that [early exit and quantization are complementary at low bit-widths]. Finally, we propose **Bounded-Entropy Probe**, which computes certified bounds on next-token entropy from a clustered vocabulary and falls back to the full LM head only when the bounds are inconclusive, reducing probe cost by [Y×] with [Z%] decision agreement and no significant accuracy loss. Code and traces will be released.

---

## 12. Positioning Against Closest Work, Risks, and Contingency Plans

### 12.1. Positioning (for Related Work and reviewer responses)
| Closest work | What they do | How we differ |
|---|---|---|
| Lotfi et al. 2026 [4] | Quantization → more overthinking; logit penalty; token-only measurement | We evaluate **existing early-exit methods** on quantized models, measure **real latency and energy**, and analyze **stopping-signal calibration**. Their logit penalty is a baseline in Table 2 |
| Alimaskina, Rudas et al. 2026 [5] | 2-bit: loop rescue + FP16 planning; speed derived from throughput | We use a realistic precision range (FP8/W4/W3), compare against many exit methods, and measure directly |
| EffiReason-Bench [6], OckBench [7] | Token-based efficient-reasoning benchmarks | We focus on reasoning models, with probe overhead, quantization, energy, and edge hardware |
| FlashHead [31] | Replaces the LM head with cluster-based retrieval for decoding | BEP serves only the **decision signal**, with **certified bounds and fallback**, and never changes the output |
| Top-k entropy [32]–[35] | Approximate entropy after logits are available | BEP does not need full logits |

### 12.2. Main risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A new paper (2027) covers exactly the D+F direction | Medium | High | Scan arXiv every 2 weeks; if overlapped, emphasize Part G and edge energy measurement as differentiators |
| Methods' code is hard to reproduce | High | Medium | Prioritize methods with original code (DEER, Dynasor, ROM); reimplement from the paper otherwise and document in the appendix |
| BEP bounds too loose, frequent fallback | Medium | Medium | Try smaller clusters or per-token bounds for hot clusters; if still loose, report honestly and turn Part G into an LM-head overhead analysis plus an approximate top-k variant (without guarantees) |
| Unstable energy measurement on a laptop | Medium | Medium | Lock clocks, repeat runs, report CIs; cross-check on a rented GPU |
| Running out of time | High | High | Priority order: RQ1 → RQ2 → RQ3. If late, reduce RQ3 to an overhead analysis (no new method); RQ1+RQ2 still suffice for Findings |
| BF16 for the 4B model cannot run locally | Certain | Low | Use W8/FP8 as the practical reference (nearly lossless per [3]); run BF16 on a rented GPU for a subset |

---

## 13. Limitations & Ethics (required by ACL)

**Limitations (draft):**
- Only small reasoning models (≤ 4B) and one main consumer GPU are evaluated. Conclusions about ρ may differ on NPUs, CPUs, or datacenter GPUs at large batch sizes.
- Benchmarks focus on math and science, like most related studies. Open-ended, agentic, and conversational tasks are not tested.
- Offline replay is exact only for pure stopping rules. Steering methods are evaluated in fewer configurations.
- Energy is measured for the GPU only (and the CPU if possible), not the whole system (display, RAM).
- Reimplemented methods (without original code) may differ from the originals.

**Ethics:**
- Only public benchmarks are used; no personal data.
- Benefits: lower inference energy and better feasibility of running models locally (privacy protection).
- Risks: early exit may cause the model to stop early and answer incorrectly with high confidence. The false-early-exit rate must be reported, not just average accuracy.
- Disclose the use of AI assistants in writing the paper, per ACL policy. **Manually verify every citation:** at least one study has found hundreds of papers at ACL/NAACL/EMNLP 2024–2025 with fabricated citations (HalluCitation), with the most at EMNLP 2025 [47].

---

## 14. Timeline to Deadline (ARR, expected late May 2027)

| Period | Milestone | Completion criterion |
|---|---|---|
| 2026-11-20 – 2026-12-15 | Read the 15 core papers in depth; B1–B2 | Reproduce R1-Distill-1.5B BF16 and W4 accuracy on MATH-500 |
| 2026-12-16 – 2027-01-15 | B3–B6: framework, replay, energy measurement | Table 1 v1 on one model |
| 2027-01-16 – 2027-02-28 | B7: scale to the full matrix | Tables 1 and 2 complete; **go/no-go decision for Part G** |
| 2027-03-01 – 2027-04-10 | B8–B9: BEP + kernel | Table 3 |
| 2027-04-11 – 2027-04-30 | B10 + ablations + cross-check on another GPU | All final figures and tables |
| 2027-05-01 – 2027-05-20 | Write the paper, get external feedback | Complete draft ≥ 1 week before the deadline |
| ~2027-05-28 | ARR submission | — |

**Go/no-go milestone (2027-02-28):** if by then RQ1 and RQ2 have produced no significant findings (e.g., ρ ≈ 1 for all methods, and quantization does not affect calibration), shift the focus to RQ3 and an analysis of LM-head overhead at the edge. Conversely, if RQ1 and RQ2 are strong, RQ3 may be narrowed to protect quality.

---

## 15. References

> All references below have been checked for title, authors, and arXiv ID against their source pages (updated 2026-11-19). When moving them into the paper, take BibTeX directly from the ACL Anthology or arXiv.

**Early exit / overthinking**
- [1] Y. Sui et al. *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* TMLR 2025. arXiv:2503.16419. https://arxiv.org/abs/2503.16419
- [2] C. Yang, Q. Si et al. *Dynamic Early Exit in Reasoning Models (DEER).* ICLR 2026. arXiv:2504.15895. https://arxiv.org/abs/2504.15895 · code: https://github.com/iie-ycx/DEER
- [8] R. Yang et al. *SpecExit: Accelerating Large Reasoning Model via Speculative Exit.* arXiv:2509.24248. https://arxiv.org/abs/2509.24248
- [9] Y. Fu et al. *Reasoning Without Self-Doubt: More Efficient Chain-of-Thought Through Certainty Probing (Dynasor-CoT).* ICLR 2025 Workshop FM-Wild. https://mlanthology.org/iclrw/2025/fu2025iclrw-reasoning/ · blog: https://hao-ai-lab.github.io/blogs/dynasor-cot
- [10] Y. Fu et al. *Efficiently Scaling LLM Reasoning Programs with Certaindex.* NeurIPS 2025. arXiv:2412.20993. https://arxiv.org/abs/2412.20993 · code: https://github.com/hao-ai-lab/Dynasor
- [11] X. Wang, J. McInerney, L. Wang, N. Kallus. *Entropy After </Think> for Reasoning Model Early Exiting (EAT).* arXiv:2509.26522. https://arxiv.org/abs/2509.26522
- [12] A. Zhang et al. *Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification.* arXiv:2504.05419. https://arxiv.org/abs/2504.05419
- [13] *LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning.* arXiv:2512.05325. https://arxiv.org/abs/2512.05325
- [14] X. Wang, X. Liu, M. Pei, C. Xiao. *ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention.* arXiv preprint, arXiv:2603.22016 (v1, 03/2026, has 3 authors without Ming Pei; v4, 10/2026, has 4 authors; neither version states a venue. Cite the latest version). https://arxiv.org/abs/2603.22016 · code: https://github.com/SaFo-Lab/ROM
- [15] K. Fu et al. *BLADE: Boundary-Expanded and Layer-Adaptive Dynamic Exit for Efficient LLM Reasoning.* arXiv:2607.28966. https://arxiv.org/abs/2607.28966
- [16] A. Nagle et al. *TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning.* arXiv:2603.12529. https://arxiv.org/abs/2603.12529
- [17] Y. Xiang et al. *When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning (DTSR).* arXiv:2604.06787. https://arxiv.org/abs/2604.06787
- [18] C. Wang et al. *Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency (NoWait).* arXiv:2506.08343. https://arxiv.org/abs/2506.08343
- [19] N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei et al. *s1: Simple test-time scaling* (budget forcing). arXiv:2501.19393, 2025. https://arxiv.org/abs/2501.19393
- [20] W. Ma et al. *Reasoning Models Can Be Effective Without Thinking (NoThinking).* arXiv:2504.09858. https://arxiv.org/abs/2504.09858

**Efficient-reasoning benchmarks**
- [6] J. Huang et al. *EffiReason-Bench: A Unified Benchmark for Evaluating and Advancing Efficient Reasoning in LLMs.* arXiv:2511.10201. https://arxiv.org/abs/2511.10201
- [7] *OckBench: Measuring the Efficiency of LLM Reasoning.* arXiv:2511.05722. https://arxiv.org/abs/2511.05722

**Quantization for reasoning models and foundations**
- [3] R. Liu et al. *Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models.* arXiv:2504.04823. https://arxiv.org/abs/2504.04823 · code: https://github.com/ruikangliu/Quantized-Reasoning-Models
- [4] S. Lotfi, P. Kirichenko, S. Li, Z. Liu. *Quantized Reasoning Models Think They Need to Think Longer, but They Do Not.* NeurIPS 2026 (oral). arXiv:2606.00206. https://arxiv.org/abs/2606.00206
- [5] E. Alimaskina, D. Rudas, D. Shveykin, G. Molodtsov, P. Vasiliev, A. Beznosikov. *Extreme Low-Bit Inference in Reasoning Models: Failure Modes and Targeted Recovery.* NeurIPS 2026. arXiv:2606.02011. https://arxiv.org/abs/2606.02011
- [21] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323. https://arxiv.org/abs/2210.17323
- [22] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen et al. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys 2024 (Best Paper). arXiv:2306.00978. https://arxiv.org/abs/2306.00978
- [23] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023 (PMLR 202). arXiv:2211.10438. https://arxiv.org/abs/2211.10438
- [24] P. Micikevicius, D. Stosic, P. Judd, J. Kamalu, S. Oberman et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022. https://arxiv.org/abs/2209.05433
- [25] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024 (PMLR 235). arXiv:2402.02750. https://arxiv.org/abs/2402.02750

**LM head / large vocabularies / entropy approximation**
- [26] W. Zhao et al. *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling.* ACL 2025. arXiv:2502.14856. https://aclanthology.org/2025.acl-long.198
- [27] R. Goel et al. *VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs.* ICML 2025 Workshop ES-FoMo. arXiv:2506.22694. https://arxiv.org/abs/2506.22694
- [28] J. Zhang et al. *DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models.* arXiv:2510.13847. https://arxiv.org/abs/2510.13847
- [29] Z. Chen et al. *MicroSpec: Accelerating Speculative Decoding with Lightweight In-Context Vocabularies.* arXiv:2605.26444. https://arxiv.org/abs/2605.26444
- [30] O. Ben Shoham. *Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding.* arXiv:2603.05210. https://arxiv.org/abs/2603.05210
- [31] W. Tranheden et al. *FlashHead: Efficient Drop-in Replacement for the Classification Head in Language Model Inference.* arXiv:2603.14591. https://arxiv.org/abs/2603.14591
- [32] *Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention* (section on top-k entropy approximation). arXiv:2603.03310. https://arxiv.org/abs/2603.03310
- [33] *Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate* (tail bound for top-K entropy). arXiv:2509.04492. https://arxiv.org/abs/2509.04492
- [34] P. Memoli Buffa, L. Del Corro. *Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM.* arXiv:2601.09001. https://arxiv.org/abs/2601.09001
- [35] M. Nguyen, S. Gupta, H. Le. *Probabilities Are All You Need: A Probability-Only Approach to Uncertainty Estimation in LLMs.* arXiv:2511.07694. https://arxiv.org/abs/2511.07694

**Energy and efficiency measurement**
- [36] S. Samsi et al. *From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference.* IEEE HPEC 2023. arXiv:2310.03003. https://arxiv.org/abs/2310.03003
- [37] J.-W. Chung et al. *The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization.* NeurIPS 2025. arXiv:2505.06371. https://arxiv.org/abs/2505.06371 · Zeus: https://github.com/ml-energy/zeus
- [38] Y. Jin, G.-Y. Wei, D. Brooks. *The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute.* arXiv:2505.14733. https://arxiv.org/abs/2505.14733
- [39] Z. Yang, K. Adámek, W. Armour. *Part-time Power Measurements: nvidia-smi's Lack of Attention.* arXiv:2312.02741. https://arxiv.org/abs/2312.02741
- [40] M. Dehghani, A. Arnab, L. Beyer, A. Vaswani, Y. Tay. *The Efficiency Misnomer.* ICLR 2022. arXiv:2110.12894. https://arxiv.org/abs/2110.12894

**Models and benchmarks**
- [41] Liquid AI. *LFM2.5-1.2B-Thinking: On-Device Reasoning Under 1GB* (01/2026). https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb
- [42] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun et al. *Training Verifiers to Solve Math Word Problems* (GSM8K). arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168
- [43] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart et al. *Measuring Mathematical Problem Solving With the MATH Dataset.* NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.03874. https://arxiv.org/abs/2103.03874
- [44] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker et al. *Let's Verify Step by Step.* arXiv:2305.20050, 2023. https://arxiv.org/abs/2305.20050 (origin of MATH-500: 500 problems sampled uniformly at random from the MATH test split and held out for evaluation; see Appendix C. The paper does not use the name "MATH-500".)
- [45] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang et al. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022, 2023. https://arxiv.org/abs/2311.12022
- [46] N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* arXiv:2403.07974, 2024. https://arxiv.org/abs/2403.07974
- [48] DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948, 2025 (v2: 01/2026). https://arxiv.org/abs/2501.12948
- [49] Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388

**Research integrity**
- [47] *HalluCitation in NLP venues* (analysis of fabricated citations at ACL/NAACL/EMNLP 2024–2025). ACL 2026. https://aclanthology.org/2026.acl-long.2189.pdf · arXiv:2601.18724

**Additional related work to read:** DiffAdapt (ICLR 2026, arXiv:2510.19669); *Think Faster Than Words* (ACL 2026, https://aclanthology.org/2026.acl-long.1330.pdf); *Don't Overthink It: A Survey of Efficient R1-style LRMs* (arXiv:2508.02120); *The Zero-Step Thinking* (arXiv:2510.19176); CGRS (AAAI, https://ojs.aaai.org/index.php/AAAI/article/view/40379); Kara (arXiv:2607.01237); *When Reasoning Meets Compression* (2025).

---

### Note on copyright and citation practice in this report
All descriptions of other work have been **paraphrased in our own words**, with no verbatim passages copied. Numbers are taken from the original papers and always carry a reference number. When writing the paper, continue to paraphrase, cite with `\citet`/`\citep` using BibTeX from the ACL Anthology or arXiv, and do not copy figures or tables from other papers (if needed, redraw them from published numbers and credit the source).
