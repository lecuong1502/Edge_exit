# EdgeExit

**Does early exit survive the edge?** EdgeExit is a benchmark and toolkit for evaluating early-exit methods for reasoning language models on consumer/edge GPUs, under realistic deployment conditions: quantized weights, non-trivial probe overhead, and measured energy.

> **Status:** 🚧 Research in progress (target: EMNLP 2027). No results have been published yet.

## Overview
Early-exit methods (e.g., DEER, Dynasor-CoT, EAT, hidden-state probes) cut overthinking in reasoning models, but are usually evaluated on full-precision models in datacenters and report only token savings. EdgeExit asks whether those savings hold on edge hardware.

The project has three parts:
- **EdgeExit-Bench**: a unified evaluation framework with a probe-aware cost model, exact offline replay of stopping rules, and per-query energy measurement.
- **Quantization × early-exit study**: how BF16 / FP8 / INT4 / INT3 quantization affects stopping-signal calibration and savings.
- **Bounded-Entropy Probe (BEP)**: certified bounds on next-token entropy from a clustered vocabulary, falling back to the full LM head only when the bounds are inconclusive.

See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the full research plan.

## Repository structure
```
edgeexit/
├── engines/    # trace generation backends (HF, vLLM, custom CUDA kernels)
├── quant/      # quantized checkpoint builders
├── methods/    # early-exit methods with a shared interface
├── bep/        # Bounded-Entropy Probe
├── replay/     # offline replay engine
├── measure/    # NVML energy, CUDA events, thermal logging
├── eval/       # answer grading
└── analysis/   # Pareto, calibration, statistics
```

## Getting started
_Coming soon._

## Hardware
Primary target: NVIDIA RTX 4050 Laptop GPU (6 GB, Ada / sm_89).

## Language policy
All documents, code, comments, commit messages, issues, and pull requests in this repository are written in **English**.

## Citation
If you use this work, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License
Code is released under the [Apache License 2.0](LICENSE).
