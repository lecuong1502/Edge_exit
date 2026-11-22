# Contributing to EdgeExit

Thank you for your interest in contributing! This is an active research project, so please read these guidelines before opening an issue or pull request.

## Language
All contributions (code, comments, docs, commit messages, issues, PRs) must be written in **English**.

## Ways to contribute
- Report bugs or reproducibility problems
- Add or improve an early-exit method implementation
- Improve measurement accuracy (latency, energy)
- Improve documentation

## Development workflow
1. Fork the repository and create a branch from `main`:
   `feat/<short-name>`, `fix/<short-name>`, or `exp/<short-name>` (experiments).
2. Keep each pull request focused on one change.
3. Run tests and linters before submitting (see below).
4. Open a pull request using the PR template.

## Commit messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(methods): add EAT stopping rule
fix(measure): correct idle-power subtraction
docs: update replay protocol
exp: add W3 traces for R1-Distill-1.5B
```

## Code style
- Python: formatted with `ruff format`, linted with `ruff check`, type hints for public functions.
- CUDA/C++: `clang-format` (config in repo); every kernel must have a correctness test against a PyTorch reference.

## Adding a new early-exit method
1. Implement the shared interface in `methods/base.py`.
2. Classify it as a **pure stopping rule** (replay-compatible) or a **steering** method (online only).
3. Cite the original paper in the module docstring (title, authors, arXiv ID).
4. If reimplementing without official code, state this explicitly and document any deviations.

## Research integrity
- **Never commit fabricated or hand-edited results.** All numbers must be produced by scripts in this repository and be reproducible from logged configs and seeds.
- Every reported experiment must log: model, precision, method, hyperparameters, seed, hardware, driver/CUDA versions.
- Verify every citation against its source (arXiv / ACL Anthology) before adding it to docs or papers.
- Do not copy text, figures, or tables from other papers; paraphrase and cite.

## Reporting issues
Use the issue templates. For reproducibility issues, include your hardware, software versions, and the exact command you ran.

## Code of Conduct
By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
