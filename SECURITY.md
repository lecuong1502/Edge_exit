# Security Policy

## Supported versions
EdgeExit is research software. Only the latest version on the `main` branch receives fixes.

## Reporting a vulnerability
Please **do not** open a public issue for security problems.

Report vulnerabilities privately via GitHub's **"Report a vulnerability"** button (Security tab → Advisories), or by email to **<your-email>**.

Please include:
- A description of the issue and its potential impact
- Steps to reproduce
- Affected files or components

You can expect an initial response within 7 days.

## Scope
Of particular concern:
- Unsafe loading of untrusted model checkpoints or trace files (e.g., pickle deserialization)
- Custom CUDA kernels that could cause out-of-bounds memory access
- Leaked credentials or tokens in the repository
