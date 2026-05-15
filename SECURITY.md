# Security Policy

## Supported versions

| Version | Security fixes |
| --- | --- |
| 0.2.x | Yes |
| 0.1.x | No — please upgrade |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **anteneh@yayasystems.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal Python snippet or test case is ideal)
- Any proof-of-concept code (we won't weaponize it)

You will receive an acknowledgement within **24 hours** and a status update
within **5 business days**.

## Disclosure process

1. We confirm the report and assess severity.
2. We develop and test a fix on a private branch.
3. We release a patched version and push to PyPI.
4. We publish a GitHub Security Advisory crediting the reporter (unless
   they prefer to remain anonymous).

We ask that reporters follow responsible disclosure and wait for the fix to
be released before public disclosure. We target a **14-day** fix window for
critical issues and **30 days** for medium/low.

## Scope

In scope:

- Remote code execution or arbitrary file read/write via crafted SMILES,
  Mol blocks, SDF files, or Parquet files
- Supply chain attacks against the PyPI package or GitHub Actions release pipeline
- Dependency vulnerabilities in `rdkit`, `torch`, `numpy`, `pyarrow`, or
  other direct dependencies that have a practical exploit path through molcore's API

Out of scope:

- Vulnerabilities in optional upstream libraries (`molfeat`, `datamol`) that
  are not reachable through molcore's public API
- Denial-of-service from intentionally malformed but astronomically large
  inputs beyond molcore's documented limits (`_MAX_SMILES_LEN = 10 000`,
  `_MAX_MOLBLOCK_LEN = 1 MB`)
- Issues that require physical access to the machine running molcore

## Automated scanning

Every push and pull request runs:

- **cargo audit** — Rust dependency CVE scan against the RustSec advisory database
- **pip-audit** — Python dependency CVE scan
- **bandit** — Python static security analysis (medium+ severity)
- **gitleaks** — secret / credential leak detection across the full git history

Dependency updates are automated via **Dependabot** (weekly cadence for
GitHub Actions, Cargo, and pip).
