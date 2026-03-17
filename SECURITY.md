# Security Policy

## Supported Scope
This repository provides research code and reference implementation artifacts for MHADFormer.

## Reporting a Vulnerability
If you discover a security issue (e.g., dependency vulnerability, unsafe serialization behavior, or model-serving exposure risk), please report it privately to:

- francisjesmar.montalbo@g.batstate-u.edu.ph
- francismontalbo@ieee.org

Please include:
- vulnerability description,
- impact assessment,
- reproduction steps,
- and suggested mitigation (if available).

## Security Best Practices for Users
- Install dependencies from trusted indexes and pin package versions.
- Run inference/training in isolated environments (venv/containers).
- Avoid exposing raw model endpoints publicly without authentication and rate limiting.
- Validate all inbound payloads in production inference APIs.
- Log and monitor model service usage for anomaly detection.

## Responsible Use Notice
This codebase is intended for research and engineering experimentation. Any medical or clinical deployment must undergo independent verification, risk assessment, and regulatory/ethics review.
