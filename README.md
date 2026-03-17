# MHADFormer 🧠

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.asoc.2026.114624-blue)](https://doi.org/10.1016/j.asoc.2026.114624)
[![Journal](https://img.shields.io/badge/Journal-Applied%20Soft%20Computing-informational)](https://www.sciencedirect.com/science/article/pii/S1568494626000724)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)](https://www.tensorflow.org/)

Official research implementation and reference code for **MHADFormer**, a hybrid deep learning architecture for **medical image classification**.

This repository is designed for:
- reproducible experimentation,
- fast onboarding for researchers and engineers,
- practical deployment handoff,
- and strong discoverability in search engines and LLM-based research tools.

## Table of Contents
- [Graphical Abstract](#graphical-abstract)
- [Paper Links](#paper-links)
- [Keywords (SEO / LLM Discoverability)](#keywords-seo--llm-discoverability)
- [Repository Overview](#repository-overview)
- [Quick Start](#quick-start)
- [Minimal Usage](#minimal-usage)
- [Validation & Reproducibility](#validation--reproducibility)
- [Production Readiness & Security](#production-readiness--security)
- [Citation Guidance](#citation-guidance)
- [Contact](#contact)

## Graphical Abstract

![MHADFormer graphical abstract](mhadformer_2025_graphical_abstract.webp)

## Paper Links

- ScienceDirect article: https://www.sciencedirect.com/science/article/pii/S1568494626000724
- DOI landing page: https://doi.org/10.1016/j.asoc.2026.114624

## Keywords

MHADFormer, medical image classification, Applied Soft Computing, TensorFlow, Keras, hybrid CNN transformer, efficient transformer, lightweight deep learning, healthcare AI, clinical AI, reproducible medical AI, deployable AI.

## Repository Overview

```text
.
├── main.py                         # robust CLI: smoke test + optional SavedModel export
├── mhadformer.py                   # MHADFormer model definition (paper-aligned)
├── requirements.txt                # runtime dependencies
├── scripts/validate_alignment.py   # static paper/repo alignment checks
├── mhadformer_model.ipynb          # notebook playground
└── utils/blocks/
    ├── stemblock.py
    ├── cefe.py
    ├── emvit.py
    └── faces.py
```

## Quick Start

### 1) Setup

```bash
git clone https://github.com/francismontalbo/mhadformer.git
cd mhadformer
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run a smoke test

```bash
python main.py --num-classes 5 --image-size 224 --batch-size 2
```

### 3) Enable deterministic TensorFlow ops (optional)

```bash
python main.py --deterministic-ops --seed 42
```

### 4) Export a SavedModel artifact

```bash
python main.py --save-dir ./artifacts/mhadformer_saved_model/v1
```

## Minimal Usage

```python
import tensorflow as tf
from mhadformer import MHADFormer

model = MHADFormer(num_classes=5, image_size=224)
x = tf.random.normal((2, 224, 224, 3))
y = model(x, training=False)
print(y.shape)  # (2, 5)
```

## Validation & Reproducibility

Run the static alignment validator before publishing results:

```bash
python scripts/validate_alignment.py
```

What this validates:
- Official DOI and ScienceDirect links in `README.md`.
- Baseline MHADFormer architecture declarations in `mhadformer.py` (Stem, EMViT, CeFE, FACeS, classifier head).

Reproducibility best practices:
- Keep seeds fixed (`--seed`) and record them in experiment logs.
- Keep preprocessing, augmentation, and split strategy fixed across benchmarks.
- Version both model artifacts and training code.

## Production Readiness & Security

- Use versioned model exports (e.g., `v1`, `v2`) and immutable artifact storage.
- Pin runtime dependencies and record hardware/software stack (CUDA, cuDNN, TensorFlow).
- Add explicit input validation and class-label mapping in serving wrappers.
- Follow repository security process in [SECURITY.md](SECURITY.md).
- Do not deploy models for clinical decision-making without proper validation, governance, and regulatory review.

## Citation Guidance

For strict bibliographic consistency, export your citation directly from official sources (ScienceDirect or Crossref) for BibTeX / RIS / EndNote.

## Contact

**Francis Jesmar P. Montalbo**  
Batangas State University  
francisjesmar.montalbo@g.batstate-u.edu.ph  
francismontalbo@ieee.org
