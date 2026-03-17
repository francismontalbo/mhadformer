"""MHADFormer CLI utility.

Provides:
1) deterministic smoke testing,
2) model summary inspection,
3) optional SavedModel export.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger("mhadformer.cli")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got: {value}")
    return parsed


@dataclass(frozen=True)
class RunConfig:
    num_classes: int
    image_size: int
    batch_size: int
    seed: int
    save_dir: str
    deterministic_ops: bool


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="MHADFormer smoke-test and model export utility")
    parser.add_argument("--num-classes", type=positive_int, default=5, help="Number of output classes")
    parser.add_argument("--image-size", type=positive_int, default=224, help="Square input resolution")
    parser.add_argument("--batch-size", type=positive_int, default=2, help="Batch size for smoke test")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--save-dir", type=str, default="", help="Optional directory to save the model")
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help="Enable deterministic TensorFlow ops where supported",
    )
    args = parser.parse_args(argv)
    return RunConfig(
        num_classes=args.num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        save_dir=args.save_dir,
        deterministic_ops=args.deterministic_ops,
    )


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def configure_reproducibility(seed: int, deterministic_ops: bool, np, tf) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
        LOGGER.info("Deterministic TensorFlow ops enabled.")


def normalize_save_dir(save_dir: str) -> Path:
    path = Path(save_dir).expanduser().resolve()
    return path


def run(config: RunConfig) -> None:
    try:
        import numpy as np
        import tensorflow as tf

        from mhadformer import MHADFormer
    except ModuleNotFoundError as exc:
        missing = exc.name or "required dependency"
        raise SystemExit(
            f"Missing dependency: {missing}. Install requirements first with `pip install -r requirements.txt`."
        ) from exc

    configure_reproducibility(config.seed, config.deterministic_ops, np=np, tf=tf)
    LOGGER.info("Building MHADFormer with num_classes=%d image_size=%d", config.num_classes, config.image_size)

    model = MHADFormer(num_classes=config.num_classes, image_size=config.image_size)
    dummy_batch = tf.random.normal((config.batch_size, config.image_size, config.image_size, 3))
    outputs = model(dummy_batch, training=False)

    LOGGER.info("MHADFormer initialized successfully")
    LOGGER.info("Input shape: %s", dummy_batch.shape)
    LOGGER.info("Output shape: %s", outputs.shape)
    model.summary()

    if config.save_dir:
        export_dir = normalize_save_dir(config.save_dir)
        os.makedirs(export_dir, exist_ok=True)
        model.save(export_dir)
        LOGGER.info("Saved model to: %s", export_dir)


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    config = parse_args(argv)
    run(config)


if __name__ == "__main__":
    main()
