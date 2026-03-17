"""Static validation checks for MHADFormer paper/repo alignment and hardening.

This script intentionally avoids importing TensorFlow so it can run in constrained
CI or documentation environments.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(pattern: str, text: str, label: str) -> bool:
    ok = re.search(pattern, text, flags=re.MULTILINE) is not None
    print(f"[{'OK' if ok else 'FAIL'}] {label}")
    return ok


def require_file(path: Path, label: str) -> bool:
    ok = path.exists()
    print(f"[{'OK' if ok else 'FAIL'}] {label}")
    return ok


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model = (ROOT / "mhadformer.py").read_text(encoding="utf-8")
    cli = (ROOT / "main.py").read_text(encoding="utf-8")

    checks = []

    # Paper links and discoverability metadata
    checks.append(require(r"10\.1016/j\.asoc\.2026\.114624", readme, "README includes official DOI"))
    checks.append(
        require(
            r"sciencedirect\.com/science/article/pii/S1568494626000724",
            readme,
            "README includes official ScienceDirect URL",
        )
    )

    # Core architecture declarations in repository baseline
    checks.append(require(r"self\.stem\s*=\s*StemBlock", model, "StemBlock is present"))
    checks.append(require(r"self\.emvit1\s*=\s*EMViTBlock\(num_blocks=1, projection_dim=16", model, "EMViT block 1 config"))
    checks.append(require(r"self\.cefe1\s*=\s*CeFEBlock\(filters=32, strides=1", model, "CeFE block 1 config"))
    checks.append(require(r"self\.cefe2\s*=\s*CeFEBlock\(filters=64, strides=2", model, "CeFE block 2 config"))
    checks.append(require(r"self\.emvit2\s*=\s*EMViTBlock\(num_blocks=1, projection_dim=128", model, "EMViT block 2 config"))
    checks.append(require(r"self\.faces\s*=\s*FACeSBlock", model, "FACeS block is present"))
    checks.append(require(r"self\.final_conv\s*=\s*layers\.Conv2D\(filters=256", model, "Final PWConv config"))
    checks.append(require(r"self\.final_dropout\s*=\s*layers\.Dropout\(0\.5", model, "Dropout config"))
    checks.append(require(r"self\.output_layer\s*=\s*layers\.Dense\(num_classes, activation=\"softmax\"", model, "Classifier head config"))

    # CLI hardening and reproducibility checks
    checks.append(require(r"--deterministic-ops", cli, "CLI supports deterministic ops flag"))
    checks.append(require(r"def positive_int", cli, "CLI validates positive integer inputs"))
    checks.append(require(r"Missing dependency:", cli, "CLI provides actionable dependency error"))

    # Repo-level safety docs
    checks.append(require_file(ROOT / "SECURITY.md", "SECURITY.md exists"))

    failed = sum(1 for c in checks if not c)
    if failed:
        print(f"\nValidation failed: {failed} check(s) did not match expected baseline/hardening checks.")
        return 1

    print("\nValidation passed: repository metadata, baseline architecture, and hardening checks are aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
