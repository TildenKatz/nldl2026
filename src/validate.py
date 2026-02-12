import argparse
import sys
from pathlib import Path

from src.config.loader import load_config


def validate(f: Path) -> bool:
    try:
        config = load_config(f)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {f} -> {e}")
        return False

    print(f"OK: {f} -> {config.experiment_name}")
    return True


def validate_settings() -> bool:
    try:
        from src.config.base import infra  # noqa: PLC0415

    except Exception as e:  # noqa: BLE001
        print(f"Infrastructure settings invalid: {e}")
        return False

    print(f"Infrastructure settings valid: {infra}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment configs")
    parser.add_argument(
        "config_dir",
        type=Path,
        nargs="?",
        default=Path("configs"),
        help="Config directory (default: configs/)",
    )
    args = parser.parse_args()
    config_dir: Path = args.config_dir

    config_files = sorted(config_dir.rglob("*.json"))

    if not config_files:
        print(f"No JSON configs found in {config_dir}", file=sys.stderr)
        return 1

    failed = [config_path for config_path in config_files if not validate(config_path)]

    if failed:
        print(f"\n{len(failed)}/{len(config_files)} configs failed validation")
        return 1

    print(f"\nAll {len(config_files)} configs valid")

    if not validate_settings():
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
