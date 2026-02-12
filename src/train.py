import argparse
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

import wandb
from src.config.base import infra
from src.config.experiments import CLIPAlignmentConfig, SupervisedExperimentConfig
from src.config.loader import load_config
from src.experiments.clip import run_clip
from src.experiments.supervised import run_supervised


def _download_checkpoint(run_id: str) -> Path:
    api = wandb.Api()
    run_path = f"{infra.wandb_project}/{run_id}"
    if infra.wandb_entity:
        run_path = f"{infra.wandb_entity}/{run_path}"

    run = api.run(run_path)
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        raise RuntimeError(f"No model artifacts found for run {run_id}")

    download_dir = infra.wandb_save_dir / "resume" / run_id
    artifacts[-1].download(root=str(download_dir))

    ckpts = list(download_dir.glob("*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"No .ckpt files in artifact for run {run_id}")

    print(f"Downloaded checkpoint: {ckpts[0]}")
    return ckpts[0]


def main() -> int:
    load_dotenv()
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser(description="Run training experiment")
    parser.add_argument("config", type=Path, help="Path to JSON config file")
    parser.add_argument("--resume", type=str, help="Wandb run ID to resume from")
    parser.add_argument("--seed", type=int, help="Override config seed")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"Config validation failed: {e}", file=sys.stderr)
        return 1

    if args.seed is not None:
        config = config.model_copy(
            update={
                "training": config.training.model_copy(update={"seed": args.seed}),
            },
        )

    print(f"Config valid: {config.experiment_name} ({config.experiment_type})")

    ckpt_path = _download_checkpoint(args.resume) if args.resume else None

    match config:
        case SupervisedExperimentConfig():
            run_supervised(config, ckpt_path=ckpt_path, wandb_run_id=args.resume)
        case CLIPAlignmentConfig():
            run_clip(config, ckpt_path=ckpt_path, wandb_run_id=args.resume)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
