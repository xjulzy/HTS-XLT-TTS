"""
Stage 2: Extremely low-resource target-language adaptation.

Stage 2 is initialized from the averaged Stage-1 checkpoint
(llm_avg3.pt) and further adapted using extremely low-resource
target-language data.

Only the language modeling module is optimized.
"""

import argparse
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 target-language adaptation"
    )

    parser.add_argument(
        "--trainer",
        type=str,
        required=True,
        help="Path to the CosyVoice3 LLM training script",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        required=True,
        help="Stage-1 Avg3 checkpoint (llm_avg3.pt)",
    )

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cv-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train-manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dev-manifest",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--qwen-pretrain-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--cv-interval",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--prefetch",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="TARGET",
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="stage2_target",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    set_seed(args.seed)

    stage1_checkpoint = Path(
        args.stage1_checkpoint
    )

    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage-1 Avg3 checkpoint not found: "
            f"{stage1_checkpoint}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.tensorboard_dir is None:
        tensorboard_dir = (
            output_dir / "tensorboard"
        )
    else:
        tensorboard_dir = Path(
            args.tensorboard_dir
        )

    tensorboard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "python",
        args.trainer,

        "--domain",
        args.domain,

        "--run_tag",
        args.run_tag,

        "--train_engine",
        "torch_ddp",

        # Only adapt the language modeling module.
        "--model",
        "llm",

        "--config",
        args.config,

        "--train_data",
        args.train_data,

        "--cv_data",
        args.cv_data,

        "--train_manifest",
        args.train_manifest,

        "--dev_manifest",
        args.dev_manifest,

        "--qwen_pretrain_path",
        args.qwen_pretrain_path,

        "--onnx_path",
        args.onnx_path,

        # Hierarchical transfer:
        # initialize Stage 2 from the Stage-1 Avg3 model.
        "--checkpoint",
        str(stage1_checkpoint),

        "--model_dir",
        str(output_dir),

        "--tensorboard_dir",
        str(tensorboard_dir),

        "--num_workers",
        str(args.num_workers),

        "--prefetch",
        str(args.prefetch),

        "--pin_memory",
        "--use_amp",

        "--dist_backend",
        "nccl",

        "--max_train_steps",
        str(args.max_train_steps),

        "--save_interval",
        str(args.save_interval),

        "--cv_interval",
        str(args.cv_interval),

        "--seed",
        str(args.seed),

        "--timeout",
        "60",
    ]

    print("=" * 70)
    print(
        "Stage 2: Extremely low-resource "
        "target-language adaptation"
    )
    print("=" * 70)

    print(
        f"Stage-1 Avg3 checkpoint : "
        f"{stage1_checkpoint}"
    )

    print(
        f"Target training data    : "
        f"{args.train_data}"
    )

    print(
        f"Target development data : "
        f"{args.cv_data}"
    )

    print(
        f"Maximum steps           : "
        f"{args.max_train_steps}"
    )

    print(
        f"Save interval           : "
        f"{args.save_interval}"
    )

    print(
        f"CV interval             : "
        f"{args.cv_interval}"
    )

    print(
        f"Random seed             : "
        f"{args.seed}"
    )

    print("=" * 70)

    subprocess.run(
        command,
        check=True,
    )

    print()
    print("Stage 2 finished.")


if __name__ == "__main__":
    main()
