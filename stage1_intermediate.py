"""
Stage 1: Intermediate-language adaptation.

The pretrained multilingual CosyVoice3 LLM is adapted using
linguistically selected intermediate-language data.

Only the language modeling module is optimized. The speech
generation modules remain frozen.

After Stage 1, multiple checkpoints are evaluated on the
development set. The top-3 checkpoints with the lowest
development loss are averaged before Stage 2.
"""

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 intermediate-language adaptation"
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
        help="Training configuration file",
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Pretrained CosyVoice3 LLM checkpoint",
    )

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Intermediate-language training data.list",
    )

    parser.add_argument(
        "--cv-data",
        type=str,
        required=True,
        help="Intermediate-language development data.list",
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
        default=32400,
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=810,
    )

    parser.add_argument(
        "--cv-interval",
        type=int,
        default=810,
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
        "--domain",
        type=str,
        default="INTERMEDIATE",
    )

    parser.add_argument(
        "--run-tag",
        type=str,
        default="stage1_intermediate",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tensorboard_dir is None:
        tensorboard_dir = output_dir / "tensorboard"
    else:
        tensorboard_dir = Path(args.tensorboard_dir)

    tensorboard_dir.mkdir(parents=True, exist_ok=True)

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

        # Stage 1 starts from the pretrained multilingual model.
        "--checkpoint",
        args.pretrained_checkpoint,

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

        "--timeout",
        "60",
    ]

    print("=" * 70)
    print("Stage 1: Intermediate-language adaptation")
    print("=" * 70)
    print(f"Pretrained checkpoint : {args.pretrained_checkpoint}")
    print(f"Training data         : {args.train_data}")
    print(f"Development data      : {args.cv_data}")
    print(f"Maximum steps         : {args.max_train_steps}")
    print(f"Save interval         : {args.save_interval}")
    print(f"CV interval           : {args.cv_interval}")
    print(f"Output directory      : {output_dir}")
    print("=" * 70)

    subprocess.run(command, check=True)

    print()
    print("Stage 1 finished.")
    print(
        "Next: select the three checkpoints with the lowest "
        "development loss and run average_stage1_checkpoints.py."
    )


if __name__ == "__main__":
    main()
