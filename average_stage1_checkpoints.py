"""
Average the top-3 Stage-1 checkpoints selected by development loss.

Selection:
    1. Read Stage-1 development metrics.
    2. Rank checkpoints by development loss.
    3. Select the three checkpoints with the lowest dev loss.
    4. Average their model parameters.
    5. Save the averaged checkpoint as llm_avg3.pt.

The resulting checkpoint is used to initialize Stage 2.
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the Stage-1 Avg3 checkpoint"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing Stage-1 checkpoints",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Stage-1 train_dev_metrics.jsonl",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Avg3 checkpoint",
    )

    return parser.parse_args()


def read_dev_losses(metrics_file):
    """
    Read development loss for each evaluated training step.
    """

    dev_results = []

    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("phase", "").lower() != "dev":
                continue

            step = int(record["step"])

            dev_loss = record.get("dev_loss")

            # Fallback for logs in which loss is stored in metrics.
            if dev_loss is None:
                metrics = record.get("metrics", {})

                for key in [
                    "loss",
                    "llm_loss",
                    "total_loss",
                    "loss_total",
                ]:
                    if key in metrics:
                        dev_loss = metrics[key]
                        break

            if dev_loss is None:
                continue

            dev_results.append(
                {
                    "step": step,
                    "dev_loss": float(dev_loss),
                }
            )

    return dev_results


def select_top3(model_dir, dev_results):
    """
    Select the three available checkpoints with the lowest dev loss.
    """

    candidates = []

    for item in dev_results:
        step = item["step"]

        checkpoint = model_dir / f"llm_step_{step}.pt"

        if checkpoint.is_file():
            candidates.append(
                {
                    "step": step,
                    "dev_loss": item["dev_loss"],
                    "checkpoint": checkpoint,
                }
            )

    candidates.sort(
        key=lambda x: x["dev_loss"]
    )

    if len(candidates) < 3:
        raise RuntimeError(
            f"Need at least 3 valid checkpoints, "
            f"but only found {len(candidates)}."
        )

    return candidates[:3]


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dict: {path}"
        )

    return checkpoint


def average_checkpoints(checkpoint_paths):
    """
    Average floating-point tensors across the selected checkpoints.

    Non-floating tensors are copied from the first checkpoint.
    """

    states = [
        load_checkpoint(path)
        for path in checkpoint_paths
    ]

    reference_keys = set(states[0].keys())

    for i, state in enumerate(states[1:], start=2):
        if set(state.keys()) != reference_keys:
            raise RuntimeError(
                f"Checkpoint {i} has a different structure."
            )

    averaged = {}

    for key in states[0].keys():
        values = [
            state[key]
            for state in states
        ]

        first = values[0]

        if torch.is_tensor(first):
            if first.is_floating_point():
                value = first.clone().float()

                for tensor in values[1:]:
                    value += tensor.float()

                value /= len(values)

                averaged[key] = value.to(
                    dtype=first.dtype
                )

            else:
                averaged[key] = first.clone()

        else:
            # Metadata such as step/epoch is not averaged.
            averaged[key] = first

    # Remove checkpoint-specific training metadata if present.
    averaged.pop("step", None)
    averaged.pop("epoch", None)

    return averaged


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)

    if args.metrics is None:
        metrics_file = (
            model_dir / "train_dev_metrics.jsonl"
        )
    else:
        metrics_file = Path(args.metrics)

    if args.output is None:
        output_file = model_dir / "llm_avg3.pt"
    else:
        output_file = Path(args.output)

    if not metrics_file.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_file}"
        )

    dev_results = read_dev_losses(
        metrics_file
    )

    top3 = select_top3(
        model_dir,
        dev_results,
    )

    print("=" * 70)
    print("Stage-1 checkpoint selection")
    print("=" * 70)

    for rank, item in enumerate(
        top3,
        start=1,
    ):
        print(
            f"Top-{rank}: "
            f"step={item['step']}, "
            f"dev_loss={item['dev_loss']:.6f}, "
            f"checkpoint={item['checkpoint']}"
        )

    checkpoint_paths = [
        item["checkpoint"]
        for item in top3
    ]

    averaged_state = average_checkpoints(
        checkpoint_paths
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        averaged_state,
        output_file,
    )

    manifest = {
        "selection":
            "top3_lowest_dev_loss_available_at_early_stop",

        "num_checkpoints": 3,

        "top3": [
            {
                "rank": rank,
                "step": item["step"],
                "dev_loss": item["dev_loss"],
                "checkpoint":
                    str(item["checkpoint"]),
            }
            for rank, item in enumerate(
                top3,
                start=1,
            )
        ],

        "output":
            str(output_file),
    }

    manifest_path = (
        output_file.parent /
        "avg3_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Avg3 checkpoint created")
    print("=" * 70)
    print(f"Output   : {output_file}")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    main()
