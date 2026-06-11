import argparse
import pandas as pd

from models import (
    load_model_and_tokenizer,
    apply_lora,
)
from trainer_SR import train_one_seed
from baselines import run_baselines
from utils import (
    set_seed,
    ensure_dir,
    save_json,
    aggregate_metrics,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_emb", required=True)
    parser.add_argument("--test_emb", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_root", required=True)

    parser.add_argument(
        "--prompt_style",
        choices=[
            "direct",
            "cot",
            "synthetic_rationale",
        ],
        default="direct",
    )

    parser.add_argument(
        "--feature_mode",
        choices=[
            "all",
            "ppi_only",
            "seq_only",
            "no_ppi",
            "no_seq",
            "no_emb",
        ],
        default="all",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument(
        "--run_baselines",
        action="store_true",
    )

    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_emb)
    test_df = pd.read_parquet(args.test_emb)

    if args.run_baselines:
        baseline_results = run_baselines(
            train_df,
            test_df,
        )
        print(baseline_results)


        save_json(
            baseline_results,
            f"{args.output_root}/baseline_results.json",
        )

        return
    all_metrics = []

    for seed in args.seeds:
        print(f"Running seed {seed}")

        set_seed(seed)

        output_dir = ensure_dir(
            f"{args.output_root}/seed_{seed}"
        )

        model, tokenizer, family = load_model_and_tokenizer(
            args.base_model
        )

        model = apply_lora(model, family)

        metrics = train_one_seed(
            train_df=train_df,
            test_df=test_df,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            prompt_style=args.prompt_style,
            feature_mode=args.feature_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_len=args.max_len,
        )

        all_metrics.append(metrics)

        del model
        torch.cuda.empty_cache()

    agg = aggregate_metrics(all_metrics)

    save_json(
        agg,
        f"{args.output_root}/aggregate_metrics.json",
    )


if __name__ == "__main__":
    import torch
    main()
