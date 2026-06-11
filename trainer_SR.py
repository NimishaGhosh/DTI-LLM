from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from dataset import frame_to_dataset
from evaluate_SR import evaluate_generation
from synthetic_rationale import compute_thresholds
from utils import ensure_dir
import torch

def train_one_seed(
    train_df,
    test_df,
    model,
    tokenizer,
    output_dir,
    prompt_style,
    feature_mode,
    epochs,
    batch_size,
    grad_accum,
    max_len,
):
    thresholds = None

    if prompt_style == "synthetic_rationale":
        thresholds = compute_thresholds(train_df)

    eval_df = test_df.sample(
        min(1000, len(test_df)),
        random_state=42,
    )

    train_dataset = frame_to_dataset(
        df=train_df,
        tokenizer=tokenizer,
        prompt_style=prompt_style,
        feature_mode=feature_mode,
        thresholds=thresholds,
    )

    eval_dataset = frame_to_dataset(
        df=eval_df,
        tokenizer=tokenizer,
        prompt_style=prompt_style,
        feature_mode=feature_mode,
        thresholds=thresholds,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        bf16=bf16,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=max_len,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            )
        ],
    )

    trainer.train()

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    model.config.use_cache = True

    metrics = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        output_dir=output_dir,
        prompt_style=prompt_style,
        feature_mode=feature_mode,
        max_len=max_len,
    )

    return metrics
