import pandas as pd
from datasets import Dataset

from prompts import build_messages


def frame_to_dataset(
    df,
    tokenizer,
    prompt_style,
    feature_mode,
    thresholds=None,
):
    records = []

    for _, row in df.iterrows():
        messages = build_messages(
            row=row,
            prompt_style=prompt_style,
            feature_mode=feature_mode,
            thresholds=thresholds,
            include_label=True,
        )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        records.append({"text": text})

    return Dataset.from_pandas(
        pd.DataFrame(records),
        preserve_index=False,
    )


def build_inference_prompt(
    row,
    tokenizer,
    prompt_style,
    feature_mode,
):
    messages = build_messages(
        row=row,
        prompt_style=prompt_style,
        feature_mode=feature_mode,
        thresholds=None,
        include_label=False,
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )