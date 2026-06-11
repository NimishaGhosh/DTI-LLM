import json
import re
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)

from dataset import build_inference_prompt


def parse_prediction(text):
    match = re.search(r"Prediction:\s*([01])", text)

    if match:
        return int(match.group(1))

    return 0


def evaluate_generation(
    model,
    tokenizer,
    test_df,
    output_dir,
    prompt_style,
    feature_mode,
    max_len=512,
):
    model.eval()

    y_true = []
    y_pred = []
    rows = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        prompt = build_inference_prompt(
            row=row,
            tokenizer=tokenizer,
            prompt_style=prompt_style,
            feature_mode=feature_mode,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )

        pred = parse_prediction(decoded)
        label = int(row["label"])

        y_true.append(label)
        y_pred.append(pred)

        out_row = row.to_dict()
        out_row.update({
            "true_label": label,
            "pred_label": pred,
            "generated_text": decoded,
        })

        rows.append(out_row)

        if idx % 100 == 0:
            print(f"Evaluated {idx}/{len(test_df)}")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    pd.DataFrame(rows).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    np.save(
        output_dir / "confusion_matrix.npy",
        confusion_matrix(y_true, y_pred),
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred))

    return metrics