from synthetic_rationale import generate_rationale


def build_feature_block(row, feature_mode):
    lines = []

    if feature_mode in ["all", "ppi_only"] or feature_mode == "no_seq" or feature_mode == "no_emb":
        if feature_mode != "no_ppi":
            lines.append(f"PPI score: {row['ppi_score']:.4f}")

    if feature_mode in ["all", "seq_only"] or feature_mode == "no_ppi" or feature_mode == "no_emb":
        if feature_mode != "no_seq":
            lines.append(
                f"Sequence similarity: {row['seq_sim_simple']:.4f}"
            )

    if feature_mode not in ["ppi_only", "seq_only", "no_emb"]:
        lines.extend([
            f"Embedding similarity (P1-P2): {row['emb_sim_p1_p2']:.4f}",
            f"Embedding similarity (Drug-P1): {row['emb_sim_d_p1']:.4f}",
            f"Embedding similarity (Drug-P2): {row['emb_sim_d_p2']:.4f}",
        ])

    return "\n".join(lines)


def build_messages(
    row,
    prompt_style,
    feature_mode,
    thresholds=None,
    include_label=True,
):
    evidence = build_feature_block(row, feature_mode)

    if prompt_style == "direct":
        instruction = "Predict whether the drug interacts with protein P2."

    elif prompt_style == "cot":
        instruction = (
            "Reason step by step and predict whether the drug interacts with protein P2."
        )

    else:
        instruction = (
            "Analyze the biomedical evidence and predict interaction."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a biomedical expert.",
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nEvidence:\n{evidence}",
        },
    ]

    if include_label:
        label = int(row["label"])

        if prompt_style == "synthetic_rationale":
            rationale = generate_rationale(row, thresholds)

            content = (
                f"Reasoning:\n"
                f"{rationale}\n\n"
                f"Prediction: {label}"
            )

        elif prompt_style == "cot":
            if label == 1:
                reasoning = (
                    "Considering the evidence step by step, interaction appears likely."
                )
            else:
                reasoning = (
                    "Considering the evidence step by step, interaction appears unlikely."
                )

            content = (
                f"Reasoning:\n"
                f"{reasoning}\n\n"
                f"Prediction: {label}"
            )

        else:
            content = f"Prediction: {label}"

        messages.append({
            "role": "assistant",
            "content": content,
        })

    return messages