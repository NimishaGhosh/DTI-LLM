def compute_thresholds(train_df):
    cols = [
        "ppi_score",
        "seq_sim_simple",
        "emb_sim_p1_p2",
        "emb_sim_d_p1",
        "emb_sim_d_p2",
    ]

    thresholds = {}

    for col in cols:
        thresholds[col] = float(train_df[col].median())

    return thresholds


def generate_rationale(row, thresholds):
    reasons = []

    if row["ppi_score"] >= thresholds["ppi_score"]:
        reasons.append(
            "PPI score is high, suggesting stronger protein association."
        )
    else:
        reasons.append(
            "PPI score is low, suggesting weaker protein association."
        )

    if row["seq_sim_simple"] >= thresholds["seq_sim_simple"]:
        reasons.append(
            "Sequence similarity is relatively high."
        )
    else:
        reasons.append(
            "Sequence similarity is relatively low."
        )

    if row["emb_sim_d_p2"] >= thresholds["emb_sim_d_p2"]:
        reasons.append(
            "Drug embedding similarity with protein P2 is high."
        )
    else:
        reasons.append(
            "Drug embedding similarity with protein P2 is low."
        )

    if row["emb_sim_p1_p2"] >= thresholds["emb_sim_p1_p2"]:
        reasons.append(
            "Protein embedding similarity suggests related representations."
        )

    #if int(row["label"]) == 1:
     #   reasons.append("Therefore interaction is likely.")
    #else:
        #reasons.append("Therefore interaction is unlikely.")
    reasons.append(
        "These features collectively inform the interaction prediction."
    )
    return " ".join(reasons)
