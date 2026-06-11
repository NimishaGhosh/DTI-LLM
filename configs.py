MODEL_CONFIGS = {
    "llama": {
        "padding_side": "right",
        "use_fast": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    },
    "mistral": {
        "padding_side": "right",
        "use_fast": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "qwen": {
        "padding_side": "left",
        "use_fast": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
    },
}