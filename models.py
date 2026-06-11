import torch

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from configs import MODEL_CONFIGS


def detect_model_family(model_name):
    name = model_name.lower()

    if "mistral" in name:
        return "mistral"

    if "qwen" in name:
        return "qwen"

    if "llama" in name:
        return "llama"

    raise ValueError(f"Unsupported model: {model_name}")


def load_model_and_tokenizer(model_name):
    family = detect_model_family(model_name)
    cfg = MODEL_CONFIGS[family]

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=cfg["use_fast"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = cfg["padding_side"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        #attn_implementation="flash_attention_2",
    )

    model.config.use_cache = False

    return model, tokenizer, family


def apply_lora(model, family):
    cfg = MODEL_CONFIGS[family]

    config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)

    return model
