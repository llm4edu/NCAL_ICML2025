"""Model setup and configuration utilities."""

import logging
from typing import Tuple

import torch
from config import LoRAArguments, ModelArguments
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }

    if dtype_str not in dtype_mapping:
        logger.warning(f"Unknown dtype {dtype_str}, using float16")
        return torch.float16

    return dtype_mapping[dtype_str]


def setup_model_and_tokenizer(
    model_args: ModelArguments,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and setup model and tokenizer."""

    logger.info(f"Loading model from {model_args.model_name_or_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added pad token (using eos_token)")

    # Determine torch dtype
    torch_dtype = get_torch_dtype(model_args.torch_dtype)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
        low_cpu_mem_usage=True,
    )

    logger.info(f"Model loaded with dtype: {model.dtype}")
    logger.info(
        f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}"
    )

    return model, tokenizer


def setup_lora(model: PreTrainedModel, lora_args: LoRAArguments) -> PreTrainedModel:
    """Setup LoRA configuration and apply to model."""

    if not lora_args.use_lora:
        logger.info("LoRA is disabled, using full fine-tuning")
        return model

    logger.info("Setting up LoRA configuration")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias=lora_args.lora_bias,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters info
    model.print_trainable_parameters()

    # Log LoRA configuration
    logger.info(f"LoRA config: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}")
    logger.info(f"Target modules: {lora_args.target_modules}")
    logger.info(f"LoRA dropout: {lora_args.lora_dropout}")

    return model


def get_model_memory_footprint(model: PreTrainedModel) -> str:
    """Get model memory footprint information."""
    try:
        if hasattr(model, "get_memory_footprint"):
            memory_mb = model.get_memory_footprint() / 1024 / 1024
            return f"{memory_mb:.2f} MB"
        else:
            # Estimate based on parameters
            param_count = sum(p.numel() for p in model.parameters())
            # Rough estimate: 4 bytes per parameter for float32, 2 for float16
            bytes_per_param = 2 if model.dtype == torch.float16 else 4
            memory_mb = (param_count * bytes_per_param) / 1024 / 1024
            return f"~{memory_mb:.2f} MB (estimated)"
    except Exception as e:
        logger.warning(f"Could not calculate memory footprint: {e}")
        return "Unknown"


def log_model_info(model: PreTrainedModel):
    """Log detailed model information."""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {param_count:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params/param_count*100:.2f}%")
    logger.info(f"Model memory footprint: {get_model_memory_footprint(model)}")
