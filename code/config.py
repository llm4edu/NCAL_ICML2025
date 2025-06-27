"""Configuration classes for training parameters."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataArguments:
    """Arguments for data preprocessing."""

    dataset_path: str = field(
        metadata={"help": "Path to the dataset directory or JSON file"}
    )
    cutoff_len: int = field(default=2048, metadata={"help": "Maximum sequence length"})
    train_on_prompt: bool = field(
        default=False, metadata={"help": "Whether to train on prompts"}
    )
    mask_history: bool = field(
        default=True, metadata={"help": "Whether to mask conversation history"}
    )
    label_types: List[str] = field(
        default_factory=lambda: [
            "回忆",
            "识别",
            "选取策略",
            "表征建模",
            "分析",
            "推断",
            "过程实施",
        ],
        metadata={"help": "List of label types for classification"},
    )


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "PyTorch dtype for model weights (float16, float32, bfloat16)"
        },
    )
    device_map: str = field(
        default="auto", metadata={"help": "Device mapping strategy for model loading"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""

    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LoRA fine-tuning"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        metadata={"help": "Target modules for LoRA"},
    )
    lora_bias: str = field(
        default="none", metadata={"help": "LoRA bias setting (none, all, lora_only)"}
    )


@dataclass
class ETFLossArguments:
    """Arguments for ETF loss configuration."""

    use_etf_loss: bool = field(
        default=True, metadata={"help": "Whether to use ETF loss for neural collapse"}
    )
    lambda_lc: float = field(default=0.1, metadata={"help": "Weight for ETF loss"})
    e_w: float = field(
        default=1.0, metadata={"help": "Norm constraint for feature vectors"}
    )
    class_diff: bool = field(
        default=True,
        metadata={"help": "Whether to compute dot product only for different classes"},
    )


@dataclass
class TrainingArguments:
    """Extended training arguments."""

    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written"
        }
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform"}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass"
        },
    )
    warmup_ratio: float = field(default=0.05, metadata={"help": "Linear warmup ratio"})
    learning_rate: float = field(
        default=2e-4, metadata={"help": "The initial learning rate for AdamW"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit"},
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps"}
    )
    save_strategy: str = field(
        default="epoch", metadata={"help": "The checkpoint save strategy to use"}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X updates steps"}
    )
    eval_strategy: str = field(
        default="no", metadata={"help": "The evaluation strategy to use"}
    )
    dataloader_drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch"}
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={"help": "The list of integrations to report results and logs to"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
