#!/usr/bin/env python3
"""
Main training script for LoRA fine-tuning with ETF loss.

Usage:
    python train.py --dataset_path /path/to/data --output_dir /path/to/output [other options]
"""

import logging
import os
import sys
from typing import Optional

import torch
from config import DataArguments, ETFLossArguments, LoRAArguments, ModelArguments
from dataset import ConversationDataset, create_data_collator
from model import log_model_info, setup_lora, setup_model_and_tokenizer
from trainer import ETFLossTrainer
from transformers import HfArgumentParser, TrainingArguments, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def check_arguments(args_dict: dict) -> None:
    """Validate and check arguments."""
    data_args, model_args, lora_args, etf_args, training_args = args_dict.values()

    # Check required paths
    if not os.path.exists(data_args.dataset_path):
        raise ValueError(f"Dataset path does not exist: {data_args.dataset_path}")

    # Create output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Validate ETF loss configuration
    if etf_args.use_etf_loss:
        if etf_args.lambda_lc <= 0:
            logger.warning(f"ETF loss weight is non-positive: {etf_args.lambda_lc}")
        if len(data_args.label_types) < 2:
            logger.warning(
                f"ETF loss with only {len(data_args.label_types)} classes may not be effective"
            )

    # Log key configurations
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {data_args.dataset_path}")
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"LoRA enabled: {lora_args.use_lora}")
    logger.info(f"ETF loss enabled: {etf_args.use_etf_loss}")
    logger.info(f"Label types: {data_args.label_types}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info("=" * 60)


def setup_logging_file(output_dir: str) -> None:
    """Setup file logging in addition to console logging."""
    log_file = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add file handler to root logger
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")


def main():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser(
        (
            DataArguments,
            ModelArguments,
            LoRAArguments,
            ETFLossArguments,
            TrainingArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON config file
        data_args, model_args, lora_args, etf_args, training_args = (
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        )
    else:
        # Parse from command line
        data_args, model_args, lora_args, etf_args, training_args = (
            parser.parse_args_into_dataclasses()
        )

    # Check and validate arguments
    args_dict = {
        "data": data_args,
        "model": model_args,
        "lora": lora_args,
        "etf": etf_args,
        "training": training_args,
    }
    check_arguments(args_dict)

    # Setup file logging
    setup_logging_file(training_args.output_dir)

    # Set random seed for reproducibility
    set_seed(training_args.seed)
    logger.info(f"Random seed set to: {training_args.seed}")

    try:
        # Setup model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = setup_model_and_tokenizer(model_args)
        log_model_info(model)

        # Setup LoRA if enabled
        if lora_args.use_lora:
            logger.info("Setting up LoRA...")
            model = setup_lora(model, lora_args)
            log_model_info(model)  # Log info after LoRA setup

        # Create dataset
        logger.info("Creating dataset...")
        train_dataset = ConversationDataset(
            dataset_path=data_args.dataset_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )

        # Create data collator
        data_collator = create_data_collator(tokenizer)

        # Update ETF loss arguments with number of classes from dataset
        etf_args.num_classes = len(data_args.label_types)

        # Create trainer
        logger.info("Creating trainer...")
        trainer = ETFLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            etf_args=etf_args,
        )

        # Log training info
        logger.info(f"Starting training with {len(train_dataset)} examples...")
        logger.info(f"Total training steps: {trainer.state.max_steps}")

        # Start training
        train_result = trainer.train()

        # Log training results
        logger.info("Training completed!")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")

        # Save the final model
        logger.info("Saving model...")
        trainer.save_model()

        # Save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)

        # Save training state
        trainer.save_state()

        logger.info(f"Model and tokenizer saved to: {training_args.output_dir}")

        # Log final memory usage
        if torch.cuda.is_available():
            logger.info(
                f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
            )

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
