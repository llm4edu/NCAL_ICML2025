#!/bin/bash

# ETF Loss LoRA Fine-tuning Script
# Usage: bash train.sh [custom arguments]

set -e  # Exit on any error

# Default values (can be overridden by command line arguments)
DATASET_PATH="../data/train_example.json"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="./results_tmwpl"

# Training parameters
NUM_EPOCHS=5
BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
LEARNING_RATE=2e-4
MAX_LENGTH=2048

# LoRA parameters
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1

# ETF Loss parameters
USE_ETF_LOSS=true
LAMBDA_LC=0.1
E_W=1.0
CLASS_DIFF=true

# Hardware settings
FP16=true
DEVICE_MAP="auto"

# Logging
LOGGING_STEPS=10
SAVE_STRATEGY="epoch"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --model_name_or_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lambda_lc)
            LAMBDA_LC="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --disable_etf_loss)
            USE_ETF_LOSS=false
            shift
            ;;
        --disable_lora)
            USE_LORA=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH              Dataset path (default: $DATASET_PATH)"
            echo "  --model_name_or_path PATH        Model path (default: $MODEL_PATH)"
            echo "  --output_dir PATH                Output directory (default: $OUTPUT_DIR)"
            echo "  --num_train_epochs N             Number of epochs (default: $NUM_EPOCHS)"
            echo "  --per_device_train_batch_size N  Batch size (default: $BATCH_SIZE)"
            echo "  --learning_rate RATE             Learning rate (default: $LEARNING_RATE)"
            echo "  --lambda_lc WEIGHT               ETF loss weight (default: $LAMBDA_LC)"
            echo "  --lora_r RANK                    LoRA rank (default: $LORA_R)"
            echo "  --disable_etf_loss               Disable ETF loss"
            echo "  --disable_lora                   Disable LoRA (full fine-tuning)"
            echo "  -h, --help                       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set USE_LORA default if not set by --disable_lora
USE_LORA=${USE_LORA:-true}

# Print configuration
echo "=================================="
echo "Training Configuration"
echo "=================================="
echo "Dataset Path: $DATASET_PATH"
echo "Model Path: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "LoRA Enabled: $USE_LORA"
echo "ETF Loss Enabled: $USE_ETF_LOSS"
echo "ETF Loss Weight: $LAMBDA_LC"
echo "=================================="

# Check if required files exist
if [ ! -f "$DATASET_PATH" ] && [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Export CUDA settings for better memory management
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
echo "Starting training..."
python train.py \
    --dataset_path "$DATASET_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --cutoff_len "$MAX_LENGTH" \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --use_etf_loss "$USE_ETF_LOSS" \
    --lambda_lc "$LAMBDA_LC" \
    --e_w "$E_W" \
    --class_diff "$CLASS_DIFF" \
    --fp16 "$FP16" \
    --device_map "$DEVICE_MAP" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy "$SAVE_STRATEGY" \
    --remove_unused_columns false \
    --dataloader_drop_last true \
    --warmup_ratio 0.05 \
    --report_to none

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"