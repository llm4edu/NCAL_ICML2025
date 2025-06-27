#!/bin/bash

# Model Inference Script
# Usage: bash inference.sh [options]

set -e

# Default values
MODEL_PATH=""
BASE_MODEL_PATH=""
INTERACTIVE=false
MAX_LENGTH=512
TEMPERATURE=0.7
TOP_P=0.9

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --prompts_file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH         Path to trained model directory (required)"
            echo "  --base_model_path PATH    Path to base model for LoRA models"
            echo "  --prompt TEXT             Single prompt for inference"
            echo "  --prompts_file PATH       File with multiple prompts"
            echo "  --output_file PATH        Output file for results"
            echo "  --max_length N            Maximum generation length (default: $MAX_LENGTH)"
            echo "  --temperature FLOAT       Sampling temperature (default: $TEMPERATURE)"
            echo "  --top_p FLOAT             Top-p sampling (default: $TOP_P)"
            echo "  --interactive             Start interactive mode"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Interactive mode"
            echo "  $0 --model_path ./results --interactive"
            echo ""
            echo "  # Single prompt"
            echo "  $0 --model_path ./results --prompt \"Hello, how are you?\""
            echo ""
            echo "  # Batch inference"
            echo "  $0 --model_path ./results --prompts_file prompts.txt --output_file results.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Print configuration
echo "=================================="
echo "Inference Configuration"
echo "=================================="
echo "Model Path: $MODEL_PATH"
if [ -n "$BASE_MODEL_PATH" ]; then
    echo "Base Model Path: $BASE_MODEL_PATH"
fi
echo "Max Length: $MAX_LENGTH"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Interactive Mode: $INTERACTIVE"
echo "=================================="

# Build command
CMD="python inference.py --model_path \"$MODEL_PATH\""

if [ -n "$BASE_MODEL_PATH" ]; then
    CMD="$CMD --base_model_path \"$BASE_MODEL_PATH\""
fi

CMD="$CMD --max_length $MAX_LENGTH"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_p $TOP_P"

if [ "$INTERACTIVE" = true ]; then
    CMD="$CMD --interactive"
elif [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt \"$PROMPT\""
    if [ -n "$OUTPUT_FILE" ]; then
        CMD="$CMD --output_file \"$OUTPUT_FILE\""
    fi
elif [ -n "$PROMPTS_FILE" ]; then
    if [ ! -f "$PROMPTS_FILE" ]; then
        echo "Error: Prompts file does not exist: $PROMPTS_FILE"
        exit 1
    fi
    CMD="$CMD --prompts_file \"$PROMPTS_FILE\""
    if [ -n "$OUTPUT_FILE" ]; then
        CMD="$CMD --output_file \"$OUTPUT_FILE\""
    fi
else
    echo "No input specified. Use --prompt, --prompts_file, or --interactive"
    echo "Use -h or --help for usage information"
    exit 1
fi

# Run inference
echo "Starting inference..."
eval $CMD

echo ""
echo "Inference completed!"