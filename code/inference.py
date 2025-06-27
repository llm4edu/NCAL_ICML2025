#!/usr/bin/env python3
"""
Inference script for trained LoRA model.

Usage:
    python inference.py --model_path /path/to/trained/model --prompt "Your prompt here"
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInference:
    """Inference wrapper for trained LoRA model."""

    def __init__(self, model_path: str, base_model_path: Optional[str] = None):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_model()

    def _load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",  # For inference
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if it's a LoRA model
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            # Load LoRA model
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get("base_model_name_or_path")
            if self.base_model_path:
                base_model_name = self.base_model_path

            logger.info(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            logger.info("Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Load full model
            logger.info("Loading full model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()
        logger.info(f"Model loaded on device: {next(self.model.parameters()).device}")

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate response for a given prompt."""

        # Format prompt using chat template
        messages = [{"role": "user", "content": prompt}]

        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, falling back to simple format"
            )
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    def batch_generate(
        self, prompts: List[str], max_length: int = 512, **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, max_length, **kwargs)
            responses.append(response)
        return responses

    def generate_with_conversation_history(
        self,
        conversation: List[dict],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """Generate response with conversation history."""

        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(
                f"Failed to apply chat template: {e}, falling back to simple format"
            )
            # Fallback: simple concatenation
            formatted_prompt = ""
            for msg in conversation:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_prompt += f"{role}: {content}\n"
            formatted_prompt += "assistant:"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Inference with trained LoRA model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--base_model_path", type=str, help="Path to base model (for LoRA models)"
    )
    parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="File containing multiple prompts (one per line)",
    )
    parser.add_argument("--output_file", type=str, help="Output file for responses")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Load model
    inference_model = ModelInference(args.model_path, args.base_model_path)

    if args.interactive:
        # Interactive mode
        logger.info("Starting interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nUser: ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                if not prompt:
                    continue

                response = inference_model.generate_response(
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(f"Assistant: {response}")

            except KeyboardInterrupt:
                break

    elif args.prompt:
        # Single prompt
        response = inference_model.generate_response(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "prompt": args.prompt,
                        "response": response,
                        "parameters": {
                            "max_length": args.max_length,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    elif args.prompts_file:
        # Multiple prompts from file
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        logger.info(f"Processing {len(prompts)} prompts...")

        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            response = inference_model.generate_response(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            result = {
                "prompt": prompt,
                "response": response,
            }
            results.append(result)
            print(f"Prompt {i+1}: {prompt}")
            print(f"Response {i+1}: {response}")
            print("-" * 50)

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "results": results,
                        "parameters": {
                            "max_length": args.max_length,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Results saved to {args.output_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
