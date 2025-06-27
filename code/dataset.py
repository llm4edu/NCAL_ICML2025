"""Dataset handling for conversation data with class labels."""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

import torch
from config import DataArguments
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """Custom dataset for conversation data with class labels."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.label_types = data_args.label_types

        # Load dataset
        self.raw_data = self._load_dataset(dataset_path)
        logger.info(f"Loaded {len(self.raw_data)} examples")

        # Log label distribution for debugging
        self._log_label_distribution()

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from JSON file or directory."""
        if os.path.isfile(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif os.path.isdir(dataset_path):
            data = []
            for file in os.listdir(dataset_path):
                if file.endswith(".json"):
                    file_path = os.path.join(dataset_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                data.extend(file_data)
                            else:
                                data.append(file_data)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        else:
            raise ValueError(
                f"Dataset path {dataset_path} is neither a file nor a directory"
            )

        if not data:
            raise ValueError(f"No data found in {dataset_path}")

        return data

    def _log_label_distribution(self):
        """Log the distribution of labels in the dataset."""
        label_counts = defaultdict(int)
        for example in self.raw_data:
            label = example.get("label", "others")
            label_counts[label] += 1

        logger.info("Label distribution:")
        total_examples = len(self.raw_data)
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_examples) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        example = self.raw_data[idx]

        # Extract conversation and label
        conversation = example.get("conversation", [])
        label = example.get("label", "others")

        # Process the conversation
        processed = self._preprocess_conversation(conversation, label)

        return processed

    def _preprocess_conversation(
        self, conversation: List[Dict], label: str
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single conversation."""

        if not conversation:
            input_text = ""
        else:
            # Format conversation as input
            try:
                input_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,  # ✅ Add generation prompt
                )
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}, falling back to simple concatenation"
                )
                # Fallback: simple concatenation of all content
                input_text = ""
                for msg in conversation:
                    content = msg.get("content", "")
                    role = msg.get("role", "")
                    input_text += f"{role}: {content}\n"
                input_text = input_text.strip() + "\nAssistant:"

        # The target output is the label
        target_text = label + self.tokenizer.eos_token

        # Tokenize input and target separately first
        input_tokenized = self.tokenizer(
            input_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        target_tokenized = self.tokenizer(
            target_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )

        input_ids = input_tokenized["input_ids"]
        target_ids = target_tokenized["input_ids"]

        # Combine input and target
        full_input_ids = input_ids + target_ids

        # Create attention mask
        attention_mask = [1] * len(full_input_ids)

        # Create labels: -100 for input part (no loss), real tokens for target part
        labels = [-100] * len(input_ids) + target_ids

        # Truncate if too long
        max_len = self.data_args.cutoff_len
        if len(full_input_ids) > max_len:
            full_input_ids = full_input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            labels = labels[:max_len]

        # Get class index for additional loss calculation
        try:
            class_idx = self.label_types.index(label)
        except ValueError:
            logger.warning(f"Unknown label '{label}', using default class")
            class_idx = len(self.label_types) - 1 if self.label_types else 0

        return {
            "input_ids": torch.tensor(full_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "class_indices": torch.tensor(class_idx, dtype=torch.long),
        }


def create_data_collator(tokenizer: PreTrainedTokenizer):
    """Create data collator for batching."""

    def collate_fn(batch):
        # Extract all fields
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        class_indices_list = [item["class_indices"] for item in batch]

        # Pad sequences to the same length
        max_len = max(len(ids) for ids in input_ids_list)

        # Pad input_ids, attention_mask, and labels
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i in range(len(batch)):
            input_ids = input_ids_list[i]
            attention_mask = attention_mask_list[i]
            labels = labels_list[i]

            pad_len = max_len - len(input_ids)

            # Pad input_ids
            padded_input_ids.append(
                torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (pad_len,), tokenizer.pad_token_id, dtype=torch.long
                        ),
                    ]
                )
            )

            # Pad attention_mask
            padded_attention_mask.append(
                torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            )

            # Pad labels
            padded_labels.append(
                torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "class_indices": torch.stack(class_indices_list),
        }

    return collate_fn
