"""Custom trainer with ETF loss for neural collapse."""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
from config import ETFLossArguments
from transformers import Trainer

logger = logging.getLogger(__name__)


class ETFLossTrainer(Trainer):
    """Custom trainer with ETF loss for neural collapse."""

    def __init__(
        self,
        etf_args: ETFLossArguments,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.etf_args = etf_args
        self.lambda_lc = etf_args.lambda_lc
        self.e_w = etf_args.e_w
        self.class_diff = etf_args.class_diff
        self.use_etf_loss = etf_args.use_etf_loss

        # Calculate num_classes from dataset if available
        if hasattr(self.train_dataset, "label_types"):
            self.num_classes = len(self.train_dataset.label_types)
        else:
            # Fallback: try to infer from class_indices in dataset
            try:
                max_class = max(
                    item["class_indices"].item() for item in self.train_dataset
                )
                self.num_classes = max_class + 1
            except:
                self.num_classes = 7  # Default fallback
                logger.warning(
                    f"Could not determine number of classes, using default: {self.num_classes}"
                )

        logger.info(f"ETF Trainer initialized with {self.num_classes} classes")
        if self.use_etf_loss:
            logger.info(
                f"ETF loss enabled: λ={self.lambda_lc}, e_w={self.e_w}, class_diff={self.class_diff}"
            )
        else:
            logger.info("ETF loss disabled")

    def etf_loss_all_pairs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute ETF loss for all pairs."""
        mu = -1 / (self.num_classes - 1)
        N, seq_len, hidden_size = hidden_states.size()

        # Select embeddings from the last time step
        embeddings = hidden_states[:, -1, :]

        # Normalize to E_W norm
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings * torch.clamp_max(norms, self.e_w) / (norms + 1e-8)

        # Compute dot products
        dot_products = torch.matmul(embeddings, embeddings.t())

        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(dot_products), diagonal=1).bool()
        upper_triangular = dot_products[mask]

        # Compute loss
        loss = torch.mean((upper_triangular - mu) ** 2)

        return loss

    def etf_loss_diff_classes(
        self, hidden_states: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute ETF loss only for different class pairs."""
        mu = -1 / (self.num_classes - 1)
        N, seq_len, hidden_size = hidden_states.size()

        # Select embeddings from the last time step
        embeddings = hidden_states[:, -1, :]

        # Normalize to E_W norm
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        embeddings = embeddings * torch.clamp_max(norms, self.e_w) / (norms + 1e-8)

        # Compute dot products
        dot_products = torch.matmul(embeddings, embeddings.t())

        # Create mask for different classes only
        class_indices = class_indices.view(-1, 1)
        diff_class_mask = (class_indices != class_indices.t()).float()

        # Get upper triangular mask
        upper_mask = torch.triu(torch.ones_like(dot_products), diagonal=1).float()

        # Combine masks
        combined_mask = diff_class_mask * upper_mask

        # Apply mask
        masked_dot_products = dot_products * combined_mask

        # Compute loss only for valid pairs
        valid_pairs = combined_mask.sum()
        if valid_pairs > 0:
            loss = (
                torch.sum((masked_dot_products - mu * combined_mask) ** 2) / valid_pairs
            )
        else:
            loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        return loss

    def compute_etf_loss(
        self, hidden_states: torch.Tensor, class_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute ETF loss based on configuration."""
        try:
            if not self.class_diff:
                return self.etf_loss_all_pairs(hidden_states)
            else:
                if class_indices is None:
                    logger.warning(
                        "class_indices not provided for class-diff ETF loss, falling back to all pairs"
                    )
                    return self.etf_loss_all_pairs(hidden_states)
                return self.etf_loss_diff_classes(hidden_states, class_indices)
        except Exception as e:
            logger.warning(f"Error computing ETF loss: {e}")
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Compute combined loss: base loss + ETF loss."""

        # Extract class indices if available
        class_indices = inputs.pop("class_indices", None)

        # Get model outputs
        outputs = model(**inputs, output_hidden_states=self.use_etf_loss)

        # Compute base loss
        base_loss = outputs.loss
        total_loss = base_loss

        # Add ETF loss if enabled
        if self.use_etf_loss and outputs.hidden_states is not None:
            try:
                # Get hidden states from the last layer
                hidden_states = outputs.hidden_states[-1]
                etf_loss = self.compute_etf_loss(hidden_states, class_indices)
                total_loss = base_loss + self.lambda_lc * etf_loss

                # Log losses periodically
                if self.state.global_step % self.args.logging_steps == 0:
                    logger.info(
                        f"Step {self.state.global_step}: "
                        f"Base Loss: {base_loss:.4f}, "
                        f"ETF Loss: {etf_loss:.4f}, "
                        f"Total Loss: {total_loss:.4f}"
                    )
            except Exception as e:
                logger.warning(f"Error computing ETF loss: {e}, using base loss only")
                total_loss = base_loss

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    def log_metrics(self, split: str, metrics: Dict[str, float]) -> None:
        """Enhanced logging with additional metrics."""
        super().log_metrics(split, metrics)

        # Log memory usage if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.info(
                f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
            )

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """Enhanced model saving with additional metadata."""
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        # Save training configuration
        import json
        import os

        config_path = os.path.join(output_dir, "training_config.json")
        training_config = {
            "etf_loss_enabled": self.use_etf_loss,
            "lambda_lc": self.lambda_lc,
            "e_w": self.e_w,
            "num_classes": self.num_classes,
            "class_diff": self.class_diff,
        }

        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)

        logger.info(f"Training configuration saved to {config_path}")
