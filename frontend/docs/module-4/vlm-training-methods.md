---
sidebar_position: 9
title: "VLM Training Methods"
---

# Vision-Language-Model Training Methods

## Introduction to VLM Training

Vision-Language Models (VLMs) require specialized training methodologies that can effectively learn correspondences between visual and linguistic modalities. Unlike traditional computer vision or NLP models that operate on single modalities, VLMs must learn to understand the complex relationships between images and text. This module explores the key training approaches, techniques, and methodologies used to train effective vision-language models for humanoid robotics applications.

## Pre-training Paradigms

### 1. Contrastive Learning

Contrastive learning has emerged as the dominant paradigm for pre-training vision-language models. The core idea is to learn representations that bring matching image-text pairs closer together while pushing non-matching pairs apart.

```python
# Contrastive learning implementation for VLM pre-training
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import numpy as np

class ContrastiveVLM(nn.Module):
    def __init__(self, vision_encoder, text_encoder, projection_dim=512):
        super().__init__()

        # Vision encoder (e.g., ViT)
        self.vision_encoder = vision_encoder
        self.vision_proj = nn.Linear(vision_encoder.config.hidden_size, projection_dim)

        # Text encoder (e.g., BERT/RoBERTa)
        self.text_encoder = text_encoder
        self.text_proj = nn.Linear(text_encoder.config.hidden_size, projection_dim)

        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Normalization layers
        self.vision_norm = nn.LayerNorm(projection_dim)
        self.text_norm = nn.LayerNorm(projection_dim)

    def encode_image(self, images):
        """Encode images to visual representations"""
        vision_outputs = self.vision_encoder(pixel_values=images)
        # Use CLS token representation
        vision_features = vision_outputs.last_hidden_state[:, 0, :]  # [B, vision_hidden_size]
        vision_proj = self.vision_proj(vision_features)  # [B, projection_dim]
        vision_proj = self.vision_norm(vision_proj)  # Normalize for contrastive learning
        return vision_proj

    def encode_text(self, input_ids, attention_mask):
        """Encode text to textual representations"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use CLS token representation for BERT-like models
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [B, text_hidden_size]
        text_proj = self.text_projection(text_features)  # [B, projection_dim]
        text_proj = self.text_norm(text_proj)  # Normalize for contrastive learning
        return text_proj

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass for contrastive learning
        Args:
            images: [B, C, H, W] images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask
        Returns:
            Contrastive loss and similarity matrix
        """
        # Encode vision and text
        vision_features = self.encode_image(images)  # [B, proj_dim]
        text_features = self.encode_text(input_ids, attention_mask)  # [B, proj_dim]

        # Compute similarity matrix
        # Similarity between all image-text pairs
        logits_per_image = torch.matmul(vision_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()  # Transpose for symmetry

        # Create ground truth: diagonal elements are positive pairs
        ground_truth = torch.arange(len(vision_features), device=images.device)

        # Compute contrastive loss
        loss_v2t = F.cross_entropy(logits_per_image, ground_truth)
        loss_t2v = F.cross_entropy(logits_per_text, ground_truth)

        contrastive_loss = (loss_v2t + loss_t2v) / 2.0

        return {
            'loss': contrastive_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'vision_features': vision_features,
            'text_features': text_features
        }

    def compute_alignment_metrics(self, logits_per_image):
        """Compute metrics for vision-language alignment"""
        B = logits_per_image.shape[0]

        # Compute ranking metrics
        ranks = []
        for i in range(B):
            # Sort similarities in descending order
            _, indices = torch.sort(logits_per_image[i, :], descending=True)
            # Find rank of correct text (diagonal element)
            rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

        # Convert to numpy for metric computation
        ranks = torch.tensor(ranks)
        r1 = 100.0 * len(torch.where(ranks <= 1)[0]) / len(ranks)
        r5 = 100.0 * len(torch.where(ranks <= 5)[0]) / len(ranks)
        r10 = 100.0 * len(torch.where(ranks <= 10)[0]) / len(ranks)
        medr = torch.median(ranks).item()
        meanr = torch.mean(ranks.float()).item()

        return {
            'R@1': r1.item(),
            'R@5': r5.item(),
            'R@10': r10.item(),
            'MedR': medr,
            'MeanR': meanr
        }

# Training loop for contrastive learning
class ContrastiveTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2
        )
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    def train_step(self, batch):
        """Single training step with mixed precision"""
        images = batch['images'].cuda()
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision forward pass
            outputs = self.model(images, input_ids, attention_mask)
            loss = outputs['loss']

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Gradient clipping and optimization
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_logits = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].cuda()
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images, input_ids, attention_mask)
                    loss = outputs['loss']

                total_loss += loss.item()
                all_logits.append(outputs['logits_per_image'])

                num_batches += 1

        # Compute alignment metrics
        if all_logits:
            combined_logits = torch.cat(all_logits, dim=0)
            alignment_metrics = self.model.compute_alignment_metrics(combined_logits)
        else:
            alignment_metrics = {}

        avg_loss = total_loss / num_batches
        return avg_loss, alignment_metrics
```

### 2. Masked Language Modeling with Vision

Building on the success of masked language modeling, some approaches extend this concept to vision-language models:

```python
# Masked vision-language modeling
class MaskedVLModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, hidden_dim=768):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.hidden_dim = hidden_dim

        # Cross-modal transformer for fusion
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

        # Vision and text masking heads
        self.vision_mask_head = nn.Linear(hidden_dim, vision_encoder.config.num_channels)
        self.text_mask_head = nn.Linear(hidden_dim, text_encoder.config.vocab_size)

        # Cross-attention for vision-text interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=12,
            batch_first=True
        )

    def forward(self, images, input_ids, attention_mask,
                vision_mask_ratio=0.15, text_mask_ratio=0.15):
        """
        Forward pass with masked reconstruction
        Args:
            images: [B, C, H, W] input images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask
            vision_mask_ratio: ratio of image patches to mask
            text_mask_ratio: ratio of text tokens to mask
        Returns:
            Reconstruction losses and representations
        """
        B, C, H, W = images.shape

        # Encode vision with masking
        vision_features = self.encode_vision_with_masking(
            images, vision_mask_ratio
        )

        # Encode text with masking
        text_features = self.encode_text_with_masking(
            input_ids, attention_mask, text_mask_ratio
        )

        # Cross-modal fusion
        fused_features = self.fuse_modalities(
            vision_features, text_features, attention_mask
        )

        # Reconstruct masked elements
        vision_reconstruction = self.reconstruct_vision(fused_features)
        text_reconstruction = self.reconstruct_text(fused_features, attention_mask)

        # Compute reconstruction losses
        vision_loss = self.compute_vision_reconstruction_loss(
            vision_reconstruction, images, vision_mask_ratio
        )
        text_loss = self.compute_text_reconstruction_loss(
            text_reconstruction, input_ids, attention_mask, text_mask_ratio
        )

        total_loss = vision_loss + text_loss

        return {
            'total_loss': total_loss,
            'vision_loss': vision_loss,
            'text_loss': text_loss,
            'fused_features': fused_features,
            'vision_reconstruction': vision_reconstruction,
            'text_reconstruction': text_reconstruction
        }

    def encode_vision_with_masking(self, images, mask_ratio):
        """Encode vision with random masking"""
        # Get vision features
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # [B, num_patches, hidden_dim]

        # Apply random masking
        B, num_patches, hidden_dim = vision_features.shape
        num_masked = int(num_patches * mask_ratio)

        # Randomly select patches to mask
        mask_indices = torch.randperm(num_patches)[:num_masked].to(vision_features.device)
        batch_indices = torch.arange(B, device=vision_features.device).unsqueeze(1).expand(-1, num_masked)

        # Create mask
        mask = torch.zeros(B, num_patches, device=vision_features.device, dtype=torch.bool)
        mask[batch_indices, mask_indices] = True

        # Store mask for reconstruction
        self.vision_mask = mask

        # Apply mask (set to zero)
        masked_features = vision_features * (~mask.unsqueeze(-1).expand(-1, -1, hidden_dim))

        return masked_features

    def encode_text_with_masking(self, input_ids, attention_mask, mask_ratio):
        """Encode text with random masking"""
        # Create mask for text tokens
        B, seq_len = input_ids.shape

        # Only mask non-padding tokens
        valid_tokens = attention_mask.bool() & (input_ids != self.text_encoder.config.pad_token_id)

        # Calculate number of tokens to mask per sequence
        num_valid_tokens = valid_tokens.sum(dim=1)  # [B]
        num_masked_per_seq = (num_valid_tokens * mask_ratio).long().clamp(min=1)

        # Create mask for each sequence
        text_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(B):
            valid_indices = torch.where(valid_tokens[b])[0]
            if len(valid_indices) > 0:
                num_to_mask = min(num_masked_per_seq[b].item(), len(valid_indices))
                mask_indices = valid_indices[
                    torch.randperm(len(valid_indices))[:num_to_mask]
                ]
                text_mask[b, mask_indices] = True

        # Store mask for reconstruction
        self.text_mask = text_mask

        # Apply masking by replacing with [MASK] token
        masked_input_ids = input_ids.clone()
        masked_input_ids[text_mask] = self.text_encoder.config.mask_token_id

        # Encode masked text
        text_outputs = self.text_encoder(
            input_ids=masked_input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        return text_features

    def fuse_modalities(self, vision_features, text_features, attention_mask):
        """Fuse vision and text features using cross-modal attention"""
        # Concatenate features
        all_features = torch.cat([vision_features, text_features], dim=1)  # [B, num_patches+seq_len, hidden_dim]

        # Create attention mask for concatenated sequence
        vision_attn_mask = torch.ones(B, vision_features.size(1), device=attention_mask.device)
        combined_attn_mask = torch.cat([vision_attn_mask, attention_mask], dim=1)  # [B, num_patches+seq_len]

        # Apply fusion transformer
        fused_features = self.fusion_transformer(
            src=all_features,
            mask=~combined_attn_mask.bool()  # Convert to attention mask format
        )

        return fused_features

    def reconstruct_vision(self, fused_features):
        """Reconstruct vision features"""
        # Get vision portion of fused features
        vision_portion = fused_features[:, :self.num_vision_patches, :]  # [B, num_patches, hidden_dim]

        # Apply reconstruction head
        reconstructed_vision = self.vision_reconstruction_head(vision_portion)  # [B, num_patches, channels]

        return reconstructed_vision

    def reconstruct_text(self, fused_features, attention_mask):
        """Reconstruct text features"""
        # Get text portion of fused features
        text_start_idx = self.num_vision_patches
        text_portion = fused_features[:, text_start_idx:, :]  # [B, seq_len, hidden_dim]

        # Apply reconstruction head
        reconstructed_text = self.text_reconstruction_head(text_portion)  # [B, seq_len, vocab_size]

        return reconstructed_text

    def compute_vision_reconstruction_loss(self, reconstruction, targets, mask_ratio):
        """Compute vision reconstruction loss"""
        # Only compute loss on masked patches
        masked_patches = self.vision_mask  # [B, num_patches]

        if masked_patches.sum() > 0:
            # Get original vision features (for comparison)
            with torch.no_grad():
                original_vision = self.vision_encoder(pixel_values=targets).last_hidden_state

            # Compute MSE loss on masked patches only
            mask_expanded = masked_patches.unsqueeze(-1).expand(-1, -1, reconstruction.size(-1))
            masked_reconstruction = reconstruction[mask_expanded].view(-1, reconstruction.size(-1))
            masked_original = original_vision[mask_expanded].view(-1, original_vision.size(-1))

            loss = F.mse_loss(masked_reconstruction, masked_original)
        else:
            loss = torch.tensor(0.0, device=reconstruction.device)

        return loss

    def compute_text_reconstruction_loss(self, reconstruction, targets, attention_mask, mask_ratio):
        """Compute text reconstruction loss"""
        # Only compute loss on masked tokens
        masked_tokens = self.text_mask  # [B, seq_len]

        if masked_tokens.sum() > 0:
            # Compute cross-entropy loss on masked tokens only
            masked_reconstruction = reconstruction[masked_tokens]  # [num_masked, vocab_size]
            masked_targets = targets[masked_tokens]  # [num_masked]

            loss = F.cross_entropy(masked_reconstruction, masked_targets)
        else:
            loss = torch.tensor(0.0, device=reconstruction.device)

        return loss
```

## Fine-tuning Strategies

### 1. Task-Specific Fine-tuning

Fine-tuning pre-trained VLMs for specific robotics tasks:

```python
# Task-specific fine-tuning for humanoid robotics
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset

class TaskSpecificFineTuner:
    def __init__(self, base_model, task_type, dataset):
        self.base_model = base_model
        self.task_type = task_type
        self.dataset = dataset

        # Task-specific heads
        self.task_heads = self.create_task_heads()

        # Optimizer setup
        self.setup_optimization_strategy()

    def create_task_heads(self):
        """Create task-specific heads for different robotics tasks"""
        task_heads = {}

        if self.task_type == 'object_detection':
            task_heads['detection'] = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 4 + 80)  # 4 bbox coords + 80 class scores
            )

        elif self.task_type == 'grasping':
            task_heads['grasping'] = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 6)  # 6-DoF grasp pose (3 pos + 3 rot)
            )

        elif self.task_type == 'navigation':
            task_heads['navigation'] = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3)  # 3D target position
            )

        elif self.task_type == 'manipulation':
            task_heads['manipulation'] = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 7)  # 7-DoF joint angles for arm
            )

        return task_heads

    def setup_optimization_strategy(self):
        """Set up optimization strategy based on task requirements"""
        # Different learning rates for different components
        param_groups = [
            {
                'params': self.base_model.parameters(),
                'lr': 1e-6,  # Lower LR for pre-trained features
                'weight_decay': 0.01
            },
            {
                'params': self.task_heads.parameters(),
                'lr': 1e-4,  # Higher LR for task-specific heads
                'weight_decay': 0.01
            }
        ]

        self.optimizer = torch.optim.AdamW(param_groups)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2
        )

    def forward(self, images, commands, labels=None):
        """Forward pass with task-specific processing"""
        # Get base vision-language features
        outputs = self.base_model(
            pixel_values=images,
            input_ids=commands['input_ids'],
            attention_mask=commands['attention_mask']
        )

        # Get fused representations
        fused_features = outputs.last_hidden_state[:, 0, :]  # CLS token representation

        # Apply task-specific head
        task_output = self.task_heads[self.task_type](fused_features)

        if labels is not None:
            # Compute task-specific loss
            loss = self.compute_task_loss(task_output, labels)
            return {'loss': loss, 'predictions': task_output}
        else:
            return {'predictions': task_output}

    def compute_task_loss(self, predictions, labels):
        """Compute task-specific loss"""
        if self.task_type == 'object_detection':
            # Detection loss (combination of classification and bbox regression)
            pred_bbox = predictions[:, :4]
            pred_cls = predictions[:, 4:]
            gt_bbox = labels['bbox']
            gt_cls = labels['class']

            bbox_loss = F.smooth_l1_loss(pred_bbox, gt_bbox)
            cls_loss = F.cross_entropy(pred_cls, gt_cls)
            return bbox_loss + cls_loss

        elif self.task_type == 'grasping':
            # Grasp pose loss
            return F.mse_loss(predictions, labels['grasp_pose'])

        elif self.task_type == 'navigation':
            # Navigation target loss
            return F.mse_loss(predictions, labels['target_position'])

        elif self.task_type == 'manipulation':
            # Manipulation joint loss
            return F.mse_loss(predictions, labels['joint_angles'])

        else:
            # Default: MSE loss
            return F.mse_loss(predictions, labels)

class ProgressiveFineTuning:
    """Progressive fine-tuning with gradual task complexity"""
    def __init__(self, base_model, task_hierarchy):
        self.base_model = base_model
        self.task_hierarchy = task_hierarchy  # Ordered list of tasks from simple to complex
        self.current_stage = 0

    def fine_tune_progressively(self, datasets_by_stage):
        """Fine-tune progressively through task stages"""
        for stage_idx, (task_name, dataset) in enumerate(datasets_by_stage.items()):
            print(f"Fine-tuning stage {stage_idx + 1}: {task_name}")

            # Create fine-tuner for current task
            fine_tuner = TaskSpecificFineTuner(
                base_model=self.base_model,
                task_type=task_name,
                dataset=dataset
            )

            # Train on current task
            self.train_stage(fine_tuner, dataset)

            # Evaluate performance
            performance = self.evaluate_stage(fine_tuner, dataset)

            # Check if ready to move to next stage
            if performance['accuracy'] > self.task_hierarchy[stage_idx]['threshold']:
                print(f"Stage {task_name} completed successfully")
                self.current_stage = stage_idx + 1
            else:
                print(f"Stage {task_name} needs more training")
                # Continue training on current stage

    def train_stage(self, fine_tuner, dataset):
        """Train a single stage"""
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(10):  # Few epochs per stage
            total_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()

                outputs = fine_tuner(
                    images=batch['images'].cuda(),
                    commands=batch['commands'].cuda(),
                    labels=batch['labels'].cuda()
                )

                loss = outputs['loss']
                loss.backward()

                torch.nn.utils.clip_grad_norm_(fine_tuner.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    def evaluate_stage(self, fine_tuner, dataset):
        """Evaluate performance on current stage"""
        val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.base_model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = fine_tuner(
                    images=batch['images'].cuda(),
                    commands=batch['commands'].cuda()
                )

                predictions = outputs['predictions']
                labels = batch['labels'].cuda()

                # Calculate accuracy based on task type
                if fine_tuner.task_type in ['object_detection', 'grasping', 'navigation']:
                    # For regression tasks, use threshold-based accuracy
                    accuracy = self.calculate_regression_accuracy(predictions, labels)
                else:
                    # For classification tasks
                    predicted_classes = torch.argmax(predictions, dim=-1)
                    correct = (predicted_classes == labels).sum().item()
                    total_correct += correct
                    total_samples += len(labels)

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {'accuracy': accuracy, 'total_samples': total_samples}

    def calculate_regression_accuracy(self, predictions, targets, threshold=0.1):
        """Calculate accuracy for regression tasks using threshold"""
        errors = torch.abs(predictions - targets)
        within_threshold = (errors < threshold).float().mean()
        return within_threshold.mean().item()
```

### 2. Multi-Task Learning

Training VLMs on multiple tasks simultaneously:

```python
# Multi-task learning for VLMs
class MultiTaskVLM(nn.Module):
    def __init__(self, base_model, task_configs):
        super().__init__()
        self.base_model = base_model
        self.task_configs = task_configs

        # Shared feature extractor
        self.shared_encoder = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = self.create_task_head(config)

        # Task weighting (learnable)
        self.task_weights = nn.ParameterDict({
            task_name: nn.Parameter(torch.ones(1)) for task_name in task_configs.keys()
        })

        # Uncertainty-based loss weighting
        self.uncertainty_weights = nn.ParameterDict({
            task_name: nn.Parameter(torch.zeros(1)) for task_name in task_configs.keys()
        })

    def create_task_head(self, config):
        """Create task-specific head based on configuration"""
        layers = []
        input_dim = 512  # Output of shared encoder

        for hidden_dim in config['hidden_dims']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.1))
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, config['output_dim']))

        return nn.Sequential(*layers)

    def forward(self, images, commands, task_weights=None):
        """
        Forward pass with multi-task learning
        Args:
            images: [B, C, H, W] input images
            commands: [B, seq_len] tokenized commands
            task_weights: Optional manual task weights
        Returns:
            Dictionary of task-specific outputs and losses
        """
        # Get base vision-language features
        outputs = self.base_model(
            pixel_values=images,
            input_ids=commands['input_ids'],
            attention_mask=commands['attention_mask']
        )

        # Get shared representations
        shared_features = self.shared_encoder(outputs.last_hidden_state[:, 0, :])  # [B, 512]

        # Compute task-specific outputs
        task_outputs = {}
        task_losses = {}

        for task_name in self.task_configs.keys():
            # Apply task-specific head
            task_pred = self.task_heads[task_name](shared_features)
            task_outputs[task_name] = task_pred

        return {
            'task_outputs': task_outputs,
            'shared_features': shared_features
        }

    def compute_multi_task_loss(self, task_outputs, labels):
        """Compute multi-task loss with uncertainty-based weighting"""
        total_loss = 0
        task_losses = {}

        for task_name, predictions in task_outputs.items():
            if task_name in labels:
                # Compute task-specific loss
                target = labels[task_name]

                if self.task_configs[task_name]['task_type'] == 'classification':
                    task_loss = F.cross_entropy(predictions, target)
                elif self.task_configs[task_name]['task_type'] == 'regression':
                    task_loss = F.mse_loss(predictions, target)
                elif self.task_configs[task_name]['task_type'] == 'detection':
                    task_loss = self.compute_detection_loss(predictions, target)
                else:
                    task_loss = F.mse_loss(predictions, target)

                # Apply uncertainty-based weighting
                log_var = self.uncertainty_weights[task_name]
                weighted_loss = torch.exp(-log_var) * task_loss + log_var

                task_losses[task_name] = weighted_loss
                total_loss += weighted_loss

        return total_loss, task_losses

    def compute_detection_loss(self, predictions, targets):
        """Compute detection loss (classification + bounding box regression)"""
        # Split predictions into bbox and class components
        pred_bbox = predictions[:, :4]
        pred_cls = predictions[:, 4:]

        # Compute bbox loss
        bbox_loss = F.smooth_l1_loss(pred_bbox, targets['bbox'])

        # Compute classification loss
        cls_loss = F.cross_entropy(pred_cls, targets['class'])

        return bbox_loss + cls_loss

class MultiTaskTrainer:
    def __init__(self, model, task_weights_schedule=None):
        self.model = model
        self.task_weights_schedule = task_weights_schedule or {}
        self.current_epoch = 0

        # Optimizer for multi-task learning
        self.optimizer = torch.optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': 1e-6},
            {'params': model.shared_encoder.parameters(), 'lr': 1e-5},
            {'params': model.task_heads.parameters(), 'lr': 1e-4},
            {'params': model.uncertainty_weights.parameters(), 'lr': 1e-3}  # Learn uncertainty weights
        ])

    def train_step(self, batch):
        """Multi-task training step"""
        images = batch['images'].cuda()
        commands = batch['commands'].cuda()
        labels = {k: v.cuda() for k, v in batch['labels'].items()}

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(images, commands)

        # Compute multi-task loss
        total_loss, task_losses = self.model.compute_multi_task_loss(
            outputs['task_outputs'], labels
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'task_losses': {k: v.item() for k, v in task_losses.items()}
        }

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        task_losses_accum = defaultdict(float)
        num_batches = 0

        for batch in train_loader:
            step_results = self.train_step(batch)

            total_loss += step_results['total_loss']
            for task_name, loss in step_results['task_losses'].items():
                task_losses_accum[task_name] += loss

            num_batches += 1

        avg_total_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses_accum.items()}

        return {
            'total_loss': avg_total_loss,
            'task_losses': avg_task_losses
        }

    def adaptive_task_weighting(self, task_losses):
        """Adjust task weights based on current performance"""
        # Implement uncertainty-based or gradient-based task weighting
        with torch.no_grad():
            for task_name in task_losses.keys():
                # Update uncertainty weights based on recent performance
                current_loss = task_losses[task_name]

                # If loss is decreasing, reduce the weight (less focus needed)
                # If loss is increasing, increase the weight (more focus needed)
                if hasattr(self, 'prev_task_losses'):
                    if current_loss > self.prev_task_losses.get(task_name, float('inf')):
                        self.model.uncertainty_weights[task_name] += 0.01
                    else:
                        self.model.uncertainty_weights[task_name] = max(
                            0.1, self.model.uncertainty_weights[task_name] - 0.01
                        )

        self.prev_task_losses = {k: v for k, v in task_losses.items()}
```

## Data Augmentation Strategies

### 1. Vision-Language Data Augmentation

```python
# Advanced data augmentation for vision-language pairs
import torchvision.transforms as T
from PIL import Image
import numpy as np

class VisionLanguageAugmenter:
    def __init__(self):
        # Vision augmentations
        self.vision_augmentations = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=10),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Text augmentations
        self.text_augmenter = TextAugmenter()

    def augment_batch(self, images, texts):
        """
        Augment vision-language batch
        Args:
            images: [B, C, H, W] images
            texts: List of text strings
        Returns:
            Augmented images and texts
        """
        augmented_images = []
        augmented_texts = []

        for img, text in zip(images, texts):
            # Convert tensor back to PIL for vision augmentation
            img_pil = T.ToPILImage()(denormalize_tensor(img))

            # Apply vision augmentation
            aug_img = self.vision_augmentations(img_pil)

            # Apply text augmentation
            aug_text = self.text_augmenter.augment_text(text)

            augmented_images.append(aug_img)
            augmented_texts.append(aug_text)

        return torch.stack(augmented_images), augmented_texts

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for PIL conversion"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

class TextAugmenter:
    def __init__(self):
        self.synonym_replacements = {
            'pick up': ['grasp', 'grab', 'take', 'lift'],
            'put down': ['place', 'set down', 'release', 'drop'],
            'move to': ['go to', 'navigate to', 'walk to', 'travel to'],
            'humanoid robot': ['humanoid', 'robot', 'human-like robot', 'bipedal robot'],
            'humanoid': ['human-like', 'bipedal', 'walking robot', 'two-legged robot'],
            'robot': ['machine', 'automaton', 'android', 'mechanical assistant']
        }

        self.paraphrase_templates = [
            "Please {action} the {object}",
            "Could you {action} the {object}?",
            "I'd like you to {action} the {object}",
            "Can you {action} the {object} for me?",
            "{action} the {object} please"
        ]

    def augment_text(self, text):
        """Augment text while preserving meaning"""
        augmented_text = text

        # Synonym replacement
        augmented_text = self.replace_synonyms(augmented_text)

        # Paraphrasing
        if np.random.random() < 0.3:  # 30% chance of paraphrasing
            augmented_text = self.paraphrase_sentence(augmented_text)

        # Add noise (synthetic errors)
        if np.random.random() < 0.1:  # 10% chance of adding noise
            augmented_text = self.add_typo_noise(augmented_text)

        return augmented_text

    def replace_synonyms(self, text):
        """Replace words with synonyms"""
        augmented_text = text.lower()

        for original, synonyms in self.synonym_replacements.items():
            if original in augmented_text:
                if np.random.random() < 0.5:  # 50% chance of replacement
                    replacement = np.random.choice(synonyms)
                    augmented_text = augmented_text.replace(original, replacement)

        return augmented_text

    def paraphrase_sentence(self, text):
        """Paraphrase sentence using templates"""
        # Extract action and object from text
        action = self.extract_action(text)
        obj = self.extract_object(text)

        if action and obj:
            template = np.random.choice(self.paraphrase_templates)
            return template.format(action=action, object=obj)

        return text

    def add_typo_noise(self, text):
        """Add realistic typos to text"""
        augmented_text = list(text)

        for i in range(len(augmented_text)):
            if np.random.random() < 0.05:  # 5% chance per character
                typo_type = np.random.choice(['swap', 'insert', 'delete', 'substitute'])

                if typo_type == 'swap' and i < len(augmented_text) - 1:
                    # Swap adjacent characters
                    augmented_text[i], augmented_text[i+1] = augmented_text[i+1], augmented_text[i]
                elif typo_type == 'insert':
                    # Insert random character
                    random_char = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                    augmented_text.insert(i, random_char)
                elif typo_type == 'delete':
                    # Delete character
                    augmented_text.pop(i)
                elif typo_type == 'substitute':
                    # Substitute with random character
                    random_char = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
                    augmented_text[i] = random_char

        return ''.join(augmented_text)

    def extract_action(self, text):
        """Extract action verb from text"""
        # Simple extraction - in practice, use NLP
        words = text.lower().split()
        actions = ['pick', 'put', 'move', 'grasp', 'place', 'take', 'lift', 'drop']

        for word in words:
            if word in actions:
                return word

        return None

    def extract_object(self, text):
        """Extract object noun from text"""
        # Simple extraction - in practice, use NLP
        words = text.lower().split()
        # Common objects in robotics
        objects = ['cup', 'ball', 'box', 'book', 'toy', 'object', 'item', 'thing']

        for word in reversed(words):  # Check from end
            if word in objects:
                return word

        return None

# Domain randomization for simulation-to-reality transfer
class DomainRandomization:
    def __init__(self):
        self.lighting_params = {
            'brightness_range': (0.5, 1.5),
            'contrast_range': (0.8, 1.2),
            'saturation_range': (0.8, 1.2),
            'hue_range': (-0.1, 0.1),
            'light_direction_range': (-180, 180)  # degrees
        }

        self.texture_params = {
            'roughness_range': (0.0, 1.0),
            'metallic_range': (0.0, 1.0),
            'normal_map_strength': (0.0, 1.0)
        }

        self.camera_params = {
            'fov_range': (45, 60),  # degrees
            'focus_range': (0.1, 10.0),  # meters
            'noise_level_range': (0.0, 0.05)
        }

    def randomize_lighting(self, image):
        """Apply random lighting conditions"""
        brightness_factor = np.random.uniform(*self.lighting_params['brightness_range'])
        contrast_factor = np.random.uniform(*self.lighting_params['contrast_range'])
        saturation_factor = np.random.uniform(*self.lighting_params['saturation_range'])
        hue_factor = np.random.uniform(*self.lighting_params['hue_range'])

        # Apply transformations using PIL
        img_pil = T.ToPILImage()(denormalize_tensor(image))

        # Brightness
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness_factor)

        # Contrast
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast_factor)

        # Saturation
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(saturation_factor)

        # Convert back to tensor
        tensor_img = T.ToTensor()(img_pil)
        return normalize_tensor(tensor_img)

    def randomize_camera_effects(self, image):
        """Apply random camera effects"""
        # Add noise
        noise_level = np.random.uniform(0, self.camera_params['noise_level_range'][1])
        noise = torch.randn_like(image) * noise_level
        noisy_image = torch.clamp(image + noise, 0, 1)

        # Apply blur
        blur_radius = np.random.uniform(0, 1.0)
        if blur_radius > 0.1:
            kernel_size = int(blur_radius * 10) | 1  # Ensure odd kernel size
            blurred = T.GaussianBlur(kernel_size=kernel_size)(noisy_image)
            return blurred

        return noisy_image

    def randomize_texture(self, texture_map):
        """Randomize texture properties"""
        # This would typically modify material properties in a 3D renderer
        # For now, we'll apply texture-like filters to 2D images
        roughness = np.random.uniform(*self.texture_params['roughness_range'])
        metallic = np.random.uniform(*self.texture_params['metallic_range'])

        # Apply texture effects (simplified)
        texture_effect = torch.ones_like(texture_map) * metallic
        return texture_map * (1 - roughness) + texture_effect * roughness

# MixUp-style augmentation for vision-language pairs
class VisionLanguageMixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def mix_batch(self, images, texts, labels):
        """
        Mix vision-language pairs in the batch
        Args:
            images: [B, C, H, W] images
            texts: List of text strings
            labels: Dict of task-specific labels
        Returns:
            Mixed images, texts, and labels with mixing coefficients
        """
        batch_size = images.size(0)
        lambda_param = np.random.beta(self.alpha, self.alpha)

        # Random permutation
        perm = torch.randperm(batch_size)

        # Mix images
        mixed_images = lambda_param * images + (1 - lambda_param) * images[perm]

        # Mix texts (interpolate between texts)
        mixed_texts = []
        for i in range(batch_size):
            mixed_text = self.interpolate_texts(texts[i], texts[perm[i]], lambda_param)
            mixed_texts.append(mixed_text)

        # Mix labels
        mixed_labels = self.mix_labels(labels, perm, lambda_param)

        return mixed_images, mixed_texts, mixed_labels, lambda_param

    def interpolate_texts(self, text1, text2, lambda_param):
        """Interpolate between two texts"""
        # This is a simplified approach - in practice, use more sophisticated text mixing
        if lambda_param > 0.7:
            return text1
        elif lambda_param < 0.3:
            return text2
        else:
            # Combine both texts
            return f"{text1} and {text2}"

    def mix_labels(self, labels, perm, lambda_param):
        """Mix task-specific labels"""
        mixed_labels = {}

        for task_name, task_labels in labels.items():
            if isinstance(task_labels, torch.Tensor):
                # For continuous values, interpolate
                mixed_labels[task_name] = lambda_param * task_labels + (1 - lambda_param) * task_labels[perm]
            elif isinstance(task_labels, list):
                # For discrete labels, use majority vote or random selection
                mixed_labels[task_name] = [
                    lbl if np.random.random() < lambda_param else task_labels[perm[i]]
                    for i, lbl in enumerate(task_labels)
                ]
            else:
                mixed_labels[task_name] = task_labels  # Fallback

        return mixed_labels
```

## Curriculum Learning for VLMs

### 1. Difficulty-Based Curriculum

```python
# Curriculum learning for VLM training
import numpy as np
from torch.utils.data import Subset
import random

class VLACurriculum:
    def __init__(self, dataset, difficulty_metrics):
        self.dataset = dataset
        self.difficulty_metrics = difficulty_metrics
        self.current_stage = 0
        self.performance_history = []
        self.stage_thresholds = [0.6, 0.7, 0.8, 0.85]  # Performance thresholds for advancing

    def create_curriculum_stages(self):
        """Create curriculum stages based on difficulty"""
        # Calculate difficulty scores for each sample
        difficulty_scores = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            difficulty = self.calculate_difficulty(sample)
            difficulty_scores.append(difficulty)

        # Sort samples by difficulty
        sorted_indices = np.argsort(difficulty_scores)

        # Create stages with increasing difficulty
        num_stages = 4
        stage_size = len(sorted_indices) // num_stages

        self.curriculum_stages = []
        for stage_idx in range(num_stages):
            start_idx = stage_idx * stage_size
            end_idx = (stage_idx + 1) * stage_size if stage_idx < num_stages - 1 else len(sorted_indices)
            stage_indices = sorted_indices[start_idx:end_idx]
            self.curriculum_stages.append(stage_indices)

    def calculate_difficulty(self, sample):
        """Calculate difficulty score for a sample"""
        difficulty = 0.0

        # Visual complexity (number of objects, occlusion, etc.)
        if 'num_objects' in sample:
            difficulty += sample['num_objects'] * 0.1

        # Text complexity (length, vocabulary, etc.)
        if 'text' in sample:
            text_length = len(sample['text'].split())
            difficulty += min(text_length * 0.01, 0.5)  # Cap text difficulty

        # Task complexity (number of steps, precision required, etc.)
        if 'task_complexity' in sample:
            difficulty += sample['task_complexity'] * 0.3

        # Scene complexity (clutter, lighting conditions, etc.)
        if 'scene_complexity' in sample:
            difficulty += sample['scene_complexity'] * 0.2

        return difficulty

    def get_current_stage_dataset(self):
        """Get dataset for current curriculum stage"""
        current_indices = self.curriculum_stages[self.current_stage]
        return Subset(self.dataset, current_indices)

    def should_advance_stage(self, current_performance):
        """Check if should advance to next curriculum stage"""
        self.performance_history.append(current_performance)

        # Check if performance is consistently above threshold
        if len(self.performance_history) >= 5:  # Need 5 evaluations
            recent_performance = np.mean(self.performance_history[-5:])
            threshold = self.stage_thresholds[self.current_stage] if self.current_stage < len(self.stage_thresholds) else 0.9

            if recent_performance >= threshold:
                return True

        return False

    def advance_stage(self):
        """Advance to next curriculum stage"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            print(f"Advancing to curriculum stage {self.current_stage + 1}")
            return True
        else:
            print("Reached final curriculum stage")
            return False

class ProgressiveDifficultySampler:
    """Dynamically adjust difficulty based on model performance"""
    def __init__(self, dataset, initial_difficulty=0.3, growth_rate=0.05):
        self.dataset = dataset
        self.initial_difficulty = initial_difficulty
        self.growth_rate = growth_rate
        self.current_difficulty = initial_difficulty
        self.performance_history = []

    def get_difficulty_based_sample(self, target_difficulty, num_samples=32):
        """Get samples with difficulty close to target difficulty"""
        # Calculate difficulties for all samples
        difficulties = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            difficulty = self.calculate_sample_difficulty(sample)
            difficulties.append((i, difficulty))

        # Sort by difficulty difference from target
        difficulties.sort(key=lambda x: abs(x[1] - target_difficulty))

        # Select samples with difficulty close to target
        selected_indices = [idx for idx, diff in difficulties[:num_samples]]

        return Subset(self.dataset, selected_indices)

    def calculate_sample_difficulty(self, sample):
        """Calculate difficulty of a single sample"""
        # This would implement a comprehensive difficulty assessment
        # For now, return a placeholder
        return np.random.random()  # Placeholder

    def update_difficulty(self, current_performance):
        """Update target difficulty based on performance"""
        # If performance is high, increase difficulty
        if current_performance > 0.8:
            self.current_difficulty = min(1.0, self.current_difficulty + self.growth_rate)
        # If performance is low, decrease difficulty
        elif current_performance < 0.5:
            self.current_difficulty = max(0.1, self.current_difficulty - self.growth_rate)

        # Gradual progression
        self.current_difficulty = min(
            self.current_difficulty,
            self.current_difficulty + self.growth_rate * (current_performance - 0.5)
        )

        return self.current_difficulty
```

## Training Optimization Techniques

### 1. Advanced Optimization Strategies

```python
# Advanced optimization techniques for VLM training
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing scheduler with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            multiplier = self.last_epoch / float(max(1, self.warmup_epochs))
            return [base_lr * multiplier for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]

class GradientAccumulationOptimizer:
    """Gradient accumulation wrapper for memory-efficient training"""
    def __init__(self, optimizer, accumulation_steps=4):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            # Scale gradients for accumulation
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad /= self.accumulation_steps

            # Step optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()  # Reset gradients after step

class AdaptiveBatchSizeScheduler:
    """Dynamically adjust batch size based on memory usage"""
    def __init__(self, initial_batch_size=32, min_batch_size=8, max_batch_size=128):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_history = []

    def adjust_batch_size(self, memory_usage_gb, target_memory_gb=10.0):
        """Adjust batch size based on memory usage"""
        memory_ratio = memory_usage_gb / target_memory_gb

        if memory_ratio > 0.9:  # Memory usage too high
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        elif memory_ratio < 0.6:  # Memory usage low, can increase
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size * 2)

        return self.current_batch_size

class VLAMixedPrecisionTrainer:
    """Mixed precision training for VLM models"""
    def __init__(self, model, optimizer, loss_scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = loss_scaler or torch.cuda.amp.GradScaler()

    def train_step(self, batch):
        """Single training step with mixed precision"""
        images = batch['images'].cuda()
        commands = batch['commands'].cuda()
        labels = batch['labels']

        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(images, commands)
            loss = self.compute_loss(outputs, labels)

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Unscaled gradients for clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def compute_loss(self, outputs, labels):
        """Compute loss for mixed precision training"""
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        else:
            # Compute task-specific loss
            task_outputs = outputs['task_outputs']
            total_loss = 0

            for task_name, predictions in task_outputs.items():
                if task_name in labels:
                    target = labels[task_name]
                    if task_name == 'detection':
                        task_loss = F.mse_loss(predictions, target)
                    elif task_name == 'classification':
                        task_loss = F.cross_entropy(predictions, target)
                    else:
                        task_loss = F.mse_loss(predictions, target)

                    total_loss += task_loss

            return total_loss

# Knowledge distillation for VLM compression
class KnowledgeDistillationTrainer:
    """Train smaller student model using larger teacher model"""
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft target loss

        # Freeze teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        """Compute knowledge distillation loss"""
        # Soft target loss (KL divergence between teacher and student predictions)
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=-1)
        student_logits = student_outputs / self.temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard target loss (standard cross-entropy with true labels)
        hard_loss = F.cross_entropy(student_outputs, labels)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss, soft_loss, hard_loss

    def train_step(self, batch):
        """Knowledge distillation training step"""
        images = batch['images'].cuda()
        commands = batch['commands'].cuda()
        labels = batch['labels'].cuda()

        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(images, commands)

        # Get student predictions
        student_outputs = self.student_model(images, commands)

        # Compute distillation loss
        loss, soft_loss, hard_loss = self.distillation_loss(
            student_outputs, teacher_outputs, labels
        )

        return {
            'total_loss': loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss
        }
```

## Evaluation and Validation

### 1. Comprehensive Evaluation Framework

```python
# Comprehensive evaluation framework for VLM models
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class VLAEvaluationFramework:
    def __init__(self, model, test_datasets):
        self.model = model
        self.test_datasets = test_datasets

    def comprehensive_evaluation(self):
        """Perform comprehensive evaluation across multiple metrics and datasets"""
        results = {}

        for dataset_name, dataset_loader in self.test_datasets.items():
            print(f"Evaluating on {dataset_name}...")
            dataset_results = self.evaluate_dataset(dataset_loader)
            results[dataset_name] = dataset_results

        # Compute overall metrics
        overall_metrics = self.compute_overall_metrics(results)
        results['overall'] = overall_metrics

        return results

    def evaluate_dataset(self, data_loader):
        """Evaluate model on a specific dataset"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_features = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch['images'].cuda()
                commands = batch['commands'].cuda()
                targets = batch['labels']

                # Get model predictions
                outputs = self.model(images, commands)

                if isinstance(outputs, dict) and 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    predictions = outputs

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

                # Store features for additional analysis
                if 'features' in outputs:
                    all_features.append(outputs['features'].cpu())

        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        if all_features:
            all_features = torch.cat(all_features, dim=0)

        # Compute metrics
        metrics = self.compute_task_metrics(all_predictions, all_targets)

        # Additional analysis
        if all_features:
            feature_analysis = self.analyze_features(all_features, all_targets)
            metrics.update(feature_analysis)

        return metrics

    def compute_task_metrics(self, predictions, targets):
        """Compute task-specific metrics"""
        metrics = {}

        # For classification tasks
        if predictions.dim() == 2 and predictions.size(1) > 1:  # Likely classification
            pred_classes = torch.argmax(predictions, dim=1)
            target_classes = targets

            accuracy = (pred_classes == target_classes).float().mean().item()
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_classes.numpy(), pred_classes.numpy(), average='weighted'
            )

            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

        # For regression tasks
        else:
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()
            rmse = np.sqrt(mse)

            metrics.update({
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            })

        return metrics

    def analyze_features(self, features, targets):
        """Analyze feature representations"""
        analysis = {}

        # Feature diversity (how varied are the representations)
        feature_std = torch.std(features, dim=0).mean().item()
        analysis['feature_diversity'] = feature_std

        # Feature separability (how well different classes are separated)
        if targets.dim() == 1:  # Classification task
            unique_targets = torch.unique(targets)
            inter_class_distances = []
            intra_class_distances = []

            for target_val in unique_targets:
                class_mask = (targets == target_val)
                class_features = features[class_mask]

                if len(class_features) > 1:
                    # Intra-class distances
                    for i in range(len(class_features)):
                        for j in range(i + 1, len(class_features)):
                            dist = torch.norm(class_features[i] - class_features[j])
                            intra_class_distances.append(dist.item())

            # Inter-class distances
            for i in range(len(unique_targets)):
                for j in range(i + 1, len(unique_targets)):
                    class1_mask = (targets == unique_targets[i])
                    class2_mask = (targets == unique_targets[j])
                    class1_features = features[class1_mask]
                    class2_features = features[class2_mask]

                    if len(class1_features) > 0 and len(class2_features) > 0:
                        # Compute average distance between classes
                        for f1 in class1_features[:5]:  # Sample 5 from each class
                            for f2 in class2_features[:5]:
                                dist = torch.norm(f1 - f2)
                                inter_class_distances.append(dist.item())

            if inter_class_distances and intra_class_distances:
                separability = np.mean(inter_class_distances) / (np.mean(intra_class_distances) + 1e-8)
                analysis['class_separability'] = separability

        return analysis

    def compute_cross_modal_alignment(self, vision_features, language_features):
        """Compute alignment between vision and language features"""
        # Compute similarity matrix
        similarity_matrix = torch.matmul(vision_features, language_features.t())

        # Compute alignment metrics
        # Diagonal elements should be higher (matching pairs)
        diagonal_similarities = torch.diag(similarity_matrix)
        off_diagonal_similarities = similarity_matrix[~torch.eye(similarity_matrix.size(0), dtype=torch.bool)]

        alignment_score = torch.mean(diagonal_similarities) - torch.mean(off_diagonal_similarities)

        # Compute correlation between vision and language features
        vision_norm = F.normalize(vision_features, dim=1)
        language_norm = F.normalize(language_features, dim=1)
        correlation = torch.mean(torch.sum(vision_norm * language_norm, dim=1))

        return {
            'alignment_score': alignment_score.item(),
            'correlation': correlation.item(),
            'diagonal_mean': torch.mean(diagonal_similarities).item(),
            'off_diagonal_mean': torch.mean(off_diagonal_similarities).item()
        }

    def evaluate_zeroshot_transfer(self, source_tasks, target_tasks):
        """Evaluate zero-shot transfer capability"""
        transfer_results = {}

        for source_task in source_tasks:
            for target_task in target_tasks:
                # Evaluate transfer from source to target
                transfer_acc = self.evaluate_transfer(source_task, target_task)
                transfer_results[f"{source_task}_to_{target_task}"] = transfer_acc

        return transfer_results

    def evaluate_robustness(self, test_loader, noise_levels=[0.0, 0.1, 0.2, 0.3]):
        """Evaluate model robustness to noise"""
        robustness_results = {}

        for noise_level in noise_levels:
            noisy_loader = self.add_noise_to_loader(test_loader, noise_level)
            metrics = self.evaluate_dataset(noisy_loader)
            robustness_results[f"noise_{noise_level}"] = metrics

        # Compute robustness score
        clean_acc = robustness_results["noise_0.0"].get('accuracy', 0)
        noisy_acc = [robustness_results[f"noise_{nl}"].get('accuracy', 0) for nl in noise_levels[1:]]
        robustness_score = sum(noisy_acc) / len(noisy_acc) / clean_acc if clean_acc > 0 else 0

        robustness_results['robustness_score'] = robustness_score

        return robustness_results

    def add_noise_to_loader(self, loader, noise_level):
        """Add noise to data loader for robustness evaluation"""
        # This would wrap the original loader to add noise
        # Implementation depends on the specific loader type
        pass
```

## Deployment Considerations

### 1. Model Compression for Edge Deployment

```python
# Model compression techniques for humanoid robotics
import torch
import torch.nn.utils.prune as prune

class VLAModelCompressor:
    """Compress VLA models for deployment on humanoid robots"""
    def __init__(self, model):
        self.model = model

    def prune_model(self, pruning_ratio=0.3, method='magnitude'):
        """Prune model weights"""
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        if method == 'magnitude':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio
            )
        elif method == 'random':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_ratio
            )

        return self.model

    def quantize_model(self, backend='fbgemm'):
        """Quantize model for faster inference"""
        # Set model to evaluation mode
        self.model.eval()

        # Fuse modules for better quantization
        self.fuse_modules()

        # Prepare for quantization
        torch.quantization.prepare(self.model, inplace=True)

        # Calibrate with sample data
        self.calibrate_model()

        # Convert to quantized model
        torch.quantization.convert(self.model, inplace=True)

        return self.model

    def fuse_modules(self):
        """Fuse modules for better quantization"""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                # Fuse Conv-BN-ReLU modules
                torch.quantization.fuse_modules(module, [['conv', 'bn', 'relu']], inplace=True)

    def calibrate_model(self):
        """Calibrate model for quantization"""
        # This would run the model on calibration data
        # For now, we'll use dummy data
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = self.model(dummy_input)

    def create_sparse_model(self):
        """Create sparse model using sparse tensors"""
        # Convert dense tensors to sparse where appropriate
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Apply sparsity pattern
                sparse_param = self.create_sparse_tensor(param)
                # Replace parameter with sparse version
                delattr(param)
                setattr(self.model, name.replace('.', '_'), sparse_param)

        return self.model

    def create_sparse_tensor(self, dense_tensor, sparsity=0.5):
        """Create sparse tensor with specified sparsity"""
        # Create random sparsity pattern
        mask = torch.rand_like(dense_tensor) > sparsity
        sparse_tensor = dense_tensor * mask
        return sparse_tensor.to_sparse()

class NeuralArchitectureSearch:
    """Neural Architecture Search for optimal VLA architectures"""
    def __init__(self, search_space, validation_loader):
        self.search_space = search_space
        self.validation_loader = validation_loader
        self.best_architecture = None
        self.best_performance = float('-inf')

    def search_architecture(self, num_trials=100):
        """Search for optimal architecture"""
        for trial in range(num_trials):
            # Sample architecture from search space
            architecture = self.sample_architecture()

            # Build model
            model = self.build_model_from_architecture(architecture)

            # Train model briefly
            performance = self.evaluate_architecture(model)

            # Update best architecture if better
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_architecture = architecture
                print(f"New best architecture found with performance: {performance:.4f}")

        return self.best_architecture

    def sample_architecture(self):
        """Sample architecture from search space"""
        architecture = {}
        for component, options in self.search_space.items():
            architecture[component] = np.random.choice(options)
        return architecture

    def build_model_from_architecture(self, architecture):
        """Build model based on architecture specification"""
        # This would build a model based on the architecture spec
        # For now, return a placeholder
        pass

    def evaluate_architecture(self, model):
        """Briefly evaluate architecture"""
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                images = batch['images'].cuda()
                commands = batch['commands'].cuda()
                targets = batch['labels'].cuda()

                outputs = model(images, commands)
                loss = F.mse_loss(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 10:  # Only evaluate on first 10 batches
                    break

        avg_loss = total_loss / num_batches
        return -avg_loss  # Return negative loss (higher is better)
```

## Next Steps

In the next section, we'll explore deployment strategies for VLA models on humanoid robots, learning how to optimize these models for real-time execution on embedded systems and how to handle the unique challenges of deploying vision-language-action systems in physical humanoid platforms.