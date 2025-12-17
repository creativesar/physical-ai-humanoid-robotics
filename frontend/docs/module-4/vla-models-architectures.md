---
sidebar_position: 5
title: "VLA Models and Architectures"
---

# VLA Models and Architectures

## Introduction to VLA Models

Vision-Language-Action (VLA) models represent a breakthrough in robotics AI, enabling robots to directly map visual observations and natural language commands to executable actions. These models eliminate the need for separate perception, planning, and control modules by learning end-to-end policies that can understand complex visual scenes, interpret linguistic instructions, and generate appropriate motor commands in a unified framework.

The key insight behind VLA models is that they can be trained on large-scale datasets of human demonstrations, allowing them to learn complex behaviors without explicit programming. This approach has shown remarkable success in enabling robots to perform diverse tasks with minimal task-specific engineering.

## Foundational VLA Architectures

### 1. OpenVLA: Open-Source Foundation Model

OpenVLA represents the current state-of-the-art in open-source VLA models, providing a foundation for vision-language-action learning:

```python
# OpenVLA-inspired architecture
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor
import numpy as np

class OpenVLA(nn.Module):
    def __init__(self, vision_model, language_model, action_dim=14, hidden_dim=512):
        super().__init__()

        # Vision encoder (from CLIP)
        self.vision_encoder = vision_model
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)

        # Language encoder (from CLIP)
        self.language_encoder = language_model
        self.language_proj = nn.Linear(self.language_encoder.config.hidden_size, hidden_dim)

        # Fusion module
        self.fusion = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for fused features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Temporal processing for action sequences
        self.temporal_encoder = nn.LSTM(
            input_size=action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Task conditioning
        self.task_embedding = nn.Embedding(100, hidden_dim)  # 100 different tasks

        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, images, commands, task_ids=None, return_attention=False):
        """
        Forward pass through OpenVLA model
        Args:
            images: [B, C, H, W] visual observations
            commands: [B, seq_len] tokenized language commands
            task_ids: [B] task identifiers
            return_attention: whether to return attention weights
        Returns:
            action predictions [B, action_dim] and optional attention weights
        """
        B = images.shape[0]

        # Encode visual features
        vision_outputs = self.vision_encoder(pixel_values=images, output_hidden_states=True)
        vision_features = vision_outputs.last_hidden_state  # [B, num_patches, hidden_size]

        # Project vision features
        vision_proj = self.vision_proj(vision_features)  # [B, num_patches, hidden_dim]

        # Encode language features
        language_outputs = self.language_encoder(input_ids=commands, output_hidden_states=True)
        language_features = language_outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # Project language features
        language_proj = self.language_proj(language_features)  # [B, seq_len, hidden_dim]

        # Task conditioning
        if task_ids is not None:
            task_emb = self.task_embedding(task_ids)  # [B, hidden_dim]
            task_emb = task_emb.unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            task_emb = torch.zeros(B, 1, hidden_dim, device=images.device)

        # Concatenate all features
        all_features = torch.cat([vision_proj, language_proj, task_emb], dim=1)  # [B, total_len, hidden_dim]

        # Apply fusion attention
        fused_features, attention_weights = self.fusion(
            query=all_features.transpose(0, 1),
            key=all_features.transpose(0, 1),
            value=all_features.transpose(0, 1)
        )
        fused_features = fused_features.transpose(0, 1)  # [B, total_len, hidden_dim]

        # Global feature aggregation
        global_features = torch.mean(fused_features, dim=1)  # [B, hidden_dim]

        # Decode to action space
        action_pred = self.action_decoder(global_features)  # [B, action_dim]

        if return_attention:
            return action_pred, attention_weights
        else:
            return action_pred

    def encode_state(self, image, command):
        """Encode visual and linguistic state into a unified representation"""
        vision_features = self.vision_encoder(pixel_values=image).last_hidden_state
        vision_proj = self.vision_proj(vision_features)

        language_features = self.language_encoder(input_ids=command).last_hidden_state
        language_proj = self.language_proj(language_features)

        # Simple concatenation for state representation
        state_features = torch.cat([vision_proj, language_proj], dim=1)
        return state_features

    def predict_action_sequence(self, images, commands, sequence_length=5):
        """Predict a sequence of actions"""
        action_sequence = []

        for i in range(sequence_length):
            # Get current action prediction
            current_action = self.forward(images, commands)
            action_sequence.append(current_action)

            # In a real implementation, you'd update the state based on the action
            # This is a simplified version that assumes static state for demonstration

        return torch.stack(action_sequence, dim=1)  # [B, seq_len, action_dim]

# OpenVLA training setup
class OpenVLA_Trainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    def train_step(self, batch):
        """
        Training step for OpenVLA model
        Args:
            batch: dict with 'images', 'commands', 'actions'
        """
        images = batch['images']
        commands = batch['commands']
        true_actions = batch['actions']

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision training
            pred_actions = self.model(images, commands)
            loss = self.criterion(pred_actions, true_actions)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss

        return total_loss / len(dataloader)
```

### 2. RT-1: Robot Transformer for Real-World Control

RT-1 (Robotics Transformer 1) introduced the concept of using transformer architectures for robotic control:

```python
# RT-1 inspired architecture
import torch
import torch.nn as nn
import math

class RT1(nn.Module):
    def __init__(self, vocab_size=32000, max_seq_len=512, action_dim=14, d_model=512, nhead=8, num_layers=6):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Vision encoder
        self.vision_encoder = VisionTransformerEncoder(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=d_model
        )

        # Language encoder (adapted from transformer)
        self.command_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action decoder
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim)
        )

        # Task conditioning
        self.task_embedding = nn.Embedding(50, d_model)  # 50 different tasks

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, images, commands, task_ids=None):
        """
        Forward pass through RT-1 model
        Args:
            images: [B, C, H, W] visual observations
            commands: [B, seq_len] tokenized language commands
            task_ids: [B] task identifiers
        Returns:
            action predictions [B, action_dim]
        """
        B, _, H, W = images.shape
        B, seq_len = commands.shape

        # Encode visual features
        vision_features = self.vision_encoder(images)  # [B, num_patches, d_model]

        # Encode language features
        command_embeds = self.command_embedding(commands)  # [B, seq_len, d_model]

        # Add position encoding to commands
        positions = torch.arange(seq_len, device=commands.device).unsqueeze(0)
        pos_embeds = self.position_encoding(positions)  # [1, seq_len, d_model]
        command_embeds = command_embeds + pos_embeds  # [B, seq_len, d_model]

        # Task conditioning
        if task_ids is not None:
            task_embeds = self.task_embedding(task_ids)  # [B, d_model]
            task_embeds = task_embeds.unsqueeze(1)  # [B, 1, d_model]
        else:
            task_embeds = torch.zeros(B, 1, self.d_model, device=images.device)

        # Concatenate all modalities
        all_features = torch.cat([vision_features, command_embeds, task_embeds], dim=1)  # [B, total_len, d_model]

        # Apply transformer
        transformer_output = self.transformer(all_features)  # [B, total_len, d_model]

        # Global pooling for action prediction
        global_features = torch.mean(transformer_output, dim=1)  # [B, d_model]

        # Predict actions
        actions = self.action_head(global_features)  # [B, action_dim]

        return actions

class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim)  # +1 for CLS token
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embedding(x)  # [B, hidden_dim, num_patches_h, num_patches_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, hidden_dim]

        # Add positional embedding
        x = x + self.pos_embedding

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final normalization
        x = self.norm(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)

        return x

# RT-1 training and inference
class RT1Trainer:
    def __init__(self, model, learning_rate=1e-4, warmup_steps=1000, total_steps=100000):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps - warmup_steps
                )
            ],
            milestones=[warmup_steps]
        )

        self.criterion = nn.MSELoss()

    def compute_loss(self, predictions, targets, mask=None):
        """Compute weighted loss for action prediction"""
        if mask is not None:
            # Apply mask to ignore padded elements
            loss = self.criterion(predictions * mask, targets * mask)
        else:
            loss = self.criterion(predictions, targets)

        return loss

    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        images = batch['images']
        commands = batch['commands']
        actions = batch['actions']
        task_ids = batch.get('task_ids')

        # Forward pass
        pred_actions = self.model(images, commands, task_ids)

        # Compute loss
        loss = self.compute_loss(pred_actions, actions)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate(self, eval_dataloader):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                images = batch['images']
                commands = batch['commands']
                actions = batch['actions']

                pred_actions = self.model(images, commands)
                loss = self.compute_loss(pred_actions, actions)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss
```

### 3. BC-Z: Behavior Cloning with Zero-Shot Generalization

BC-Z extends behavior cloning with zero-shot generalization capabilities:

```python
# BC-Z inspired architecture for zero-shot generalization
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BCZ(nn.Module):
    def __init__(self, vision_backbone='resnet50', language_model='bert-base-uncased',
                 action_dim=14, hidden_dim=512):
        super().__init__()

        # Vision backbone
        from torchvision.models import resnet50
        self.vision_backbone = resnet50(pretrained=True)
        self.vision_backbone.fc = nn.Identity()  # Remove classification head
        self.vision_projector = nn.Linear(2048, hidden_dim)  # ResNet50 output is 2048

        # Language model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.language_model = AutoModel.from_pretrained(language_model)
        self.language_projector = nn.Linear(
            self.language_model.config.hidden_size, hidden_dim
        )

        # Cross-modal attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for vision-language concat
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Skill library for zero-shot generalization
        self.skill_library = SkillLibrary(hidden_dim)

        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, images, commands, return_skill_weights=False):
        """
        Forward pass through BC-Z model
        Args:
            images: [B, C, H, W] visual observations
            commands: [B, seq_len] tokenized language commands
            return_skill_weights: whether to return skill composition weights
        Returns:
            action predictions [B, action_dim]
        """
        B = images.shape[0]

        # Extract visual features
        vision_features = self.vision_backbone(images)  # [B, 2048]
        vision_features = vision_features.unsqueeze(1)  # [B, 1, 2048]
        vision_proj = self.vision_projector(vision_features)  # [B, 1, hidden_dim]

        # Extract language features
        language_outputs = self.language_model(input_ids=commands)
        language_features = language_outputs.last_hidden_state  # [B, seq_len, hidden_size]
        language_proj = self.language_projector(language_features)  # [B, seq_len, hidden_dim]

        # Cross-attention fusion
        fused_features, attention_weights = self.cross_attention(
            query=vision_proj.transpose(0, 1),  # [1, B, hidden_dim]
            key=language_proj.transpose(0, 1),  # [seq_len, B, hidden_dim]
            value=language_proj.transpose(0, 1)  # [seq_len, B, hidden_dim]
        )
        fused_features = fused_features.transpose(0, 1)  # [B, 1, hidden_dim]

        # Concatenate vision and fused features
        combined_features = torch.cat([
            vision_proj, fused_features
        ], dim=-1)  # [B, 1, hidden_dim * 2]

        # Apply normalization
        combined_features = self.layer_norm(combined_features.squeeze(1))  # [B, hidden_dim * 2]

        # Predict actions
        actions = self.action_head(combined_features)  # [B, action_dim]

        if return_skill_weights:
            # Get skill composition weights
            skill_weights = self.compute_skill_weights(vision_proj, language_proj)
            return actions, skill_weights
        else:
            return actions

    def compute_skill_weights(self, vision_features, language_features):
        """Compute weights for composing skills from the skill library"""
        # Average language features for skill matching
        lang_mean = torch.mean(language_features, dim=1, keepdim=True)  # [B, 1, hidden_dim]

        # Compute similarity with skills
        skill_similarities = self.skill_library.compute_similarities(lang_mean)

        # Apply softmax to get weights
        skill_weights = torch.softmax(skill_similarities, dim=-1)

        return skill_weights

    def zero_shot_generalization(self, novel_command):
        """Perform zero-shot generalization for novel commands"""
        # Tokenize novel command
        inputs = self.tokenizer(novel_command, return_tensors='pt', padding=True, truncation=True)

        # Get skill composition for novel command
        language_features = self.language_model(input_ids=inputs['input_ids']).last_hidden_state
        language_proj = self.language_projector(language_features)
        lang_mean = torch.mean(language_proj, dim=1, keepdim=True)

        # Get skill weights
        skill_weights = self.skill_library.compute_similarities(lang_mean)
        skill_weights = torch.softmax(skill_weights, dim=-1)

        # Compose skills based on weights
        composed_action = self.skill_library.compose_skills(skill_weights)

        return composed_action

class SkillLibrary(nn.Module):
    def __init__(self, hidden_dim, num_skills=50):
        super().__init__()
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim

        # Skill embeddings
        self.skill_embeddings = nn.Parameter(torch.randn(num_skills, hidden_dim))

        # Skill-specific action decoders
        self.skill_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 14)  # Action dimension
            ) for _ in range(num_skills)
        ])

    def compute_similarities(self, query_features):
        """Compute similarities between query and all skills"""
        # query_features: [B, 1, hidden_dim]
        # skill_embeddings: [num_skills, hidden_dim]
        similarities = torch.matmul(query_features, self.skill_embeddings.t())  # [B, 1, num_skills]
        return similarities.squeeze(1)  # [B, num_skills]

    def compose_skills(self, weights):
        """Compose skills based on weights"""
        # weights: [B, num_skills]
        B, num_skills = weights.shape

        # Get action for each skill
        skill_actions = []
        for i, decoder in enumerate(self.skill_decoders):
            skill_feature = self.skill_embeddings[i:i+1].expand(B, -1)  # [B, hidden_dim]
            skill_action = decoder(skill_feature)  # [B, action_dim]
            skill_actions.append(skill_action)

        skill_actions = torch.stack(skill_actions, dim=1)  # [B, num_skills, action_dim]

        # Weighted composition
        weighted_actions = weights.unsqueeze(-1) * skill_actions  # [B, num_skills, action_dim]
        composed_action = torch.sum(weighted_actions, dim=1)  # [B, action_dim]

        return composed_action

    def add_skill(self, skill_embedding, skill_decoder_weights):
        """Add a new skill to the library"""
        # This would involve expanding the skill library
        pass

    def update_skill(self, skill_id, new_embedding, new_decoder):
        """Update an existing skill"""
        if skill_id < self.num_skills:
            self.skill_embeddings[skill_id] = new_embedding
            self.skill_decoders[skill_id] = new_decoder
```

## Advanced VLA Architectures

### 1. Diffusion-Based Action Generation

Diffusion models have shown promise for generating complex robotic actions:

```python
# Diffusion-based action generation for VLA
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionActionGenerator(nn.Module):
    def __init__(self, vision_dim=512, language_dim=768, action_dim=14,
                 time_embed_dim=128, num_diffusion_steps=100):
        super().__init__()

        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Condition embedding (vision + language)
        self.condition_proj = nn.Linear(vision_dim + language_dim, time_embed_dim)

        # Denoising network
        self.denoising_net = DenoisingUNet(
            in_channels=action_dim,
            model_channels=time_embed_dim,
            out_channels=action_dim,
            num_res_blocks=2,
            attention_resolutions=[],
            channel_mult=[1, 2, 2],
            num_heads=4
        )

        # Final projection
        self.final_proj = nn.Linear(time_embed_dim, action_dim)

    def forward(self, vision_features, language_features, num_steps=None):
        """
        Generate actions using diffusion process
        Args:
            vision_features: [B, vision_dim] visual features
            language_features: [B, seq_len, language_dim] language features
            num_steps: number of diffusion steps (default: use all)
        Returns:
            action sequence [B, action_dim]
        """
        B = vision_features.shape[0]
        num_steps = num_steps or self.num_diffusion_steps

        # Combine vision and language features
        lang_pooled = torch.mean(language_features, dim=1)  # [B, language_dim]
        combined_cond = torch.cat([vision_features, lang_pooled], dim=-1)  # [B, vision_dim + language_dim]
        cond_embedding = self.condition_proj(combined_cond)  # [B, time_embed_dim]

        # Start with random noise
        x = torch.randn(B, self.action_dim, device=vision_features.device)

        # Reverse diffusion process
        for i in range(num_steps - 1, -1, -1):
            t = torch.full((B,), i, device=vision_features.device, dtype=torch.long)
            time_embedding = self.time_embed(timestep_embedding(t, self.time_embed.shape[-1]))

            # Add condition to time embedding
            full_cond = time_embedding + cond_embedding

            # Denoise step
            noise_pred = self.denoising_net(x, full_cond)

            # Simple denoising step (Langevin dynamics)
            alpha_t = self.get_alpha_t(i)
            beta_t = self.get_beta_t(i)

            x = (x - beta_t * noise_pred) / torch.sqrt(alpha_t) + torch.sqrt(beta_t) * torch.randn_like(x)

        return x

    def get_alpha_t(self, t):
        """Get alpha_t for diffusion process"""
        # Linear schedule
        s = 0.008
        T = self.num_diffusion_steps
        t_tensor = torch.tensor(t, dtype=torch.float32) / T
        alpha_bar = torch.cos((t_tensor + s) / (1 + s) * torch.pi / 2) ** 2
        return alpha_bar

    def get_beta_t(self, t):
        """Get beta_t for diffusion process"""
        alpha_t = self.get_alpha_t(t)
        alpha_tm1 = self.get_alpha_t(max(0, t-1))
        return torch.clamp(1 - alpha_t / alpha_tm1, 0, 1)

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DenoisingUNet(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, channel_mult, num_heads):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Time embedding projection
        self.time_proj = nn.Linear(model_channels, model_channels)

        # Middle layers
        self.middle_block = nn.Sequential(
            ResidualBlock(model_channels, model_channels),
            AttentionBlock(model_channels, num_heads),
            ResidualBlock(model_channels, model_channels),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, out_channels)
        )

    def forward(self, x, time_emb):
        """
        Args:
            x: [B, action_dim] noisy action
            time_emb: [B, model_channels] time embedding
        """
        # Project input
        h = self.input_proj(x)  # [B, model_channels]

        # Add time embedding
        time_h = self.time_proj(F.silu(time_emb))  # [B, model_channels]
        h = h + time_h

        # Apply middle blocks
        h = self.middle_block(h)

        # Output projection
        out = self.output_proj(h)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Linear(channels, channels)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv_proj = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv_proj(self.norm(x)).reshape(B, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, L, C)
        return x + self.proj(attn)
```

### 2. Memory-Augmented VLA Models

Models with external memory for long-horizon tasks:

```python
# Memory-augmented VLA for long-horizon tasks
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAugmentedVLA(nn.Module):
    def __init__(self, vision_dim=512, language_dim=768, action_dim=14,
                 memory_size=100, memory_dim=256):
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Vision and language encoders
        self.vision_encoder = nn.Linear(vision_dim, memory_dim)
        self.language_encoder = nn.Linear(language_dim, memory_dim)

        # Memory module
        self.memory = ExternalMemory(memory_size, memory_dim)

        # Action generation
        self.action_generator = ActionGenerator(
            memory_dim=memory_dim,
            action_dim=action_dim
        )

        # Memory update
        self.memory_update = MemoryUpdateNetwork(memory_dim)

    def forward(self, images, commands, prev_memory_state=None):
        """
        Forward pass with memory
        Args:
            images: [B, C, H, W] current visual observation
            commands: [B, seq_len, language_dim] language command
            prev_memory_state: [B, memory_size, memory_dim] previous memory state
        Returns:
            action [B, action_dim] and updated memory state
        """
        B = images.shape[0]

        # Encode current observation
        vision_features = self.vision_encoder(images.view(B, -1))  # [B, memory_dim]
        language_features = torch.mean(self.language_encoder(commands), dim=1)  # [B, memory_dim]

        # Initialize memory if none provided
        if prev_memory_state is None:
            memory_state = self.memory.init_memory(B)  # [B, memory_size, memory_dim]
        else:
            memory_state = prev_memory_state

        # Retrieve relevant memories
        retrieved_memories = self.memory.retrieve(
            query=vision_features + language_features,  # Combined query
            memory_state=memory_state
        )  # [B, num_retrieved, memory_dim]

        # Generate action based on current observation and retrieved memories
        action = self.action_generator(
            current_features=vision_features,
            language_features=language_features,
            retrieved_memories=retrieved_memories
        )  # [B, action_dim]

        # Update memory with current observation
        updated_memory = self.memory_update(
            memory_state=memory_state,
            new_content=vision_features + language_features,
            retrieved_memories=retrieved_memories
        )  # [B, memory_size, memory_dim]

        return action, updated_memory

class ExternalMemory(nn.Module):
    def __init__(self, memory_size, memory_dim):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # Memory initialization
        self.register_buffer('memory_init', torch.randn(memory_size, memory_dim) * 0.1)

    def init_memory(self, batch_size):
        """Initialize memory for a batch"""
        return self.memory_init.unsqueeze(0).expand(batch_size, -1, -1)

    def retrieve(self, query, memory_state, k=10):
        """Retrieve top-k relevant memories"""
        # query: [B, memory_dim]
        # memory_state: [B, memory_size, memory_dim]

        # Compute similarities
        similarities = torch.bmm(
            query.unsqueeze(1),  # [B, 1, memory_dim]
            memory_state.transpose(1, 2)  # [B, memory_dim, memory_size]
        ).squeeze(1)  # [B, memory_size]

        # Get top-k indices
        _, top_k_indices = torch.topk(similarities, k=min(k, self.memory_size), dim=1)  # [B, k]

        # Gather top-k memories
        B, _ = top_k_indices.shape
        batch_indices = torch.arange(B, device=top_k_indices.device).unsqueeze(1).expand(-1, k)
        retrieved = memory_state[batch_indices, top_k_indices]  # [B, k, memory_dim]

        return retrieved

    def write(self, memory_state, content, write_weights):
        """Write content to memory with attention weights"""
        # content: [B, memory_dim]
        # write_weights: [B, memory_size]
        content_expanded = content.unsqueeze(1).expand(-1, self.memory_size, -1)  # [B, memory_size, memory_dim]
        write_weights_expanded = write_weights.unsqueeze(-1)  # [B, memory_size, 1]

        # Update memory
        updated_memory = memory_state * (1 - write_weights_expanded) + content_expanded * write_weights_expanded
        return updated_memory

class ActionGenerator(nn.Module):
    def __init__(self, memory_dim, action_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=memory_dim, num_heads=8, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),  # *3 for current + lang + retrieved
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, action_dim)
        )

    def forward(self, current_features, language_features, retrieved_memories):
        """
        Generate action from current observation and retrieved memories
        Args:
            current_features: [B, memory_dim]
            language_features: [B, memory_dim]
            retrieved_memories: [B, num_retrieved, memory_dim]
        Returns:
            action: [B, action_dim]
        """
        B, num_retrieved, mem_dim = retrieved_memories.shape

        # Combine all features
        current_expanded = current_features.unsqueeze(1)  # [B, 1, memory_dim]
        language_expanded = language_features.unsqueeze(1)  # [B, 1, memory_dim]

        # Concatenate all features
        all_features = torch.cat([
            current_expanded,
            language_expanded,
            retrieved_memories
        ], dim=1)  # [B, 2 + num_retrieved, memory_dim]

        # Apply attention across all features
        attended_features, _ = self.attention(
            query=current_expanded.transpose(0, 1),  # [1, B, memory_dim]
            key=all_features.transpose(0, 1),       # [2 + num_retrieved, B, memory_dim]
            value=all_features.transpose(0, 1)      # [2 + num_retrieved, B, memory_dim]
        )
        attended_features = attended_features.transpose(0, 1).squeeze(1)  # [B, memory_dim]

        # Generate action
        combined_input = torch.cat([
            current_features,
            language_features,
            attended_features
        ], dim=-1)  # [B, memory_dim * 3]

        action = self.action_head(combined_input)  # [B, action_dim]

        return action

class MemoryUpdateNetwork(nn.Module):
    def __init__(self, memory_dim):
        super().__init__()
        self.write_head = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),  # For content weighting
            nn.Sigmoid()  # Output write weights
        )
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid()
        )

    def forward(self, memory_state, new_content, retrieved_memories):
        """
        Update memory with new content
        Args:
            memory_state: [B, memory_size, memory_dim]
            new_content: [B, memory_dim]
            retrieved_memories: [B, num_retrieved, memory_dim]
        Returns:
            updated_memory: [B, memory_size, memory_dim]
        """
        B, memory_size, memory_dim = memory_state.shape

        # Compute write weights based on similarity to new content
        content_similarity = torch.bmm(
            new_content.unsqueeze(1),  # [B, 1, memory_dim]
            retrieved_memories.transpose(1, 2)  # [B, memory_dim, num_retrieved]
        ).squeeze(1)  # [B, num_retrieved]

        # Normalize similarities to get write weights
        write_weights = F.softmax(content_similarity, dim=1)  # [B, num_retrieved]

        # Pad write weights to match memory size
        if write_weights.shape[1] < memory_size:
            padding = torch.zeros(B, memory_size - write_weights.shape[1], device=write_weights.device)
            write_weights = torch.cat([write_weights, padding], dim=1)  # [B, memory_size]

        # Update memory
        new_content_expanded = new_content.unsqueeze(1).expand(-1, memory_size, -1)
        updated_memory = memory_state * (1 - write_weights.unsqueeze(-1)) + \
                        new_content_expanded * write_weights.unsqueeze(-1)

        return updated_memory
```

## Training VLA Models

### 1. Data Preprocessing Pipeline

```python
# VLA data preprocessing pipeline
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

class VLADataset(Dataset):
    def __init__(self, data_path, transform=None, max_seq_len=512):
        """
        VLA dataset for vision-language-action learning
        Args:
            data_path: Path to dataset (JSON format)
            transform: Image transformations
            max_seq_len: Maximum sequence length for language
        """
        self.data_path = data_path
        self.transform = transform or self.get_default_transform()
        self.max_seq_len = max_seq_len

        # Load dataset
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load and preprocess image
        image = self.load_image(sample['image_path'])
        image = self.transform(image)

        # Tokenize command
        command_text = sample['command']
        command_tokens = self.tokenizer(
            command_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Load action
        action = torch.tensor(sample['action'], dtype=torch.float32)

        # Load optional task ID
        task_id = torch.tensor(sample.get('task_id', 0), dtype=torch.long)

        return {
            'images': image,
            'commands': command_tokens['input_ids'].squeeze(0),
            'command_attention_mask': command_tokens['attention_mask'].squeeze(0),
            'actions': action,
            'task_ids': task_id,
            'episode_id': sample.get('episode_id', idx)
        }

    def load_image(self, image_path):
        """Load image from path"""
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        return image

    def get_default_transform(self):
        """Get default image transformations"""
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_vla_dataloader(dataset_path, batch_size=32, shuffle=True, num_workers=4):
    """Create dataloader for VLA training"""
    dataset = VLADataset(dataset_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vla_collate_fn
    )

def vla_collate_fn(batch):
    """Collate function for VLA batches"""
    images = torch.stack([item['images'] for item in batch])
    commands = torch.stack([item['commands'] for item in batch])
    command_attention_masks = torch.stack([item['command_attention_mask'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    task_ids = torch.stack([item['task_ids'] for item in batch])

    return {
        'images': images,
        'commands': commands,
        'command_attention_mask': command_attention_masks,
        'actions': actions,
        'task_ids': task_ids
    }
```

### 2. Training Loop with Curriculum Learning

```python
# Advanced training loop with curriculum learning
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

class VLA_Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Mixed precision training
        self.scaler = GradScaler()

        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.difficulty_thresholds = [0.1, 0.3, 0.5, 0.7]  # Performance thresholds

        # Performance tracking
        self.performance_history = []
        self.best_val_loss = float('inf')

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

    def train_step(self, batch):
        """Single training step with mixed precision"""
        images = batch['images'].cuda()
        commands = batch['commands'].cuda()
        actions = batch['actions'].cuda()
        task_ids = batch['task_ids'].cuda()

        self.optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            pred_actions = self.model(images, commands, task_ids)
            loss = self.criterion(pred_actions, actions)

        # Mixed precision backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update learning rate
        self.scheduler.step()

        return loss.item()

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].cuda()
                commands = batch['commands'].cuda()
                actions = batch['actions'].cuda()

                with autocast():
                    pred_actions = self.model(images, commands)
                    loss = self.criterion(pred_actions, actions)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Update performance history
        self.performance_history.append(avg_loss)

        # Check for best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_best_model()

        return avg_loss

    def curriculum_learning_step(self):
        """Adjust training difficulty based on performance"""
        if len(self.performance_history) < 10:
            return  # Need more data

        # Calculate recent performance
        recent_performance = sum(self.performance_history[-10:]) / 10

        # Adjust curriculum stage based on performance
        for i, threshold in enumerate(self.difficulty_thresholds):
            if recent_performance <= threshold:
                self.curriculum_stage = i
                break

        print(f"Curriculum stage updated to: {self.curriculum_stage}")

    def save_best_model(self, path='best_vla_model.pth'):
        """Save the best performing model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'curriculum_stage': self.curriculum_stage
        }, path)

    def load_model(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.curriculum_stage = checkpoint['curriculum_stage']

    def train(self, num_epochs=100):
        """Main training loop"""
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()

            # Validation
            val_loss = self.validate()

            # Curriculum learning adjustment
            self.curriculum_learning_step()

            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Curriculum Stage: {self.curriculum_stage}')
            print()

            # Early stopping check
            if len(self.performance_history) > 20:
                if all(loss > self.best_val_loss for loss in self.performance_history[-10:]):
                    print("Early stopping triggered")
                    break

# Example usage
def train_vla_model():
    """Example training function"""
    # Initialize model
    model = OpenVLA(
        vision_model=None,  # Will be initialized inside
        language_model=None,  # Will be initialized inside
        action_dim=14
    ).cuda()

    # Create data loaders
    train_loader = create_vla_dataloader('train_data.json', batch_size=16)
    val_loader = create_vla_dataloader('val_data.json', batch_size=16, shuffle=False)

    # Initialize trainer
    trainer = VLA_Trainer(model, train_loader, val_loader, learning_rate=1e-4)

    # Train model
    trainer.train(num_epochs=50)

    print("Training completed!")
```

## Evaluation and Deployment

### 1. VLA Model Evaluation

```python
# Comprehensive VLA model evaluation
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

class VLA_Evaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_commands = []
        all_episodes = []

        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['images'].cuda()
                commands = batch['commands'].cuda()
                actions = batch['actions'].cuda()

                pred_actions = self.model(images, commands)

                all_predictions.extend(pred_actions.cpu().numpy())
                all_targets.extend(actions.cpu().numpy())
                all_commands.extend(batch['commands'])
                all_episodes.extend(batch.get('episode_id', []))

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)

        # Task-specific evaluation
        task_metrics = self.evaluate_by_task(all_predictions, all_targets)

        # Sequence evaluation for temporal consistency
        sequence_metrics = self.evaluate_sequences(all_predictions, all_targets)

        return {
            'overall_metrics': metrics,
            'task_metrics': task_metrics,
            'sequence_metrics': sequence_metrics,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def calculate_metrics(self, predictions, targets):
        """Calculate standard regression metrics"""
        # Mean Squared Error
        mse = np.mean((predictions - targets) ** 2)

        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - targets))

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Mean Absolute Percentage Error (avoid division by zero)
        non_zero_targets = targets[targets != 0]
        non_zero_predictions = predictions[targets != 0]
        mape = np.mean(np.abs((non_zero_targets - non_zero_predictions) / non_zero_targets)) * 100

        # Explained Variance
        total_var = np.var(targets)
        unexplained_var = np.var(targets - predictions)
        explained_var = 1 - (unexplained_var / total_var)

        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'explained_variance': explained_var,
            'r_squared': r_squared
        }

    def evaluate_by_task(self, predictions, targets):
        """Evaluate model performance on different tasks"""
        # This assumes you have task labels in your dataset
        # For demonstration, we'll create mock task evaluation
        unique_tasks = range(5)  # Mock task IDs

        task_metrics = {}
        for task_id in unique_tasks:
            # In a real implementation, you'd filter by actual task
            task_mask = np.random.rand(len(predictions)) > 0.8  # Mock mask
            if np.sum(task_mask) > 0:
                task_preds = predictions[task_mask]
                task_targets = targets[task_mask]

                task_metrics[f'task_{task_id}'] = self.calculate_metrics(task_preds, task_targets)

        return task_metrics

    def evaluate_sequences(self, predictions, targets):
        """Evaluate temporal consistency in action sequences"""
        # Calculate smoothness metrics
        pred_velocities = np.diff(predictions, axis=0)
        target_velocities = np.diff(targets, axis=0)

        pred_accelerations = np.diff(pred_velocities, axis=0)
        target_accelerations = np.diff(target_velocities, axis=0)

        # Smoothness metrics
        pred_smoothness = np.mean(np.abs(pred_velocities))
        target_smoothness = np.mean(np.abs(target_velocities))

        pred_jerk = np.mean(np.abs(pred_accelerations))
        target_jerk = np.mean(np.abs(target_accelerations))

        return {
            'prediction_smoothness': pred_smoothness,
            'target_smoothness': target_smoothness,
            'prediction_jerk': pred_jerk,
            'target_jerk': target_jerk,
            'smoothness_ratio': pred_smoothness / (target_smoothness + 1e-8)
        }

    def plot_evaluation_results(self, results):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Prediction vs Target scatter
        preds = results['predictions'].flatten()
        targets = results['targets'].flatten()

        axes[0, 0].scatter(targets, preds, alpha=0.5)
        axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Target Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Prediction vs Target')

        # Error distribution
        errors = preds - targets
        axes[0, 1].hist(errors, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')

        # Time series comparison (first dimension)
        time_steps = min(len(preds), 100)  # Limit for clarity
        axes[1, 0].plot(range(time_steps), preds[:time_steps], label='Predicted', alpha=0.7)
        axes[1, 0].plot(range(time_steps), targets[:time_steps], label='Target', alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Time Series Comparison')
        axes[1, 0].legend()

        # Action dimension analysis
        action_errors = np.mean(np.abs(results['predictions'] - results['targets']), axis=0)
        axes[1, 1].bar(range(len(action_errors)), action_errors)
        axes[1, 1].set_xlabel('Action Dimension')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Error by Action Dimension')

        plt.tight_layout()
        plt.show()

    def compute_generalization_metrics(self, train_metrics, test_metrics):
        """Compute metrics that indicate generalization ability"""
        # Overfitting ratio
        train_mse = train_metrics.get('mse', 1.0)
        test_mse = test_metrics.get('mse', 1.0)
        overfitting_ratio = test_mse / (train_mse + 1e-8)

        # Stability metrics
        prediction_std = np.std(results['predictions'])
        target_std = np.std(results['targets'])
        stability_ratio = prediction_std / (target_std + 1e-8)

        return {
            'overfitting_ratio': overfitting_ratio,
            'stability_ratio': stability_ratio,
            'generalization_score': 1.0 / (overfitting_ratio + 1e-8)
        }
```

## Next Steps

In the next section, we'll explore human-robot interaction through VLA systems, learning how to design interfaces and interaction paradigms that leverage the power of vision-language-action understanding for natural and intuitive collaboration between humans and humanoid robots.