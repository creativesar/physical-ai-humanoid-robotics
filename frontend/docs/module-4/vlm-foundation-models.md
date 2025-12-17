---
sidebar_position: 7
title: "Vision-Language Models Foundation"
---

# Vision-Language Models Foundation

## Introduction to Vision-Language Models

Vision-Language Models (VLMs) form the foundation of modern Vision-Language-Action (VLA) systems, enabling humanoid robots to understand and reason about visual scenes through natural language. These models have revolutionized robotics by allowing robots to interpret complex visual information and respond to human commands in natural language, creating more intuitive and flexible human-robot interaction.

## Historical Development of Vision-Language Models

### 1. Early Approaches (2010-2015)

The earliest vision-language models focused on simple image captioning and visual question answering:

```python
# Early vision-language model (2014-2015 era)
import torch
import torch.nn as nn
import torchvision.models as models

class EarlyVisionLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()

        # Early vision encoder (CNN-based)
        self.vision_encoder = models.resnet18(pretrained=True)
        self.vision_encoder.fc = nn.Linear(self.vision_encoder.fc.in_features, embed_dim)

        # Early language decoder (LSTM-based)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.language_head = nn.Linear(hidden_dim, vocab_size)

        # Simple attention mechanism
        self.attention = nn.Linear(embed_dim + hidden_dim, 1)

    def forward(self, images, captions=None):
        # Encode visual features
        vision_features = self.vision_encoder(images)  # [B, embed_dim]

        # If training, use teacher forcing
        if captions is not None:
            # Embed captions
            caption_embeds = self.embedding(captions)  # [B, seq_len, embed_dim]

            # Process through LSTM with attention to visual features
            lstm_output, _ = self.lstm(caption_embeds)  # [B, seq_len, hidden_dim]

            # Apply attention-weighted combination
            attended_output = []
            for i in range(lstm_output.size(1)):  # For each time step
                att_weights = torch.softmax(
                    self.attention(
                        torch.cat([vision_features.unsqueeze(1).repeat(1, lstm_output.size(1), 1),
                                 lstm_output], dim=-1)
                    ), dim=1
                )
                attended = torch.sum(att_weights * lstm_output, dim=1)
                attended_output.append(attended)

            attended_output = torch.stack(attended_output, dim=1)  # [B, seq_len, hidden_dim]
            logits = self.language_head(attended_output)  # [B, seq_len, vocab_size]

            return logits
        else:
            # For inference, generate tokens sequentially
            batch_size = images.size(0)
            generated = torch.zeros(batch_size, 1, dtype=torch.long, device=images.device)
            hidden = None

            for i in range(20):  # Max generation length
                caption_embeds = self.embedding(generated[:, -1:])  # [B, 1, embed_dim]
                lstm_out, hidden = self.lstm(caption_embeds, hidden)  # [B, 1, hidden_dim]

                # Apply attention
                att_weights = torch.softmax(
                    self.attention(
                        torch.cat([vision_features.unsqueeze(1), lstm_out], dim=-1)
                    ), dim=-1
                )
                attended = torch.sum(att_weights * lstm_out, dim=1)
                logits = self.language_head(attended)  # [B, vocab_size]

                # Sample next token
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                generated = torch.cat([generated, next_token], dim=1)

            return generated
```

### 2. Attention-Based Models (2015-2018)

The introduction of attention mechanisms revolutionized vision-language models:

```python
# Attention-based vision-language model (2016-2018 era)
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBasedVLM(nn.Module):
    def __init__(self, vocab_size, vision_dim=2048, embed_dim=512, num_heads=8):
        super().__init__()

        self.vision_dim = vision_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Language encoder
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.create_positional_encoding(50, embed_dim)

        # Multi-head attention for vision-language fusion
        self.vision_language_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Language decoder
        self.language_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=6
        )

        # Output head
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def create_positional_encoding(self, max_len, embed_dim):
        """Create positional encoding for sequences"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, embed_dim]

    def forward(self, images, captions=None, attention_mask=None):
        batch_size = images.size(0)

        # Encode visual features
        vision_features = self.vision_encoder(images)  # [B, vision_dim] -> [B, embed_dim]
        vision_features = vision_features.unsqueeze(1)  # [B, 1, embed_dim]

        if captions is not None:
            # Encode language features
            word_embeds = self.word_embedding(captions)  # [B, seq_len, embed_dim]
            seq_len = word_embeds.size(1)

            # Add positional encoding
            pos_encoding = self.pos_encoding[:, :seq_len, :].to(images.device)
            word_embeds = word_embeds + pos_encoding

            # Apply vision-language attention
            vision_expanded = vision_features.expand(-1, seq_len, -1)  # [B, seq_len, embed_dim]

            # Self-attention with vision context
            attended_features, attention_weights = self.vision_language_attention(
                query=word_embeds,
                key=vision_expanded,
                value=vision_expanded
            )

            # Decode language
            decoded = self.language_decoder(
                tgt=attended_features,
                memory=word_embeds,
                tgt_mask=self.generate_square_subsequent_mask(seq_len).to(images.device)
            )

            # Project to vocabulary
            logits = self.output_projection(decoded)  # [B, seq_len, vocab_size]

            return logits, attention_weights
        else:
            # Inference: generate captions
            generated = torch.zeros(batch_size, 1, dtype=torch.long, device=images.device)
            vision_features_expanded = vision_features.expand(batch_size, 1, self.embed_dim)

            for i in range(20):  # Max generation length
                word_embeds = self.word_embedding(generated[:, -1:])
                pos_encoding = self.pos_encoding[:, i:i+1, :].to(images.device)
                word_embeds = word_embeds + pos_encoding

                attended, _ = self.vision_language_attention(
                    query=word_embeds,
                    key=vision_features_expanded,
                    value=vision_features_expanded
                )

                decoded = self.language_decoder(
                    tgt=attended,
                    memory=word_embeds
                )

                logits = self.output_projection(decoded[:, -1, :])  # [B, vocab_size]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if end token is generated
                if (next_token == 1).all():  # Assuming 1 is end token
                    break

            return generated

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
```

### 3. Transformer Era Models (2018-Present)

The transformer architecture transformed vision-language modeling:

```python
# Modern transformer-based VLM (CLIP-style architecture)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class CLIPStyleVLM(nn.Module):
    def __init__(self, vision_model_name='openai/clip-vit-base-patch32',
                 text_model_name='bert-base-uncased'):
        super().__init__()

        # Load pre-trained vision model
        from transformers import CLIPVisionModel, CLIPTextModel
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_model = CLIPTextModel.from_pretrained(text_model_name)

        # Projection heads for contrastive learning
        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size
        proj_dim = 512

        self.vision_projection = nn.Linear(vision_dim, proj_dim)
        self.text_projection = nn.Linear(text_dim, proj_dim)

        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def encode_images(self, images):
        """Encode images to visual features"""
        vision_outputs = self.vision_model(pixel_values=images)
        image_features = vision_outputs.pooler_output  # [B, vision_dim]
        image_features = F.normalize(self.vision_projection(image_features), dim=-1)
        return image_features

    def encode_texts(self, texts):
        """Encode texts to textual features"""
        text_outputs = self.text_model(input_ids=texts['input_ids'],
                                     attention_mask=texts['attention_mask'])
        text_features = text_outputs.pooler_output  # [B, text_dim]
        text_features = F.normalize(self.text_projection(text_features), dim=-1)
        return text_features

    def forward(self, images, texts):
        """Forward pass for contrastive learning"""
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)

        # Compute similarity matrix
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        return {
            'image_features': image_features,
            'text_features': text_features,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }

    def compute_contrastive_loss(self, logits_per_image, logits_per_text):
        """Compute contrastive loss for image-text pairs"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)

        # Image-to-text loss
        image_loss = F.cross_entropy(logits_per_image, labels)
        # Text-to-image loss
        text_loss = F.cross_entropy(logits_per_text, labels)

        return (image_loss + text_loss) / 2.0

class FlamingoStyleVLM(nn.Module):
    """Flamingo-style vision-language model with few-shot capabilities"""
    def __init__(self, vision_encoder, text_decoder, cross_attention_dim=768):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.cross_attention_dim = cross_attention_dim

        # Cross-attention layers for vision-language fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=12,
            batch_first=True
        )

        # Gated cross-attention for selective fusion
        self.gated_cross_attention = GatedCrossAttentionBlock(
            dim=cross_attention_dim,
            heads=12
        )

        # Perceiver resampler for fixed-size vision representations
        self.perceiver_resampler = PerceiverResampler(
            dim=cross_attention_dim,
            depth=6,
            dim_head=64,
            heads=12,
            num_latents=64  # Fixed number of latents
        )

    def forward(self, images, texts, instruction=None):
        """
        Forward pass with interleaved vision and language processing
        Args:
            images: [B, C, H, W] or list of images
            texts: [B, seq_len] tokenized text
            instruction: Optional instruction for few-shot prompting
        """
        batch_size = images.size(0)

        # Encode visual features
        vision_features = self.vision_encoder(images)  # [B, num_patches, vision_dim]

        # Apply perceiver resampler to get fixed-size representation
        resampled_vision = self.perceiver_resampler(vision_features)  # [B, num_latents, dim]

        # Encode text
        text_outputs = self.text_decoder(
            input_ids=texts['input_ids'],
            attention_mask=texts['attention_mask'],
            output_hidden_states=True
        )
        text_features = text_outputs.hidden_states[-1]  # [B, seq_len, text_dim]

        # Apply gated cross-attention to integrate vision and language
        attended_text = self.gated_cross_attention(
            text_features,  # Query
            resampled_vision,  # Key-value from vision
            attention_mask=texts['attention_mask']
        )

        # Generate response
        lm_logits = self.text_decoder.lm_head(attended_text)

        return {
            'logits': lm_logits,
            'vision_features': resampled_vision,
            'attended_text': attended_text,
            'cross_attention_weights': self.gated_cross_attention.last_attention_weights
        }

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_features, vision_features, attention_mask=None):
        # Cross-attention: text attends to vision
        attended_vision, attention_weights = self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
            key_padding_mask=None if attention_mask is None else ~attention_mask.bool()
        )

        # Gate the attention
        gated_attended = self.gate * attended_vision

        # Residual connection and normalization
        output = self.norm(text_features + gated_attended)

        # Store attention weights for analysis
        self.last_attention_weights = attention_weights

        return output

class PerceiverResampler(nn.Module):
    """Perceiver-style resampler to convert variable vision tokens to fixed size"""
    def __init__(self, dim, depth, dim_head, heads, num_latents):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, num_patches, dim] vision features
        Returns:
            [B, num_latents, dim] resampled features
        """
        B, N, _ = x.shape
        latents = self.latents.expand(B, -1, -1)  # [B, num_latents, dim]

        for self_attn, feedforward in self.layers:
            # Self-attention: latents attend to themselves
            latents_attn, _ = self_attn(
                query=latents,
                key=latents,
                value=latents
            )

            # Cross-attention: latents attend to vision features
            latents_cross_attn, _ = self_attn(
                query=latents_attn,
                key=x,
                value=x
            )

            # Feedforward
            latents = latents + feedforward(latents_cross_attn)

        return self.norm(latents)  # [B, num_latents, dim]
```

## Modern Foundation Models

### 1. OpenFlamingo

OpenFlamingo represents a significant advancement in vision-language modeling:

```python
# OpenFlamingo-inspired architecture
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class OpenFlamingo(nn.Module):
    def __init__(self, vision_encoder, text_decoder, cross_attention_dim=1024):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.cross_attention_dim = cross_attention_dim

        # Vision-language cross attention
        self.vision_language_attention = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )

        # Gated cross attention for conditional fusion
        self.gated_cross_attention = GatedCrossAttention(
            dim=cross_attention_dim,
            num_heads=16
        )

        # Perceiver resampler for vision features
        self.perceiver_resampler = PerceiverResampler(
            dim=cross_attention_dim,
            depth=6,
            heads=16,
            num_latents=64
        )

        # Projection layers
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, cross_attention_dim)
        self.text_proj = nn.Linear(self.text_decoder.config.hidden_size, cross_attention_dim)

    def encode_vision(self, images):
        """Encode images using vision encoder"""
        # images: [B, C, H, W]
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # [B, num_patches, vision_hidden_size]

        # Project to common dimension
        projected_vision = self.vision_proj(vision_features)  # [B, num_patches, cross_attention_dim]

        # Apply perceiver resampler to get fixed-size representation
        resampled_vision = self.perceiver_resampler(projected_vision)  # [B, num_latents, cross_attention_dim]

        return resampled_vision

    def encode_text(self, input_ids, attention_mask):
        """Encode text using text decoder"""
        # input_ids: [B, seq_len]
        text_outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_features = text_outputs.hidden_states[-1]  # [B, seq_len, text_hidden_size]

        # Project to common dimension
        projected_text = self.text_proj(text_features)  # [B, seq_len, cross_attention_dim]

        return projected_text

    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass for vision-language understanding
        Args:
            images: [B, C, H, W] images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask
            labels: [B, seq_len] labels for training (optional)
        """
        # Encode vision and text
        vision_features = self.encode_vision(images)  # [B, num_latents, cross_attention_dim]
        text_features = self.encode_text(input_ids, attention_mask)  # [B, seq_len, cross_attention_dim]

        # Apply gated cross attention: text attends to vision
        attended_features = self.gated_cross_attention(
            query=text_features,  # [B, seq_len, cross_attention_dim]
            key=vision_features,  # [B, num_latents, cross_attention_dim]
            value=vision_features   # [B, num_latents, cross_attention_dim]
        )  # [B, seq_len, cross_attention_dim]

        # Combine attended vision features with original text features
        combined_features = text_features + attended_features

        # Generate text using the decoder
        outputs = self.text_decoder(
            inputs_embeds=combined_features,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    def generate(self, images, input_ids, attention_mask, max_new_tokens=50):
        """Generate text conditioned on images"""
        with torch.no_grad():
            # Encode vision features once
            vision_features = self.encode_vision(images)  # [B, num_latents, cross_attention_dim]

            # Start with input tokens
            generated = input_ids.clone()  # [B, input_seq_len]

            for _ in range(max_new_tokens):
                current_seq_len = generated.size(1)

                # Encode current text
                text_features = self.encode_text(generated, torch.ones_like(generated))  # [B, current_seq_len, cross_attention_dim]

                # Apply cross attention
                attended_features = self.gated_cross_attention(
                    query=text_features,
                    key=vision_features,
                    value=vision_features
                )

                # Combine features
                combined_features = text_features + attended_features

                # Get logits for the last token
                last_hidden_states = combined_features[:, -1:, :]  # [B, 1, cross_attention_dim]

                # Pass through text decoder's LM head
                lm_logits = self.text_decoder.lm_head(last_hidden_states)  # [B, 1, vocab_size]

                # Sample next token
                next_token_logits = lm_logits[:, -1, :]  # [B, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if end token is generated
                if (next_token == self.text_decoder.config.eos_token_id).any():
                    break

            return generated

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value, attention_mask=None):
        # Apply cross attention: query attends to key/value
        attended, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )

        # Apply gate
        gated_attended = self.gate * attended

        # Residual connection and normalization
        output = self.norm(query + gated_attended)

        return output

    def get_attention_weights(self):
        """Return attention weights for analysis"""
        return self.attention_weights
```

### 2. BLIP-2 Architecture

BLIP-2 introduced a novel two-stage approach:

```python
# BLIP-2 inspired architecture
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BLIP2(nn.Module):
    def __init__(self, vision_encoder, qformer, text_decoder):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.qformer = qformer  # Querying transformer
        self.text_decoder = text_decoder

        # Vision-to-Qformer projection
        self.vision_proj = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.qformer.config.hidden_size
        )

        # Qformer-to-text projection
        self.qformer_proj = nn.Linear(
            self.qformer.config.hidden_size,
            self.text_decoder.config.hidden_size
        )

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, 32, self.qformer.config.hidden_size)  # 32 query tokens
        )

    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass through BLIP-2
        Args:
            images: [B, C, H, W] images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask
            labels: [B, seq_len] labels for training
        """
        batch_size = images.size(0)

        # Encode visual features
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # [B, num_patches, vision_hidden_size]

        # Project vision features to Qformer dimension
        projected_vision = self.vision_proj(vision_features)  # [B, num_patches, qformer_hidden_size]

        # Repeat query tokens for batch
        query_tokens = self.query_tokens.repeat(batch_size, 1, 1)  # [B, 32, qformer_hidden_size]

        # Qformer: vision features attend to query tokens
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=projected_vision,
            return_dict=True
        )
        qformer_features = qformer_outputs.last_hidden_state  # [B, 32, qformer_hidden_size]

        # Project Qformer features to text decoder dimension
        text_input_features = self.qformer_proj(qformer_features)  # [B, 32, text_hidden_size]

        # Prepare text decoder inputs
        text_outputs = self.text_decoder(
            inputs_embeds=text_input_features,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return text_outputs

    def generate(self, images, input_ids, max_new_tokens=50):
        """Generate text conditioned on images"""
        with torch.no_grad():
            batch_size = images.size(0)

            # Encode vision features
            vision_outputs = self.vision_encoder(pixel_values=images)
            vision_features = vision_outputs.last_hidden_state
            projected_vision = self.vision_proj(vision_features)

            # Get query-enhanced vision features
            query_tokens = self.query_tokens.repeat(batch_size, 1, 1)
            qformer_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=projected_vision,
                return_dict=True
            )
            qformer_features = qformer_outputs.last_hidden_state
            text_input_features = self.qformer_proj(qformer_features)

            # Generate using text decoder
            generated = self.text_decoder.generate(
                inputs_embeds=text_input_features,
                decoder_input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.text_decoder.config.pad_token_id,
                eos_token_id=self.text_decoder.config.eos_token_id
            )

            return generated

class QFormer(nn.Module):
    """Querying transformer that bridges vision and language"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)

        # Cross attention for vision-query interaction
        self.vision_query_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, query_embeds, encoder_hidden_states, attention_mask=None):
        """
        Args:
            query_embeds: [B, num_queries, hidden_size] learnable query tokens
            encoder_hidden_states: [B, seq_len, hidden_size] vision features
            attention_mask: [B, seq_len] attention mask for vision features
        """
        B, num_queries, hidden_size = query_embeds.shape

        # Cross attention: query tokens attend to vision features
        query_attended, attention_weights = self.vision_query_attention(
            query=query_embeds,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )

        # Self attention on query tokens
        query_output = self.encoder(query_attended)

        # Residual connection and normalization
        output = self.layer_norm(query_embeds + query_output)

        return {
            'last_hidden_state': output,
            'attention_weights': attention_weights,
            'hidden_states': None  # Could return intermediate states if needed
        }
```

## Vision-Language Alignment Techniques

### 1. Contrastive Learning

```python
# Contrastive learning for vision-language alignment
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveVLM(nn.Module):
    def __init__(self, vision_encoder, text_encoder, proj_dim=512):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # Projection heads
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)

        # Temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def encode_vision(self, images):
        """Encode images to visual representations"""
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.pooler_output  # [B, vision_hidden_size]
        vision_proj = self.vision_projection(vision_features)  # [B, proj_dim]
        vision_proj = F.normalize(vision_proj, dim=-1)  # Normalize for contrastive learning
        return vision_proj

    def encode_text(self, input_ids, attention_mask):
        """Encode text to textual representations"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.pooler_output  # [B, text_hidden_size]
        text_proj = self.text_projection(text_features)  # [B, proj_dim]
        text_proj = F.normalize(text_proj, dim=-1)  # Normalize for contrastive learning
        return text_proj

    def forward(self, images, input_ids, attention_mask):
        """Forward pass for contrastive learning"""
        # Encode vision and text
        vision_features = self.encode_vision(images)  # [B, proj_dim]
        text_features = self.encode_text(input_ids, attention_mask)  # [B, proj_dim]

        # Compute similarity matrix
        # Similarity between all image-text pairs
        logits_v2t = torch.matmul(vision_features, text_features.t()) / self.temperature  # [B, B]
        logits_t2v = logits_v2t.t()  # [B, B]

        # Create ground truth: diagonal elements are positive pairs
        ground_truth = torch.arange(len(vision_features), device=images.device)

        # Compute contrastive loss
        loss_v2t = F.cross_entropy(logits_v2t, ground_truth)
        loss_t2v = F.cross_entropy(logits_t2v, ground_truth)

        contrastive_loss = (loss_v2t + loss_t2v) / 2.0

        # Compute accuracy
        with torch.no_grad():
            acc_v2t = (torch.argmax(logits_v2t, dim=1) == ground_truth).float().mean()
            acc_t2v = (torch.argmax(logits_t2v, dim=1) == ground_truth).float().mean()
            accuracy = (acc_v2t + acc_t2v) / 2.0

        return {
            'loss': contrastive_loss,
            'accuracy': accuracy,
            'logits_v2t': logits_v2t,
            'logits_t2v': logits_t2v,
            'vision_features': vision_features,
            'text_features': text_features
        }

class HardNegativeMining(nn.Module):
    """Hard negative mining for improved contrastive learning"""
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, vision_features, text_features, labels):
        """
        Compute contrastive loss with hard negative mining
        Args:
            vision_features: [B, dim] visual features
            text_features: [B, dim] textual features
            labels: [B] ground truth labels
        """
        # Compute similarity matrix
        similarity_matrix = torch.matmul(vision_features, text_features.t())  # [B, B]

        # Create positive and negative masks
        positive_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # [B, B]
        negative_mask = 1 - positive_mask  # [B, B]

        # For each anchor, find hardest negatives
        # Subtract diagonal to exclude self-similarity
        diag_mask = torch.eye(len(vision_features), device=vision_features.device)
        similarity_no_diag = similarity_matrix - 2 * diag_mask  # Subtract to make diagonal very negative

        # Find hardest negatives (lowest similarities)
        hardest_negative_sim = torch.max(similarity_no_diag - negative_mask * 1e9, dim=1)[0]  # [B]

        # Find hardest positives (excluding anchor itself)
        positive_similarities = similarity_matrix - diag_mask * 1e9  # Mask diagonal
        hardest_positive_sim = torch.max(positive_similarities - (1 - positive_mask) * 1e9, dim=1)[0]  # [B]

        # Compute triplet loss with hardest negatives
        triplet_loss = torch.clamp(
            self.margin + hardest_negative_sim - hardest_positive_sim,
            min=0.0
        ).mean()

        return triplet_loss
```

### 2. Cross-Modal Attention Mechanisms

```python
# Advanced cross-modal attention mechanisms
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """Advanced cross-modal attention for vision-language fusion"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_vision = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_text = nn.Linear(dim, dim * 3, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vision_features, text_features, attention_mask=None):
        """
        Cross-modal attention: vision and text attend to each other
        Args:
            vision_features: [B, num_vision_tokens, dim]
            text_features: [B, num_text_tokens, dim]
            attention_mask: [B, num_text_tokens] attention mask for text
        """
        B, N_v, C = vision_features.shape
        _, N_t, _ = text_features.shape

        # Compute QKV for both modalities
        q_vision, k_vision, v_vision = self.qkv_vision(vision_features).chunk(3, dim=-1)
        q_text, k_text, v_text = self.qkv_text(text_features).chunk(3, dim=-1)

        # Reshape for multi-head attention
        q_vision = q_vision.reshape(B, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_vision = k_vision.reshape(B, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_vision = v_vision.reshape(B, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_text = q_text.reshape(B, N_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_text = k_text.reshape(B, N_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_text = v_text.reshape(B, N_t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Cross-modal attention: vision attends to text
        attn_v2t = (q_vision @ k_text.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            # Expand attention mask to match attention matrix
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N_t]
            attn_v2t = attn_v2t.masked_fill(expanded_mask == 0, float('-inf'))

        attn_v2t = self.softmax(attn_v2t)
        attn_v2t = self.attn_drop(attn_v2t)
        vision_attended = (attn_v2t @ v_text).transpose(1, 2).reshape(B, N_v, C)

        # Cross-modal attention: text attends to vision
        attn_t2v = (q_text @ k_vision.transpose(-2, -1)) * self.scale
        attn_t2v = self.softmax(attn_t2v)
        attn_t2v = self.attn_drop(attn_t2v)
        text_attended = (attn_t2v @ v_vision).transpose(1, 2).reshape(B, N_t, C)

        # Apply projections
        vision_output = self.proj(vision_attended)
        text_output = self.proj(text_attended)

        vision_output = self.proj_drop(vision_output)
        text_output = self.proj_drop(text_output)

        return {
            'vision_attended': vision_output,
            'text_attended': text_output,
            'vision_to_text_attention': attn_v2t,
            'text_to_vision_attention': attn_t2v
        }

class HierarchicalCrossModalFusion(nn.Module):
    """Hierarchical fusion of vision and language at different levels"""
    def __init__(self, dim, num_heads=8):
        super().__init__()

        # Low-level fusion (pixel/word level)
        self.low_level_fusion = CrossModalAttention(dim, num_heads)

        # Mid-level fusion (region/sentence level)
        self.mid_level_fusion = CrossModalAttention(dim, num_heads)

        # High-level fusion (global understanding)
        self.high_level_fusion = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Fusion gates for adaptive combination
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            ) for _ in range(3)  # For low, mid, high levels
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, vision_features, text_features, attention_mask=None):
        """
        Hierarchical fusion of vision and language features
        Args:
            vision_features: [B, num_vision_tokens, dim] (could be patches or regions)
            text_features: [B, num_text_tokens, dim] (words or subwords)
            attention_mask: [B, num_text_tokens] attention mask
        """
        B, N_v, C = vision_features.shape
        _, N_t, _ = text_features.shape

        # Low-level fusion: fine-grained alignment
        low_fusion_result = self.low_level_fusion(
            vision_features, text_features, attention_mask
        )
        low_vision_fused = low_fusion_result['vision_attended']
        low_text_fused = low_fusion_result['text_attended']

        # Mid-level fusion: semantic alignment
        # Group features for mid-level processing
        mid_vision = self.group_features(low_vision_fused, group_size=4)  # [B, N_v//4, C]
        mid_text = self.group_features(low_text_fused, group_size=3)     # [B, N_t//3, C]

        mid_fusion_result = self.mid_level_fusion(mid_vision, mid_text, attention_mask)
        mid_vision_fused = mid_fusion_result['vision_attended']
        mid_text_fused = mid_fusion_result['text_attended']

        # High-level fusion: global understanding
        # Global average pooling for global features
        global_vision = torch.mean(low_vision_fused, dim=1, keepdim=True)  # [B, 1, C]
        global_text = torch.mean(low_text_fused, dim=1, keepdim=True)      # [B, 1, C]

        high_vision, _ = self.high_level_fusion(
            query=global_vision,
            key=global_text,
            value=global_text
        )
        high_text, _ = self.high_level_fusion(
            query=global_text,
            key=global_vision,
            value=global_vision
        )

        # Adaptive fusion using gates
        # Upsample high-level features back to match low-level dimensions
        high_vision_up = high_vision.expand(-1, N_v, -1)  # [B, N_v, C]
        high_text_up = high_text.expand(-1, N_t, -1)      # [B, N_t, C]

        # Apply fusion gates
        gate_vision = self.fusion_gates[0](
            torch.cat([low_vision_fused, high_vision_up], dim=-1)
        )
        gate_text = self.fusion_gates[1](
            torch.cat([low_text_fused, high_text_up], dim=-1)
        )

        # Final fused representations
        final_vision = gate_vision * low_vision_fused + (1 - gate_vision) * high_vision_up
        final_text = gate_text * low_text_fused + (1 - gate_text) * high_text_up

        return {
            'fused_vision': final_vision,
            'fused_text': final_text,
            'low_level_features': low_fusion_result,
            'mid_level_features': mid_fusion_result,
            'high_level_features': {'vision': high_vision, 'text': high_text},
            'fusion_gates': [gate_vision, gate_text]
        }

    def group_features(self, features, group_size):
        """Group features for hierarchical processing"""
        B, N, C = features.shape
        if N % group_size != 0:
            # Pad with zeros if necessary
            padding = group_size - (N % group_size)
            features = F.pad(features, (0, 0, 0, padding), mode='constant', value=0)
            N = features.shape[1]

        grouped = features.view(B, N // group_size, group_size, C)
        # Average pooling within groups
        grouped_features = torch.mean(grouped, dim=2)  # [B, N//group_size, C]
        return grouped_features
```

## Training Strategies for VLMs

### 1. Multi-Stage Training

```python
# Multi-stage training for vision-language models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class MultiStageVLMTrainer:
    def __init__(self, model, train_loader, val_loader, stages_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.stages_config = stages_config

        # Initialize optimizers for different stages
        self.optimizers = {}
        self.schedulers = {}

        for stage_name, stage_config in stages_config.items():
            # Create optimizer for specific parameters
            if stage_config.get('train_vision', False):
                params = list(self.model.vision_encoder.parameters())
            elif stage_config.get('train_text', False):
                params = list(self.model.text_decoder.parameters())
            else:
                params = list(self.model.parameters())

            optimizer = torch.optim.AdamW(
                params,
                lr=stage_config['learning_rate'],
                weight_decay=stage_config.get('weight_decay', 0.01)
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=stage_config['epochs'] * len(train_loader)
            )

            self.optimizers[stage_name] = optimizer
            self.schedulers[stage_name] = scheduler

        self.current_stage = 0
        self.global_step = 0

    def train(self):
        """Execute multi-stage training"""
        for stage_idx, (stage_name, stage_config) in enumerate(self.stages_config.items()):
            print(f"\n=== Starting Stage {stage_idx + 1}: {stage_name} ===")
            self.current_stage = stage_idx

            # Freeze/unfreeze parameters based on stage
            self.configure_parameter_freezing(stage_name, stage_config)

            # Train for this stage
            self.train_stage(stage_name, stage_config)

    def configure_parameter_freezing(self, stage_name, stage_config):
        """Configure which parameters to freeze/unfreeze for current stage"""
        # Unfreeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = True

        # Freeze specific parameters based on stage config
        if stage_config.get('freeze_vision', False):
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False

        if stage_config.get('freeze_text', False):
            for param in self.model.text_decoder.parameters():
                param.requires_grad = False

        if stage_config.get('freeze_cross_attention', False):
            for param in self.model.cross_attention.parameters():
                param.requires_grad = False

    def train_stage(self, stage_name, stage_config):
        """Train for a specific stage"""
        optimizer = self.optimizers[stage_name]
        scheduler = self.schedulers[stage_name]

        num_epochs = stage_config['epochs']
        accumulation_steps = stage_config.get('accumulation_steps', 1)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Forward pass
                outputs = self.model(
                    images=batch['images'].cuda(),
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    labels=batch.get('labels', None)
                )

                loss = outputs['loss'] if isinstance(outputs, dict) else outputs

                # Scale loss for gradient accumulation
                scaled_loss = loss / accumulation_steps

                # Backward pass
                scaled_loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update learning rate
                    scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                self.global_step += 1

                # Validation checkpoint
                if self.global_step % stage_config.get('val_interval', 100) == 0:
                    val_metrics = self.validate()
                    print(f"\nValidation at step {self.global_step}: {val_metrics}")

            avg_loss = total_loss / num_batches
            print(f"Stage {stage_name}, Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    def validate(self):
        """Validate model performance"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(
                    images=batch['images'].cuda(),
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    labels=batch.get('labels', None)
                )

                if isinstance(outputs, dict):
                    loss = outputs['loss']
                    accuracy = outputs.get('accuracy', 0)
                else:
                    loss = outputs
                    accuracy = 0  # Calculate accuracy if needed

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0

        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }

# Example multi-stage training configuration
training_config = {
    'stage_1_pretrain_vision': {
        'epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'train_vision': True,
        'freeze_text': True,
        'val_interval': 50
    },
    'stage_2_pretrain_text': {
        'epochs': 5,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'train_text': True,
        'freeze_vision': True,
        'val_interval': 50
    },
    'stage_3_joint_finetuning': {
        'epochs': 20,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'accumulation_steps': 4,
        'val_interval': 100
    }
}
```

### 2. Curriculum Learning for VLMs

```python
# Curriculum learning for vision-language models
class VisionLanguageCurriculum:
    def __init__(self, difficulty_levels):
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        self.performance_history = []

    def evaluate_performance(self, model, val_loader):
        """Evaluate model performance to determine readiness for next level"""
        model.eval()
        total_accuracy = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    images=batch['images'].cuda(),
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda()
                )

                # Calculate accuracy based on your specific task
                if 'accuracy' in outputs:
                    total_accuracy += outputs['accuracy'] * len(batch['images'])
                else:
                    # Calculate accuracy from predictions vs targets
                    predictions = outputs['predictions'] if 'predictions' in outputs else outputs
                    targets = batch['targets'].cuda()
                    accuracy = (predictions == targets).float().mean()
                    total_accuracy += accuracy * len(batch['images'])

                total_samples += len(batch['images'])

        avg_accuracy = total_accuracy / total_samples
        return avg_accuracy.item()

    def should_advance_curriculum(self, current_accuracy):
        """Determine if model is ready to advance to next difficulty level"""
        current_threshold = self.difficulty_levels[self.current_level]['threshold']

        if current_accuracy >= current_threshold:
            # Check consistency over multiple evaluations
            self.performance_history.append(current_accuracy)

            if len(self.performance_history) >= 3:  # At least 3 evaluations
                recent_performance = sum(self.performance_history[-3:]) / 3
                if recent_performance >= current_threshold:
                    return True

        return False

    def get_current_dataset_sampler(self):
        """Get appropriate dataset sampler for current difficulty level"""
        current_config = self.difficulty_levels[self.current_level]

        if current_config['type'] == 'complexity_based':
            return ComplexityBasedSampler(current_config['complexity_params'])
        elif current_config['type'] == 'domain_specific':
            return DomainSpecificSampler(current_config['domain_params'])
        else:
            return StandardSampler()

class ComplexityBasedSampler:
    """Sampler that adjusts complexity based on difficulty level"""
    def __init__(self, complexity_params):
        self.complexity_params = complexity_params

    def sample_batch(self, dataset, batch_size):
        """Sample batch with appropriate complexity"""
        # Sample based on complexity parameters
        if self.complexity_params['image_complexity'] == 'simple':
            # Filter for simpler images (less objects, clearer backgrounds)
            simple_indices = [i for i, sample in enumerate(dataset) if self.is_simple_image(sample)]
        elif self.complexity_params['image_complexity'] == 'complex':
            # Filter for more complex images (more objects, cluttered backgrounds)
            complex_indices = [i for i, sample in enumerate(dataset) if self.is_complex_image(sample)]

        # Sample from filtered indices
        selected_indices = self.select_indices(complex_indices if self.complexity_params['image_complexity'] == 'complex' else simple_indices, batch_size)
        return dataset[selected_indices]

    def is_simple_image(self, sample):
        """Determine if image is simple based on object count, etc."""
        # This would analyze the image complexity
        return True  # Placeholder

    def is_complex_image(self, sample):
        """Determine if image is complex based on object count, etc."""
        # This would analyze the image complexity
        return True  # Placeholder

# Example curriculum difficulty levels
curriculum_levels = [
    {
        'name': 'basic_recognition',
        'type': 'complexity_based',
        'complexity_params': {
            'image_complexity': 'simple',
            'text_length': 'short',
            'vocabulary_size': 'small'
        },
        'threshold': 0.70  # 70% accuracy required to advance
    },
    {
        'name': 'intermediate_understanding',
        'type': 'complexity_based',
        'complexity_params': {
            'image_complexity': 'medium',
            'text_length': 'medium',
            'vocabulary_size': 'medium'
        },
        'threshold': 0.75  # 75% accuracy required to advance
    },
    {
        'name': 'advanced_reasoning',
        'type': 'complexity_based',
        'complexity_params': {
            'image_complexity': 'complex',
            'text_length': 'long',
            'vocabulary_size': 'large'
        },
        'threshold': 0.80  # 80% accuracy required to advance
    }
]
```

## Evaluation Metrics for VLMs

### 1. Vision-Language Alignment Metrics

```python
# Comprehensive evaluation metrics for vision-language models
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

class VisionLanguageEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_alignment(self, vision_features, text_features):
        """Evaluate vision-language alignment quality"""
        # Compute similarity matrix
        similarity_matrix = torch.matmul(vision_features, text_features.t())

        # Compute retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(similarity_matrix)

        # Compute alignment diversity
        alignment_diversity = self.compute_alignment_diversity(similarity_matrix)

        return {
            'retrieval_metrics': retrieval_metrics,
            'alignment_diversity': alignment_diversity,
            'similarity_stats': self.compute_similarity_statistics(similarity_matrix)
        }

    def compute_retrieval_metrics(self, similarity_matrix):
        """Compute image-to-text and text-to-image retrieval metrics"""
        B = similarity_matrix.shape[0]

        # Image-to-text retrieval (given image, find matching text)
        i2t_ranks = []
        for i in range(B):
            # Sort similarities in descending order
            _, indices = torch.sort(similarity_matrix[i, :], descending=True)
            # Find rank of correct text (diagonal element)
            rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
            i2t_ranks.append(rank)

        # Text-to-image retrieval (given text, find matching image)
        t2i_ranks = []
        for i in range(B):
            _, indices = torch.sort(similarity_matrix[:, i], descending=True)
            rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
            t2i_ranks.append(rank)

        # Compute metrics
        i2t_metrics = self.compute_ranking_metrics(i2t_ranks)
        t2i_metrics = self.compute_ranking_metrics(t2i_ranks)

        return {
            'i2t': i2t_metrics,
            't2i': t2i_metrics,
            'combined': {
                'r1': (i2t_metrics['r1'] + t2i_metrics['r1']) / 2,
                'r5': (i2t_metrics['r5'] + t2i_metrics['r5']) / 2,
                'r10': (i2t_metrics['r10'] + t2i_metrics['r10']) / 2,
                'medr': (i2t_metrics['medr'] + t2i_metrics['medr']) / 2,
                'meanr': (i2t_metrics['meanr'] + t2i_metrics['meanr']) / 2
            }
        }

    def compute_ranking_metrics(self, ranks):
        """Compute ranking-based retrieval metrics"""
        ranks = np.array(ranks)

        # R@1, R@5, R@10
        r1 = 100.0 * len(np.where(ranks <= 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks <= 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks <= 10)[0]) / len(ranks)

        # Median rank
        medr = np.median(ranks)

        # Mean rank
        meanr = np.mean(ranks)

        return {
            'r1': r1,
            'r5': r5,
            'r10': r10,
            'medr': medr,
            'meanr': meanr
        }

    def compute_alignment_diversity(self, similarity_matrix):
        """Compute diversity of alignment"""
        # Row-wise entropy (how distributed each image's attention is)
        row_entropies = []
        for i in range(similarity_matrix.shape[0]):
            probs = F.softmax(similarity_matrix[i, :], dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            row_entropies.append(entropy.item())

        # Column-wise entropy (how distributed each text's attention is)
        col_entropies = []
        for j in range(similarity_matrix.shape[1]):
            probs = F.softmax(similarity_matrix[:, j], dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            col_entropies.append(entropy.item())

        return {
            'row_entropy_mean': np.mean(row_entropies),
            'row_entropy_std': np.std(row_entropies),
            'col_entropy_mean': np.mean(col_entropies),
            'col_entropy_std': np.std(col_entropies),
            'alignment_uniformity': self.compute_uniformity(similarity_matrix)
        }

    def compute_uniformity(self, similarity_matrix):
        """Compute uniformity of similarity distribution"""
        # Convert similarities to probabilities
        probs = F.softmax(similarity_matrix, dim=-1)
        # Compute entropy of probability distribution
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        return entropy.item()

    def compute_similarity_statistics(self, similarity_matrix):
        """Compute basic statistics of similarity matrix"""
        return {
            'mean_similarity': similarity_matrix.mean().item(),
            'std_similarity': similarity_matrix.std().item(),
            'min_similarity': similarity_matrix.min().item(),
            'max_similarity': similarity_matrix.max().item(),
            'median_similarity': similarity_matrix.median().item()
        }

    def evaluate_generation_quality(self, generated_texts, reference_texts):
        """Evaluate quality of generated text"""
        metrics = {}

        # BLEU scores
        metrics['bleu_scores'] = self.compute_bleu_scores(generated_texts, reference_texts)

        # ROUGE scores
        metrics['rouge_scores'] = self.compute_rouge_scores(generated_texts, reference_texts)

        # Distinct n-grams (diversity)
        metrics['distinct_ngrams'] = self.compute_distinct_ngrams(generated_texts)

        # Embedding-based similarity
        metrics['embedding_similarity'] = self.compute_embedding_similarity(generated_texts, reference_texts)

        return metrics

    def compute_bleu_scores(self, generated_texts, reference_texts):
        """Compute BLEU scores (simplified implementation)"""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            bleu_scores = []
            for gen, ref in zip(generated_texts, reference_texts):
                gen_tokens = gen.split()
                ref_tokens = [ref.split()]  # List of reference sentences
                bleu = sentence_bleu(ref_tokens, gen_tokens)
                bleu_scores.append(bleu)
            return {
                'bleu_1': np.mean([s for s in bleu_scores]),
                'bleu_4': np.mean([s for s in bleu_scores])  # Simplified
            }
        except ImportError:
            return {'bleu_1': 0.0, 'bleu_4': 0.0}

    def compute_rouge_scores(self, generated_texts, reference_texts):
        """Compute ROUGE scores (simplified implementation)"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            rouge_scores = defaultdict(list)
            for gen, ref in zip(generated_texts, reference_texts):
                score = scorer.score(ref, gen)
                for metric, value in score.items():
                    rouge_scores[metric].append(value.fmeasure)

            return {k: np.mean(v) for k, v in rouge_scores.items()}
        except ImportError:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def compute_distinct_ngrams(self, texts):
        """Compute distinct n-gram diversity"""
        all_unigrams = set()
        all_bigrams = set()
        total_unigrams = 0
        total_bigrams = 0

        for text in texts:
            tokens = text.split()
            # Unigrams
            all_unigrams.update(tokens)
            total_unigrams += len(tokens)

            # Bigrams
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            all_bigrams.update(bigrams)
            total_bigrams += len(bigrams)

        return {
            'distinct_1': len(all_unigrams) / total_unigrams if total_unigrams > 0 else 0,
            'distinct_2': len(all_bigrams) / total_bigrams if total_bigrams > 0 else 0
        }

    def compute_embedding_similarity(self, generated_texts, reference_texts):
        """Compute similarity using sentence embeddings"""
        # This would typically use a pre-trained sentence transformer
        # For now, we'll use a simple approach
        similarities = []
        for gen, ref in zip(generated_texts, reference_texts):
            # Simple overlap-based similarity (in practice, use sentence transformers)
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            overlap = len(gen_words.intersection(ref_words))
            union = len(gen_words.union(ref_words))
            jaccard_sim = overlap / union if union > 0 else 0
            similarities.append(jaccard_sim)

        return np.mean(similarities) if similarities else 0.0

    def evaluate_task_performance(self, model, test_loader, task_type='captioning'):
        """Evaluate model on specific vision-language tasks"""
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    images=batch['images'].cuda(),
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda()
                )

                if task_type == 'captioning':
                    predictions = self.decode_captions(outputs['logits'])
                    targets = batch['captions']
                elif task_type == 'vqa':
                    predictions = self.decode_answers(outputs['logits'])
                    targets = batch['answers']
                elif task_type == 'classification':
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    targets = batch['labels']

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        # Compute task-specific metrics
        if task_type in ['captioning', 'vqa']:
            generation_metrics = self.evaluate_generation_quality(all_predictions, all_targets)
            return generation_metrics
        else:
            accuracy = accuracy_score(all_targets, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted'
            )
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
```

## Next Steps

In the next section, we'll explore how to integrate these vision-language models with action generation systems to create complete Vision-Language-Action (VLA) systems that can understand human commands and execute appropriate robotic behaviors. We'll learn about action space representation, trajectory generation, and the integration challenges involved in creating end-to-end trainable VLA systems.