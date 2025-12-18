#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Query-Aware Router for Multi-Modal Large Language Model

This script trains a lightweight router network that predicts whether each
vision encoder layer (6, 12, 18, or 23) can correctly answer a given question.

Training Strategy:
- Multi-label classification: Each layer has a binary classifier (can answer / cannot answer)
- Uses layer_results as ground truth labels
- BCEWithLogitsLoss for training

Inference Strategy:
- Select the earliest layer with confidence > threshold
- Prioritizes shallow layers for efficiency

Author: AI Research Engineer
Date: 2024
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from PIL import Image
from tqdm import tqdm
import numpy as np

from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ============================================================================
# Configuration
# ============================================================================

# Path to the root directory containing images (modify this as needed)
IMAGE_ROOT = "../LLaVA_/playground/data"

# Path to the router labels JSON file
LABELS_JSON_PATH = "playground/data/router_labels.json"

# Model paths
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
HIDDEN_DIM = 256
NUM_CLASSES = 4  # layer_6, layer_12, layer_18, layer_23

# Data split ratio
TRAIN_RATIO = 0.9
TEST_RATIO = 0.1

# Random seed for reproducibility
SEED = 42

# Layer names for reporting
LAYER_NAMES = ["layer_6", "layer_12", "layer_18", "layer_23"]

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Automatically detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


# ============================================================================
# Dataset Class
# ============================================================================

class RouterDataset(Dataset):
    """
    Custom Dataset for Query-Aware Router training.
    
    Each sample contains:
    - image: Preprocessed image tensor
    - question: Raw text question
    - layer_labels: Multi-label targets [4] indicating if each layer can answer correctly
    """
    
    def __init__(
        self,
        data: List[Dict],
        image_root: str,
        clip_processor: CLIPImageProcessor,
        clip_tokenizer: CLIPTokenizer,
        max_text_length: int = 77  # CLIP default max length
    ):
        """
        Args:
            data: List of dictionaries containing image path, question, and layer_results
            image_root: Root directory for images
            clip_processor: CLIP image processor for preprocessing
            clip_tokenizer: CLIP tokenizer for text processing
            max_text_length: Maximum length for text tokenization (CLIP default: 77)
        """
        self.data = data
        self.image_root = image_root
        self.clip_processor = clip_processor
        self.clip_tokenizer = clip_tokenizer
        self.max_text_length = max_text_length
        
        # Layer keys mapping
        self.layer_keys = ["6", "12", "18", "23"]
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Load and preprocess image
        image_path = os.path.join(self.image_root, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.clip_processor(
                images=image, 
                return_tensors="pt"
            )["pixel_values"].squeeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            pixel_values = torch.zeros(3, 336, 336)
        
        # Tokenize question using CLIP tokenizer
        text_encoding = self.clip_tokenizer(
            item["question"],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        # Extract layer_results as multi-label targets
        # layer_results: {"6": true/false, "12": true/false, "18": true/false, "23": true/false}
        layer_results = item.get("layer_results", {})
        layer_labels = []
        for key in self.layer_keys:
            # Convert boolean to float (1.0 for True, 0.0 for False)
            layer_labels.append(1.0 if layer_results.get(key, False) else 0.0)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "layer_labels": torch.tensor(layer_labels, dtype=torch.float32),
        }


# ============================================================================
# Model Architecture
# ============================================================================

class VisionBranchEarlyExit(nn.Module):
    """
    Vision branch that only computes up to layer 6 of CLIP ViT.
    
    This implements early exit to demonstrate inference acceleration:
    instead of running the full 24-layer ViT, we stop at layer 6.
    """
    
    def __init__(self, clip_model_name: str):
        super().__init__()
        
        # Load CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        
        # Freeze all parameters - backbone should not be trained
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        # Get the hidden size from config
        self.hidden_size = self.vision_model.config.hidden_size  # 1024 for ViT-L
        
        print(f"Loaded CLIP Vision Model: {clip_model_name}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Number of layers: {self.vision_model.config.num_hidden_layers}")
        print(f"  - All parameters frozen (not trainable)")
        
    def forward(self, pixel_values: torch.Tensor, return_patches: bool = True) -> torch.Tensor:
        """
        Forward pass with early exit at layer 6.
        
        Args:
            pixel_values: Input images [batch_size, 3, 336, 336]
            return_patches: If True, return all patch tokens [B, 577, D].
                          If False, return only CLS token [B, D].
            
        Returns:
            features: Vision features from layer 6
                     - If return_patches=True: [batch_size, 577, hidden_size]
                     - If return_patches=False: [batch_size, hidden_size]
        """
        # Get the vision transformer encoder
        vision_encoder = self.vision_model.vision_model
        
        # Step 1: Patch embedding
        # [batch_size, num_patches + 1, hidden_size] = [B, 577, 1024]
        hidden_states = vision_encoder.embeddings(pixel_values)
        
        # Step 2: Pre-layer norm (if exists)
        hidden_states = vision_encoder.pre_layrnorm(hidden_states)
        
        # Step 3: Only process through the first 6 layers (index 0-5)
        # This is the key optimization - we don't run the full ViT
        for i, layer in enumerate(vision_encoder.encoder.layers[:6]):
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False
            )[0]
        
        # Step 4: Return features based on return_patches flag
        if return_patches:
            # Return all patch tokens (including CLS) for Cross-Attention
            # [batch_size, 577, hidden_size]
            return hidden_states
        else:
            # Return only CLS token (first token) as the global representation
            # [batch_size, hidden_size]
            return hidden_states[:, 0, :]


class TextBranch(nn.Module):
    """
    Text branch using CLIP Text Encoder.
    
    Using CLIP's text encoder instead of BERT-tiny ensures that the text
    features are naturally aligned with CLIP's vision features, making
    the router's decision easier to learn.
    """
    
    def __init__(self, clip_model_name: str):
        super().__init__()
        
        # Load CLIP Text Encoder (same model as Vision Encoder)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        
        # Freeze all parameters - backbone should not be trained
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Get the hidden size from config
        self.hidden_size = self.text_model.config.hidden_size  # 768 for CLIP-L
        
        print(f"Loaded CLIP Text Model: {clip_model_name}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Number of layers: {self.text_model.config.num_hidden_layers}")
        print(f"  - All parameters frozen (not trainable)")
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to extract text features.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            features: Text features from pooler output [batch_size, hidden_size]
        """
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use pooler_output which is the [EOS] token representation
        # projected by a linear layer, representing the whole sentence
        pooled_output = outputs.pooler_output
        
        return pooled_output


class QueryAwareRouter(nn.Module):
    """
    Query-Aware Router Network for dynamic layer selection.
    
    This network takes an image and a text question as input, and predicts
    which vision encoder layer (6, 12, 18, or 23) should be used for
    generating the response.
    
    Architecture:
    - Vision Branch: CLIP ViT (frozen, early exit at layer 6) -> returns Patch Tokens
    - Text Branch: CLIP Text Encoder (frozen) -> returns [EOS] token
    - Fusion: Cross-Attention (Text queries Image patches) -> MLP classifier
    
    The Cross-Attention mechanism allows the text query to attend to relevant
    image patches, enabling the router to make spatially-aware decisions:
    - If the question asks about details (e.g., "what text?"), attention focuses
      on specific patches, and router can confidently choose earlier layers.
    - If the question asks about semantics (e.g., "why?"), attention spreads
      across patches, indicating deeper layers are needed.
    """
    
    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL_NAME,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.1,
        num_attention_heads: int = 4
    ):
        super().__init__()
        
        print("=" * 60)
        print("Initializing Query-Aware Router (Cross-Attention Architecture)")
        print("=" * 60)
        
        # Vision branch (frozen, early exit) - returns patch tokens
        self.vision_branch = VisionBranchEarlyExit(clip_model_name)
        
        # Text branch (CLIP Text Encoder, frozen)
        self.text_branch = TextBranch(clip_model_name)
        
        # Projection layers to map features to the same dimension
        self.vision_projector = nn.Sequential(
            nn.Linear(self.vision_branch.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_branch.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-Attention module: Text (Query) attends to Image Patches (Key/Value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head (simpler now since cross-attention does the fusion)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"\nArchitecture details:")
        print(f"  - Vision: {self.vision_branch.hidden_size} -> {hidden_dim} (Patch Tokens)")
        print(f"  - Text: {self.text_branch.hidden_size} -> {hidden_dim} (Query)")
        print(f"  - Cross-Attention: {num_attention_heads} heads, dim={hidden_dim}")
        print(f"  - Classifier: {hidden_dim} -> {num_classes}")
        print("=" * 60)
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the router with Cross-Attention.
        
        Args:
            pixel_values: Input images [batch_size, 3, 336, 336]
            input_ids: Tokenized questions [batch_size, seq_len]
            attention_mask: Attention masks [batch_size, seq_len]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # 1. Vision branch: Get Patch Features [B, 577, 1024] (CLS + 576 Patches)
        with torch.no_grad():
            vision_features = self.vision_branch(pixel_values, return_patches=True)
        
        # 2. Text branch: Get pooled text features [B, 768]
        with torch.no_grad():
            text_features = self.text_branch(input_ids, attention_mask)
        
        # 3. Project to common dimension
        # Vision: [B, 577, 1024] -> [B, 577, hidden_dim]
        vision_projected = self.vision_projector(vision_features)
        # Text: [B, 768] -> [B, 1, hidden_dim] (add sequence dimension for attention)
        text_projected = self.text_projector(text_features).unsqueeze(1)
        
        # 4. Cross-Attention: Text (Query) attends to Image Patches (Key/Value)
        # This allows the model to focus on relevant image regions based on the question
        # Query: [B, 1, hidden_dim], Key/Value: [B, 577, hidden_dim]
        attn_output, attn_weights = self.cross_attn(
            query=text_projected,
            key=vision_projected,
            value=vision_projected
        )
        # attn_output: [B, 1, hidden_dim]
        
        # 5. Apply layer norm and squeeze
        attn_output = self.attn_norm(attn_output)
        fused_features = attn_output.squeeze(1)  # [B, hidden_dim]
        
        # 6. Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The router model
        dataloader: Training data loader
        criterion: Loss function (BCEWithLogitsLoss)
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=True)
    
    for batch in pbar:
        # Move data to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        layer_labels = batch["layer_labels"].to(device)  # [B, 4] multi-label targets
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(pixel_values, input_ids, attention_mask)  # [B, 4]
        
        # Compute BCE loss for multi-label classification
        loss = criterion(logits, layer_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    confidence_threshold: float = 0.5
) -> Tuple[float, float, Dict]:
    """
    Evaluate the model on the test set.
    
    Inference Strategy:
    - Apply sigmoid to get confidence scores for each layer
    - Select the earliest layer with confidence > threshold
    - If no layer exceeds threshold, select the one with highest confidence
    - Check if the selected layer's ground truth is True (correct answer)
    
    Args:
        model: The router model
        dataloader: Test data loader
        criterion: Loss function (BCEWithLogitsLoss)
        device: Device to use
        epoch: Current epoch number
        confidence_threshold: Threshold for layer selection (default: 0.5)
        
    Returns:
        Tuple of (average loss, routing accuracy, detailed metrics dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_selected_layers = []  # Which layer was selected by the router
    all_correct = []  # Whether the selected layer can answer correctly
    all_confidences = []  # Confidence scores for all layers
    all_labels = []  # Ground truth labels
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Eval]", leave=True)
    
    for batch in pbar:
        # Move data to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        layer_labels = batch["layer_labels"].to(device)  # [B, 4]
        
        # Forward pass
        logits = model(pixel_values, input_ids, attention_mask)  # [B, 4]
        
        # Compute loss
        loss = criterion(logits, layer_labels)
        total_loss += loss.item()
        num_batches += 1
        
        # Get confidence scores (apply sigmoid)
        confidences = torch.sigmoid(logits)  # [B, 4]
        
        # For each sample, select the earliest layer with confidence > threshold
        batch_size = confidences.size(0)
        for i in range(batch_size):
            conf = confidences[i].cpu().numpy()  # [4]
            labels = layer_labels[i].cpu().numpy()  # [4]
            
            # Find earliest layer with confidence > threshold
            selected_layer = -1
            for layer_idx in range(4):
                if conf[layer_idx] > confidence_threshold:
                    selected_layer = layer_idx
                    break
            
            # If no layer exceeds threshold, select the one with highest confidence
            if selected_layer == -1:
                selected_layer = int(np.argmax(conf))
            
            # Check if the selected layer can answer correctly
            is_correct = labels[selected_layer] > 0.5
            
            all_selected_layers.append(selected_layer)
            all_correct.append(is_correct)
            all_confidences.append(conf)
            all_labels.append(labels)
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    routing_accuracy = np.mean(all_correct)
    
    # Per-layer statistics
    layer_selection_counts = [0, 0, 0, 0]
    layer_correct_counts = [0, 0, 0, 0]
    layer_total_correct = [0, 0, 0, 0]  # How many samples have this layer as correct
    
    for i, (selected, correct, labels) in enumerate(zip(all_selected_layers, all_correct, all_labels)):
        layer_selection_counts[selected] += 1
        if correct:
            layer_correct_counts[selected] += 1
        for j in range(4):
            if labels[j] > 0.5:
                layer_total_correct[j] += 1
    
    # Calculate average confidence per layer
    all_confidences = np.array(all_confidences)
    avg_confidences = np.mean(all_confidences, axis=0)
    
    metrics = {
        "selected_layers": all_selected_layers,
        "correct": all_correct,
        "layer_selection_counts": layer_selection_counts,
        "layer_correct_counts": layer_correct_counts,
        "layer_total_correct": layer_total_correct,
        "avg_confidences": avg_confidences,
        "all_labels": all_labels,
    }
    
    return avg_loss, routing_accuracy, metrics


def print_evaluation_results(
    epoch: int,
    train_loss: float,
    eval_loss: float,
    accuracy: float,
    metrics: Dict
):
    """Print formatted evaluation results for multi-label routing."""
    print("\n" + "=" * 70)
    print(f"Epoch {epoch + 1} Results")
    print("=" * 70)
    print(f"  Training Loss:   {train_loss:.4f}")
    print(f"  Evaluation Loss: {eval_loss:.4f}")
    print(f"  Routing Accuracy: {accuracy * 100:.2f}%")
    print(f"  (Routing Accuracy = selected layer can answer correctly)")
    
    print("\n" + "-" * 70)
    print("Average Confidence per Layer:")
    print("-" * 70)
    avg_conf = metrics["avg_confidences"]
    for i, layer_name in enumerate(LAYER_NAMES):
        bar = "█" * int(avg_conf[i] * 50)
        print(f"  {layer_name}: {avg_conf[i]:.4f} {bar}")
    
    print("\n" + "-" * 70)
    print("Layer Selection Distribution:")
    print("-" * 70)
    total_samples = sum(metrics["layer_selection_counts"])
    for i, layer_name in enumerate(LAYER_NAMES):
        count = metrics["layer_selection_counts"][i]
        correct = metrics["layer_correct_counts"][i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        acc = (correct / count * 100) if count > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {layer_name}: {count:>5} ({percentage:>5.1f}%) | Acc: {acc:>5.1f}% {bar}")
    
    print("\n" + "-" * 70)
    print("Ground Truth Distribution (samples where layer can answer):")
    print("-" * 70)
    for i, layer_name in enumerate(LAYER_NAMES):
        count = metrics["layer_total_correct"][i]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  {layer_name}: {count:>5} ({percentage:>5.1f}%) {bar}")
    
    print("=" * 70 + "\n")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main function to run the training pipeline."""
    
    print("\n" + "=" * 70)
    print("Query-Aware Router Training Script")
    print("=" * 70 + "\n")
    
    # Set random seed
    set_seed(SEED)
    print(f"Random seed set to: {SEED}")
    
    # Get device
    device = get_device()
    
    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"\nLoading data from: {LABELS_JSON_PATH}")
    
    with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # Analyze layer_results distribution (multi-label statistics)
    layer_true_counts = {layer: 0 for layer in LAYER_NAMES}
    valid_samples = 0
    
    for item in all_data:
        layer_results = item.get("layer_results", {})
        if layer_results:
            valid_samples += 1
            for layer_name in LAYER_NAMES:
                layer_key = layer_name.replace("layer_", "")  # "layer_6" -> "6"
                if layer_results.get(layer_key, False):
                    layer_true_counts[layer_name] += 1
    
    print(f"\nMulti-label statistics (samples where layer can answer correctly):")
    for layer_name in LAYER_NAMES:
        count = layer_true_counts[layer_name]
        percentage = (count / valid_samples) * 100 if valid_samples > 0 else 0
        print(f"  {layer_name}: {count}/{valid_samples} ({percentage:.1f}%)")
    
    # Shuffle and split data
    random.shuffle(all_data)
    split_idx = int(len(all_data) * TRAIN_RATIO)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training set: {len(train_data)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Test set: {len(test_data)} samples ({TEST_RATIO*100:.0f}%)")
    
    # -------------------------------------------------------------------------
    # Initialize Processors and Tokenizers
    # -------------------------------------------------------------------------
    print("\nLoading CLIP processor and BERT tokenizer...")
    
    clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    
    # -------------------------------------------------------------------------
    # Create Datasets and DataLoaders
    # -------------------------------------------------------------------------
    print(f"\nCreating datasets with IMAGE_ROOT: {IMAGE_ROOT}")
    
    train_dataset = RouterDataset(
        data=train_data,
        image_root=IMAGE_ROOT,
        clip_processor=clip_processor,
        clip_tokenizer=clip_tokenizer
    )
    
    test_dataset = RouterDataset(
        data=test_data,
        image_root=IMAGE_ROOT,
        clip_processor=clip_processor,
        clip_tokenizer=clip_tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # -------------------------------------------------------------------------
    # Initialize Model
    # -------------------------------------------------------------------------
    print("\nInitializing model...")
    
    model = QueryAwareRouter(
        clip_model_name=CLIP_MODEL_NAME,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )
    
    model = model.to(device)
    
    trainable_params = model.get_trainable_params()
    total_params = model.get_total_params()
    
    print(f"\nModel parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Initialize Loss, Optimizer, and Scheduler
    # -------------------------------------------------------------------------
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Only optimize trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    print(f"\nTraining configuration:")
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Loss: BCEWithLogitsLoss (multi-label classification)")
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")
    
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        # Evaluate on test set
        eval_loss, accuracy, metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        
        # Print results
        print_evaluation_results(epoch, train_loss, eval_loss, accuracy, metrics)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            
            # Save model checkpoint
            checkpoint_path = "best_router_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }, checkpoint_path)
            print(f"💾 New best model saved to {checkpoint_path}")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Best accuracy: {best_accuracy * 100:.2f}% (Epoch {best_epoch})")
    print(f"  Model saved to: best_router_model.pt")
    print("=" * 70 + "\n")
    
    # Save final model as well
    final_checkpoint_path = "final_router_model.pt"
    torch.save({
        "epoch": NUM_EPOCHS - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
