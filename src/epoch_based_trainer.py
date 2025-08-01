"""
Epoch-Based Trainer for EventMamba-FX
Implements the correct "Epoch vs. Iteration" training loop:
- Epoch-level: Data generation and feature extraction
- Iteration-level: Sliding window training on pre-computed features
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time
import psutil
import gc

from src.epoch_based_dataset import EpochBasedEventDataset


class EpochBasedTrainer:
    """
    Trainer implementing the correct Epoch-based architecture:
    - Generates long sequences and extracts features at epoch start
    - Trains on sliding windows within each epoch
    """
    
    def __init__(self, model, config, device):
        """Initialize epoch-based trainer.
        
        Args:
            model: EventDenoisingMamba model
            config: Configuration dictionary  
            device: Training device (cuda/cpu)
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize datasets (they will generate data per epoch)
        self.train_dataset = EpochBasedEventDataset(config, split='train')
        self.val_dataset = EpochBasedEventDataset(config, split='val')
        
        # Training setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        self.criterion = nn.BCELoss()
        self.epochs = self.config['training']['epochs']
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Memory monitoring
        self.process = psutil.Process()
        
        print(f"Initialized EpochBasedTrainer:")
        print(f"  Device: {device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        
    def train_one_epoch(self, epoch: int):
        """Train one epoch with the new architecture.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        print(f"\n=== TRAINING EPOCH {epoch+1}/{self.epochs} ===")
        epoch_start_time = time.time()
        
        # 1. EPOCH-LEVEL: Generate data and extract features
        print("Step 1: Epoch-level data generation...")
        self.train_dataset.generate_epoch_data(epoch)
        
        # Check memory after data generation
        memory_after_gen = self.process.memory_info().rss / (1024 * 1024)
        print(f"Memory after data generation: {memory_after_gen:.1f} MB")
        
        # 2. ITERATION-LEVEL: Train on sliding windows
        print("Step 2: Iteration-level training...")
        self.model.train()
        total_loss = 0
        num_iterations = len(self.train_dataset)
        
        print(f"Training on {num_iterations} sliding windows...")
        
        for iteration in tqdm(range(num_iterations), desc="Training iterations"):
            try:
                # Get pre-computed features and labels
                features, labels = self.train_dataset[iteration]
                
                # Move to device and add batch dimension
                features = features.unsqueeze(0).to(self.device)  # [1, seq_len, 11]
                labels = labels.unsqueeze(0).to(self.device)      # [1, seq_len]
                
                # Forward pass (no feature extraction in model!)
                self.optimizer.zero_grad()
                predictions = self.model(features)  # [1, seq_len, 1]
                
                # Compute loss
                labels_float = labels.float().unsqueeze(-1)  # [1, seq_len, 1]
                loss = self.criterion(predictions, labels_float)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Memory monitoring every 10 iterations
                if iteration % 10 == 0:
                    memory_mb = self.process.memory_info().rss / (1024 * 1024)
                    if memory_mb > 8000:  # Warning at 8GB
                        print(f"WARNING: High memory usage: {memory_mb:.1f} MB")
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                continue
        
        # Calculate average loss
        avg_loss = total_loss / max(num_iterations, 1)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1} training completed:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        return avg_loss
    
    def validate_one_epoch(self, epoch: int):
        """Validate one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        print(f"\n=== VALIDATION EPOCH {epoch+1}/{self.epochs} ===")
        
        # Generate validation data
        print("Generating validation data...")
        self.val_dataset.generate_epoch_data(epoch)
        
        # Validate
        self.model.eval()
        total_loss = 0
        num_iterations = len(self.val_dataset)
        
        print(f"Validating on {num_iterations} sliding windows...")
        
        with torch.no_grad():
            for iteration in tqdm(range(num_iterations), desc="Validation iterations"):
                try:
                    # Get pre-computed features and labels
                    features, labels = self.val_dataset[iteration]
                    
                    # Move to device and add batch dimension
                    features = features.unsqueeze(0).to(self.device)  # [1, seq_len, 11]
                    labels = labels.unsqueeze(0).to(self.device)      # [1, seq_len]
                    
                    # Forward pass
                    predictions = self.model(features)  # [1, seq_len, 1]
                    
                    # Compute loss
                    labels_float = labels.float().unsqueeze(-1)  # [1, seq_len, 1]
                    loss = self.criterion(predictions, labels_float)
                    
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"Error in validation iteration {iteration}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_iterations, 1)
        print(f"Validation completed: Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Main training loop implementing the new architecture."""
        print(f"\n🚀 Starting Epoch-Based Training ({self.epochs} epochs)")
        print(f"Architecture: Epoch-level data generation → Iteration-level sliding windows")
        
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(self.epochs):
            try:
                # Train one epoch
                train_loss = self.train_one_epoch(epoch)
                
                # Validate one epoch
                val_loss = self.validate_one_epoch(epoch)
                
                # Print epoch summary
                print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f"💾 Best model saved to {save_path} (val_loss: {val_loss:.4f})")
                
                # Memory cleanup between epochs
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ ERROR in epoch {epoch+1}: {e}")
                print("Continuing to next epoch...")
                continue
        
        total_training_time = time.time() - training_start_time
        
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"  Total time: {total_training_time:.2f}s")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Final model saved in: {self.checkpoint_dir}")
        
        return best_val_loss


def create_epoch_based_trainer(model, config, device):
    """Factory function to create epoch-based trainer."""
    return EpochBasedTrainer(model, config, device)