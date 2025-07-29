import torch
import torch.nn as nn
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        self.criterion = nn.BCELoss() # 二元交叉熵损失
        self.epochs = self.config['training']['epochs']
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        
        for raw_events, labels in tqdm(self.train_loader, desc="Training"):
            raw_events, labels = raw_events.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(raw_events)  # [batch_size, sequence_length, 1]
            
            # 调整标签格式以匹配预测输出
            labels_float = labels.float().unsqueeze(-1)  # [batch_size, sequence_length, 1]
            
            loss = self.criterion(predictions, labels_float)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    # ... (validate_one_epoch 和 train 方法也同样简化)
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for raw_events, labels in tqdm(self.val_loader, desc="Validating"):
                raw_events, labels = raw_events.to(self.device), labels.to(self.device)
                predictions = self.model(raw_events)
                
                # 调整标签格式以匹配预测输出
                labels_float = labels.float().unsqueeze(-1)
                loss = self.criterion(predictions, labels_float)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()
            
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")