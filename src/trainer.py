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
        
        # TBPTT parameters
        self.chunk_size = config['training']['chunk_size']
        print(f"Trainer initialized with TBPTT chunk_size: {self.chunk_size}")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_chunks_processed = 0

        # 外循环：遍历所有长序列 (每个epoch进行N次)
        for long_features, long_labels in tqdm(self.train_loader, desc="Epoch Progress"):
            
            # DataLoader返回(1, L, D)，需要解包
            long_features = long_features.squeeze(0).to(self.device)
            long_labels = long_labels.squeeze(0).to(self.device)

            # 内循环：在当前长序列上，以不重叠的方式进行切块
            for i in range(0, long_features.shape[0], self.chunk_size):
                
                # 1. 切分出固定长度的块
                chunk_features = long_features[i : i + self.chunk_size]
                chunk_labels = long_labels[i : i + self.chunk_size]

                # 2. 如果是最后一个不完整的块，则跳过
                if chunk_features.shape[0] != self.chunk_size:
                    continue

                # 3. 准备模型输入 (增加batch维度)
                # [chunk_size, dim] -> [1, chunk_size, dim]
                chunk_features = chunk_features.unsqueeze(0)
                
                # 4. 清空梯度
                self.optimizer.zero_grad()
                
                # 5. 模型前向传播 (状态在model内部自动重置和演化)
                predictions = self.model(chunk_features)

                # 6. 计算损失
                labels_float = chunk_labels.float().unsqueeze(0).unsqueeze(-1)
                loss = self.criterion(predictions, labels_float)
                
                # 7. 反向传播 (梯度被截断在chunk_size长度内)
                loss.backward()
                
                # 8. 更新权重
                self.optimizer.step()

                total_loss += loss.item()
                total_chunks_processed += 1
        
        return total_loss / total_chunks_processed if total_chunks_processed > 0 else 0
    
    # ... (validate_one_epoch 和 train 方法也同样简化)
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        total_chunks_processed = 0
        
        with torch.no_grad():
            # 外循环：遍历所有长序列
            for long_features, long_labels in tqdm(self.val_loader, desc="Validating"):
                
                # DataLoader返回(1, L, D)，需要解包
                long_features = long_features.squeeze(0).to(self.device)
                long_labels = long_labels.squeeze(0).to(self.device)

                # 内循环：在当前长序列上进行切块验证
                for i in range(0, long_features.shape[0], self.chunk_size):
                    
                    # 1. 切分出固定长度的块
                    chunk_features = long_features[i : i + self.chunk_size]
                    chunk_labels = long_labels[i : i + self.chunk_size]

                    # 2. 如果是最后一个不完整的块，则跳过
                    if chunk_features.shape[0] != self.chunk_size:
                        continue

                    # 3. 准备模型输入
                    chunk_features = chunk_features.unsqueeze(0)
                    
                    # 4. 模型前向传播
                    predictions = self.model(chunk_features)

                    # 5. 计算损失
                    labels_float = chunk_labels.float().unsqueeze(0).unsqueeze(-1)
                    loss = self.criterion(predictions, labels_float)
                    
                    total_loss += loss.item()
                    total_chunks_processed += 1
                    
        return total_loss / total_chunks_processed if total_chunks_processed > 0 else 0

    def train(self):
        print("🔍 DEBUG: Trainer.train() method started")
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            print(f"🔍 DEBUG: Starting epoch {epoch + 1}/{self.epochs}")
            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()
            
            print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")