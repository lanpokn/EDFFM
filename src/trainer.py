import torch
import torch.nn as nn
from tqdm import tqdm
import os
import glob

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
        # 建议使用BCEWithLogitsLoss以获得更好的数值稳定性
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = self.config['training']['epochs']
        self.checkpoint_dir = self.config['training']['checkpoint_dir']
        
        # TBPTT parameters
        self.chunk_size = config['training']['chunk_size']
        # print(f"Trainer initialized with TBPTT chunk_size: {self.chunk_size}")
        
        # 用于断点续训的状态变量
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.current_epoch = 0

        # Checkpoint保存频率 (基于全局步数)
        self.validate_every_n_steps = config['training'].get('validate_every_n_steps', 500)
        self.save_every_n_steps = config['training'].get('save_every_n_steps', 1000)
        
        # print(f"💾 Checkpoint schedule: validate every {self.validate_every_n_steps} steps, save every {self.save_every_n_steps} steps")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        
        # 外循环: 遍历长序列
        outer_loop_desc = f"Epoch {self.current_epoch+1}/{self.epochs}"
        for long_features, long_labels in tqdm(self.train_loader, desc=outer_loop_desc):
            # ### BEGIN BUGFIX 1: STATE LEAKAGE ###
            self.model.reset_hidden_state()
            # ### END BUGFIX 1 ###

            long_features = long_features.squeeze(0).to(self.device)
            long_labels = long_labels.squeeze(0).to(self.device)

            # 内循环: TBPTT切块
            for i in range(0, long_features.shape[0], self.chunk_size):
                chunk_features = long_features[i : i + self.chunk_size]
                chunk_labels = long_labels[i : i + self.chunk_size]
                if chunk_features.shape[0] < 1: # 跳过空块
                    continue
                
                # 注意：不再跳过最后一个不完整的块，让模型也学习它
                # if chunk_features.shape[0] != self.chunk_size:
                #     continue
                    
                chunk_features = chunk_features.unsqueeze(0)
                self.optimizer.zero_grad()
                
                # 假设模型输出logits (BCEWithLogitsLoss)
                predictions = self.model(chunk_features)
                
                # 调整维度以匹配损失函数
                # predictions: [1, L, 1] -> [L]
                # chunk_labels: [L]
                loss = self.criterion(predictions.squeeze(), chunk_labels.float())
                loss.backward()
                
                # ### BEGIN RISK MITIGATION 3: GRADIENT CLIPPING ###
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # ### END RISK MITIGATION 3 ###
                
                self.optimizer.step()

                # 全局步数是唯一的时间戳
                self.global_step += 1

                # 周期性验证和保存
                if self.global_step > 0 and self.global_step % self.validate_every_n_steps == 0:
                    val_loss = self.validate_one_epoch()
                    print(f"\n📊 Step {self.global_step} | Val Loss: {val_loss:.4f} | Best: {self.best_val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                
                if self.global_step > 0 and self.global_step % self.save_every_n_steps == 0:
                    self._save_checkpoint(is_best=False)
    
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        total_chunks_processed = 0
        
        with torch.no_grad():
            # 外循环：遍历所有长序列
            for long_features, long_labels in tqdm(self.val_loader, desc="Validating"):
                # ### BEGIN BUGFIX 1: STATE LEAKAGE ###
                self.model.reset_hidden_state()
                # ### END BUGFIX 1 ###

                # DataLoader返回(1, L, D)，需要解包
                long_features = long_features.squeeze(0).to(self.device)
                long_labels = long_labels.squeeze(0).to(self.device)

                # 内循环：在当前长序列上进行切块验证
                for i in range(0, long_features.shape[0], self.chunk_size):
                    
                    # 1. 切分出固定长度的块
                    chunk_features = long_features[i : i + self.chunk_size]
                    chunk_labels = long_labels[i : i + self.chunk_size]

                    if chunk_features.shape[0] < 1:
                        continue
                    
                    # 注意：验证时也不再跳过最后一个块
                    # if chunk_features.shape[0] != self.chunk_size:
                    #     continue

                    chunk_features = chunk_features.unsqueeze(0)
                    predictions = self.model(chunk_features)
                    
                    loss = self.criterion(predictions.squeeze(), chunk_labels.float())
                    
                    # 按块的长度加权损失，更公平
                    total_loss += loss.item() * len(chunk_features)
                    total_chunks_processed += len(chunk_features)
                    
        return total_loss / total_chunks_processed if total_chunks_processed > 0 else 0

    def _save_checkpoint(self, is_best=False):
        """保存检查点，支持断点续训"""
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存常规检查点，文件名包含全局步数，便于排序
        filename = os.path.join(self.checkpoint_dir, f'ckpt_step_{self.global_step:08d}.pth')
        torch.save(state, filename)
        # print(f"\n💾 Checkpoint saved to {os.path.basename(filename)}")

        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_filename)
            # print(f"🏆 Best model updated and saved")

    def _load_checkpoint(self):
        """加载最新检查点"""
        # 寻找最新的检查点
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'ckpt_step_*.pth'))
        if not checkpoints:
            # print("INFO: No checkpoint found, starting from scratch.")
            return

        latest_checkpoint_path = max(checkpoints, key=os.path.getctime)
        print(f"🔄 Resuming from checkpoint: {os.path.basename(latest_checkpoint_path)}")
        
        checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        # 关键：如果从epoch中间恢复，需要保证global_step也恢复
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"   → Epoch {self.start_epoch + 1}, Step {self.global_step}")

    def train(self):
        print("🚀 Starting TBPTT training...")
        
        # 尝试加载检查点
        self._load_checkpoint()
        
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.epochs} (step {self.global_step})")
            self.train_one_epoch() # 验证和保存逻辑已移入此函数
            
            # Epoch结束时也保存一次，以防万一
            self._save_checkpoint(is_best=False)
            
        print("🏁 Training completed.")