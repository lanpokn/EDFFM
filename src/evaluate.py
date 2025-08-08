# src/evaluate.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Evaluator:
    def __init__(self, model, test_loader, config, device):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        # 从配置中获取chunk_size，与训练时保持一致
        self.chunk_size = config['training']['chunk_size']

    def evaluate(self):
        self.model.eval()
        
        # 用于收集所有序列的完整预测和标签
        all_predictions_list = []
        all_labels_list = []

        with torch.no_grad():
            # 外循环：遍历测试集中的每一条独立的长序列
            for long_features, long_labels in tqdm(self.test_loader, desc="Evaluating"):
                
                # ### BEGIN BUGFIX 1: STATE LEAKAGE ###
                if hasattr(self.model, 'reset_hidden_state'):
                    self.model.reset_hidden_state()
                # ### END BUGFIX 1 ###
                
                long_features = long_features.squeeze(0).to(self.device)
                long_labels = long_labels.squeeze(0) # 标签保留在CPU

                # 用于收集当前这条长序列所有块的预测结果
                outputs_for_this_sequence = []

                # ### BEGIN BUGFIX 2: CHUNK-BASED INFERENCE ###
                # 内循环：以与训练相同的方式，在长序列上进行分块推理
                for i in range(0, long_features.shape[0], self.chunk_size):
                    
                    # 获取当前块，无论其长度如何 (这会自动处理最后一个不完整的块)
                    chunk_features = long_features[i : i + self.chunk_size]

                    if chunk_features.shape[0] == 0:
                        continue

                    # 添加batch维度并送入模型
                    chunk_features = chunk_features.unsqueeze(0)
                    
                    # 模型输出的是logits，不是概率
                    logits = self.model(chunk_features)
                    
                    # 将logits转换为0/1的预测标签
                    # torch.sigmoid(logits) > 0.5 等价于 logits > 0
                    predicted_labels_chunk = (logits.squeeze() > 0).long()
                    
                    outputs_for_this_sequence.append(predicted_labels_chunk.cpu())
                # ### END BUGFIX 2 ###

                # ### BEGIN BUGFIX 3: CORRECT METRIC CALCULATION ###
                # 将当前序列所有块的预测结果拼接成一个完整的预测序列
                if outputs_for_this_sequence:
                    final_predictions_for_sequence = torch.cat(outputs_for_this_sequence)
                    
                    # 确保预测长度和标签长度一致
                    assert final_predictions_for_sequence.shape[0] == long_labels.shape[0], \
                        f"Prediction length {final_predictions_for_sequence.shape[0]} does not match label length {long_labels.shape[0]}"

                    all_predictions_list.append(final_predictions_for_sequence.numpy())
                    all_labels_list.append(long_labels.numpy())
                # ### END BUGFIX 3 ###

        # 将所有序列的结果合并成一个巨大的数组，用于计算总指标
        if not all_predictions_list:
            print("Warning: No data was evaluated.")
            return {}
            
        all_preds = np.concatenate(all_predictions_list)
        all_labels = np.concatenate(all_labels_list)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print("\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("--------------------------")
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}