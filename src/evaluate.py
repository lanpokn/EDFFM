import torch
from tqdm import tqdm
# 您可以从sklearn或自己实现这些指标
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Evaluator:
    def __init__(self, model, test_loader, config, device):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for raw_events, labels in tqdm(self.test_loader, desc="Evaluating"):
                raw_events, labels = raw_events.to(self.device), labels.to(self.device)
                
                # 同上，需要特征提取
                features = raw_events

                predictions = self.model(features)
                
                # 将概率转换为0/1的预测
                preds = (predictions > 0.5).float()
                
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

        # 将所有批次的结果合并
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print("\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("--------------------------")

        # TODO: 您可以在这里添加其他指标，如SNR的计算
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}