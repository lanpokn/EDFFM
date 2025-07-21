import numpy as np

class FeatureExtractor:
    def __init__(self, config):
        # 从config中获取所有需要的参数
        self.h = config['data']['resolution_h']
        self.w = config['data']['resolution_w']
        self.coarse_time_window = config['feature_extractor']['coarse_time_window']
        # ... 其他参数

    def process_sequence(self, raw_events):
        """
        Processes a single sequence of raw events and returns a feature sequence.
        This is where your core PFD-inspired logic goes.
        
        Args:
            raw_events (np.array): Shape: [sequence_length, 4]
            
        Returns:
            feature_sequence (np.array): Shape: [sequence_length, output_dim]
        """
        num_events = raw_events.shape[0]
        output_dim = self.config['model']['input_feature_dim']
        feature_sequence = np.zeros((num_events, output_dim))

        # 初始化这个序列需要的状态图 (e.g., PFD's time maps)
        p_time_map = np.zeros((self.h, self.w))
        n_time_map = np.zeros((self.h, self.w))
        # ... 其他状态图

        # 遍历序列中的每一个事件
        for i in range(num_events):
            x, y, t, p = raw_events[i]
            
            # --- TODO: 在这里计算几十维的特征 ---
            # 1. 计算 Δt
            # 2. 计算粗滤特征 N_p (查询和更新 time_maps)
            # 3. 计算精滤特征 Mf, Ma, Ne (需要维护更多状态图)
            # ...
            
            # 将计算出的特征填入特征向量
            feature_vector = np.array([...]) # 您的几十维特征
            feature_sequence[i] = feature_vector
            
            # 更新这个事件对应的状态图
            # ...

        return feature_sequence