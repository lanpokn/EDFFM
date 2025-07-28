import numpy as np

class FeatureExtractor:
    def __init__(self, config):
        # 从config中获取所有需要的参数
        self.config = config
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
            
            # Basic features: x, y, t, p + additional dimensions (placeholder)
            x_norm = x / self.w  # Normalized x
            y_norm = y / self.h  # Normalized y
            
            # Time delta (if not first event)
            if i > 0:
                dt = t - raw_events[i-1, 2]
            else:
                dt = 0
                
            # Polarity-aware time since last event at this pixel
            if p > 0:
                dt_pixel = t - p_time_map[int(y), int(x)]
                p_time_map[int(y), int(x)] = t
            else:
                dt_pixel = t - n_time_map[int(y), int(x)] 
                n_time_map[int(y), int(x)] = t
            
            # Create basic feature vector (expandable to more dimensions)
            feature_vector = np.array([
                x_norm, y_norm, t, p,  # Basic event attributes
                dt, dt_pixel,          # Temporal features
                *np.zeros(output_dim - 6)  # Placeholder for additional features
            ])
            
            feature_sequence[i] = feature_vector

        return feature_sequence