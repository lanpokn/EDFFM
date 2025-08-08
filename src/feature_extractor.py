import numpy as np

class FeatureExtractor:
    """
    Ultra-Fast 4D Feature Extractor - temporarily disabling PFD for speed.
    
    Processes events with minimal computation: x,y normalization and dt calculation.
    """
    
    def __init__(self, config):
        self.config = config
        self.h = config['data']['resolution_h']
        self.w = config['data']['resolution_w']
        
        print(f"FastFeatureExtractor initialized: {self.h}x{self.w}, 4D features (PFD disabled)")
    
    def process_sequence(self, raw_events):
        """
        Ultra-fast processing with only 4D features: [x_norm, y_norm, dt, p]
        
        Args:
            raw_events (np.array): Shape [N, 4] in format [x, y, t, p]
            
        Returns:
            feature_sequence (np.array): Shape [N, 4] - fast normalized features
        """
        if len(raw_events) == 0:
            return np.empty((0, 4))
            
        num_events = raw_events.shape[0]
        feature_sequence = np.zeros((num_events, 4), dtype=np.float32)
        
        # Extract coordinates, timestamps, and polarities
        x = raw_events[:, 0].astype(np.float32)
        y = raw_events[:, 1].astype(np.float32)  
        t = raw_events[:, 2].astype(np.float64)
        p = raw_events[:, 3].astype(np.float32)
        
        # 1. Normalize x, y coordinates to [0, 1]
        x_norm = x / (self.w - 1)
        y_norm = y / (self.h - 1)
        
        # 2. Calculate dt (time difference with previous event)
        dt = np.zeros_like(t, dtype=np.float32)
        dt[1:] = (t[1:] - t[:-1]).astype(np.float32)
        dt[0] = 0.0  # First event has no previous event
        
        # 3. Assemble 4D features: [x_norm, y_norm, dt, p]
        feature_sequence[:, 0] = x_norm
        feature_sequence[:, 1] = y_norm  
        feature_sequence[:, 2] = dt
        feature_sequence[:, 3] = p
        
        return feature_sequence