import numpy as np

class FeatureExtractor:
    """
    Simplified Feature Extractor for immediate performance gain.
    
    Focuses on core PFD features with minimal computation overhead.
    """
    
    def __init__(self, config):
        self.config = config
        self.h = config['data']['resolution_h']
        self.w = config['data']['resolution_w']
        
        # Simplified time window for faster processing
        self.time_window = config['feature_extractor']['pfd_time_window'] // 5  # 5ms instead of 25ms
        
        print(f"SimpleFeatureExtractor initialized: {self.h}x{self.w}, time_window={self.time_window}μs")
    
    def process_sequence(self, raw_events):
        """
        Simplified processing with minimal PFD features for speed.
        
        Args:
            raw_events (np.array): Shape [N, 4] in format [x, y, t, p]
            
        Returns:
            feature_sequence (np.array): Shape [N, 6] - simplified features
        """
        if len(raw_events) == 0:
            return np.empty((0, 6))
            
        num_events = raw_events.shape[0]
        feature_sequence = np.zeros((num_events, 6), dtype=np.float32)
        
        # Simplified state tracking
        last_polarity = np.zeros((self.h, self.w), dtype=np.int8)
        last_time = np.zeros((self.h, self.w), dtype=np.float64)
        polarity_changes = np.zeros((self.h, self.w), dtype=np.int16)
        
        for i in range(num_events):
            x, y, t, p = raw_events[i]
            ix, iy = int(x), int(y)
            
            # Bounds check
            if not (0 <= ix < self.w and 0 <= iy < self.h):
                feature_sequence[i] = np.zeros(6, dtype=np.float32)
                continue
            
            # Update polarity change count (simplified)
            if last_polarity[iy, ix] != 0 and last_polarity[iy, ix] * p == -1:
                polarity_changes[iy, ix] += 1
            
            # Compute simple neighborhood activity (3x3)
            y_start, y_end = max(0, iy-1), min(self.h, iy+2)
            x_start, x_end = max(0, ix-1), min(self.w, ix+2)
            
            # Count active neighbors in time window
            neighbor_activity = 0
            for ny in range(y_start, y_end):
                for nx in range(x_start, x_end):
                    if ny != iy or nx != ix:
                        if t - last_time[ny, nx] <= self.time_window:
                            neighbor_activity += 1
            
            # Update state
            last_polarity[iy, ix] = p
            last_time[iy, ix] = t
            
            # Create simplified 6D feature vector
            feature_vector = np.array([
                (x - self.w/2) / (self.w/2),                    # 0: Normalized x
                (y - self.h/2) / (self.h/2),                    # 1: Normalized y  
                p,                                              # 2: Polarity
                np.clip(np.log(max(t - last_time[iy, ix], 1) + 1e-6), 0, 15),  # 3: Log time interval
                np.clip(polarity_changes[iy, ix], 0, 20),       # 4: Polarity changes
                np.clip(neighbor_activity, 0, 8),               # 5: Neighbor activity
            ], dtype=np.float32)
            
            feature_sequence[i] = feature_vector
        
        return feature_sequence