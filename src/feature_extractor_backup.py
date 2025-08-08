import numpy as np
from collections import deque

class FeatureExtractor:
    """
    Optimized Feature Extractor based on PFDs.cpp implementation.
    
    Key improvements:
    - O(N) time complexity (vs O(N²) original)
    - Fixed-size circular buffers (vs growing lists) 
    - Local time window features (vs global cumulative)
    - Train/test consistency (bounded feature ranges)
    """
    
    def __init__(self, config):
        # Configuration parameters
        self.config = config
        self.h = config['data']['resolution_h']
        self.w = config['data']['resolution_w']
        
        # PFD parameters - matching PFDs.cpp
        self.time_window = config['feature_extractor']['pfd_time_window']  # 25ms (25000μs)
        self.max_polarity_changes = 5  # fifoSize from PFDs.cpp - fixed buffer size
        
        print(f"FeatureExtractor initialized: {self.h}x{self.w}, time_window={self.time_window}μs")
    
    def process_sequence(self, raw_events):
        """
        Process event sequence with O(N) complexity using PFDs.cpp approach.
        
        Args:
            raw_events (np.array): Shape [N, 4] in format [x, y, t, p]
            
        Returns:
            feature_sequence (np.array): Shape [N, 10] - optimized feature vectors
        """
        if len(raw_events) == 0:
            return np.empty((0, 10))
            
        num_events = raw_events.shape[0]
        feature_sequence = np.zeros((num_events, 10), dtype=np.float32)
        
        # Initialize pixel-wise state maps (matching PFDs.cpp)
        latest_timestamp = np.zeros((self.h, self.w), dtype=np.float64)
        latest_polarity = np.zeros((self.h, self.w), dtype=np.int8)
        
        # Fixed-size circular buffers using numpy arrays (faster than deque)
        polarity_change_times = np.zeros((self.h, self.w, self.max_polarity_changes), dtype=np.float64)
        polarity_change_counts = np.zeros((self.h, self.w), dtype=np.int8)
        polarity_change_heads = np.zeros((self.h, self.w), dtype=np.int8)
        
        # Process each event - O(N) main loop
        for i in range(num_events):
            x, y, t, p = raw_events[i]
            ix, iy = int(x), int(y)
            
            # Bounds checking
            if not (0 <= ix < self.w and 0 <= iy < self.h):
                # Invalid coordinates - use zero feature vector
                feature_sequence[i] = np.zeros(10, dtype=np.float32)
                continue
            
            # === Update pixel state (matching PFDs.cpp lines 85-94) ===
            
            # Check for polarity change and record timestamp (using circular buffer)
            if latest_polarity[iy, ix] != 0:
                if latest_polarity[iy, ix] * p == -1:  # Polarity changed
                    head = polarity_change_heads[iy, ix]
                    polarity_change_times[iy, ix, head] = t
                    polarity_change_heads[iy, ix] = (head + 1) % self.max_polarity_changes
                    polarity_change_counts[iy, ix] = min(polarity_change_counts[iy, ix] + 1, 
                                                        self.max_polarity_changes)
            
            # Update pixel state
            latest_polarity[iy, ix] = p
            latest_timestamp[iy, ix] = t
            
            # === Compute PFD features (matching PFDs.cpp lines 141-174) ===
            
            # 1. Current pixel polarity frequency (Mf) in time window
            count = polarity_change_counts[iy, ix]
            current_polarity_changes = 0
            if count > 0:
                pixel_times = polarity_change_times[iy, ix, :count]
                current_polarity_changes = np.sum(t - pixel_times <= self.time_window)
            
            # 2. Neighborhood features in time window (optimized)
            neighbor_polarity_changes = 0
            active_neighbors = 0
            
            # 3x3 neighborhood scan with bounds pre-checking
            y_start = max(0, iy - 1)
            y_end = min(self.h, iy + 2)
            x_start = max(0, ix - 1)
            x_end = min(self.w, ix + 2)
            
            for ny in range(y_start, y_end):
                for nx in range(x_start, x_end):
                    if ny == iy and nx == ix:
                        continue  # Skip center pixel
                    
                    # Check if neighbor is active in time window
                    if t - latest_timestamp[ny, nx] <= self.time_window:
                        active_neighbors += 1
                    
                    # Count neighbor polarity changes in time window
                    neighbor_count = polarity_change_counts[ny, nx]
                    if neighbor_count > 0:
                        neighbor_times = polarity_change_times[ny, nx, :neighbor_count]
                        neighbor_polarity_changes += np.sum(t - neighbor_times <= self.time_window)
            
            # 3. Compute PFD scores
            density = neighbor_polarity_changes / active_neighbors if active_neighbors > 0 else 0
            pfd_a_score = abs(current_polarity_changes - density)  # PFD-A (BA noise detection)
            pfd_b_score = density  # PFD-B (flicker detection)
            
            # === Compute additional local features ===
            
            # Normalized spatial coordinates  
            x_norm = (x - self.w/2) / (self.w/2)
            y_norm = (y - self.h/2) / (self.h/2)
            
            # Time since last event at this pixel (optimized)
            prev_pixel_time = latest_timestamp[iy, ix]
            if prev_pixel_time > 0:
                dt_pixel_log = np.log(max(t - prev_pixel_time, 1) + 1e-6)
            else:
                dt_pixel_log = 0
            
            # === Create optimized 10D feature vector ===
            # All features are bounded and time-window local (no global accumulation)
            feature_vector = np.array([
                np.clip(x_norm, -1, 1),                           # 0: Normalized x coord [-1,1]
                np.clip(y_norm, -1, 1),                           # 1: Normalized y coord [-1,1]  
                p,                                                # 2: Polarity {-1, 1}
                np.clip(dt_pixel_log, 0, 15),                     # 3: Log pixel time interval [0,15]
                np.clip(current_polarity_changes, 0, 10),         # 4: Local polarity frequency (Mf) [0,10]
                np.clip(neighbor_polarity_changes, 0, 50),        # 5: Neighbor changes (Ma) [0,50]  
                np.clip(active_neighbors, 0, 8),                  # 6: Active neighbors (Ne) [0,8]
                np.clip(density, 0, 10),                          # 7: Polarity density (D) [0,10]
                np.clip(pfd_a_score, 0, 20),                      # 8: PFD-A score [0,20]
                np.clip(pfd_b_score, 0, 10),                      # 9: PFD-B score [0,10]
            ], dtype=np.float32)
            
            feature_sequence[i] = feature_vector
        
        return feature_sequence