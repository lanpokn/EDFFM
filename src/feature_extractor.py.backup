import numpy as np

class FeatureExtractor:
    def __init__(self, config):
        # 从config中获取所有需要的参数
        self.config = config
        self.h = config['data']['resolution_h']
        self.w = config['data']['resolution_w']
        self.coarse_time_window = config['feature_extractor']['coarse_time_window']
        
        # PFD parameters from config
        self.pfd_time_window = config['feature_extractor']['pfd_time_window']
        self.pfd_neighborhood_size = config['feature_extractor']['pfd_neighborhood_size']
        # ... 其他参数

    def process_sequence(self, raw_events):
        """
        Processes a single sequence of raw events and returns a feature sequence.
        Now includes PFD-inspired features for enhanced denoising capability.
        
        Args:
            raw_events (np.array): Shape: [sequence_length, 4]
            
        Returns:
            feature_sequence (np.array): Shape: [sequence_length, output_dim]
        """
        num_events = raw_events.shape[0]
        output_dim = self.config['model']['input_feature_dim']
        feature_sequence = np.zeros((num_events, output_dim))

        # Initialize traditional time maps
        p_time_map = np.zeros((self.h, self.w))
        n_time_map = np.zeros((self.h, self.w))
        
        # Initialize PFD-inspired maps (based on the paper and C++ implementation)
        polarity_map = np.zeros((self.h, self.w), dtype=int)  # Mp: latest polarity at each pixel
        event_count_map = np.zeros((self.h, self.w), dtype=int)  # Count_img: event count per pixel
        polarity_change_map = np.zeros((self.h, self.w), dtype=int)  # Count_img_pol: polarity changes
        polarity_frequency_map = np.zeros((self.h, self.w), dtype=int)  # Mf: polarity changes in time window
        
        # For time window-based calculations, store recent events info
        recent_events = []  # Store (x, y, t, p, index) for time window calculations
        
        # 遍历序列中的每一个事件
        for i in range(num_events):
            x, y, t, p = raw_events[i]
            ix, iy = int(x), int(y)  # Integer indices for maps
            
            # Spatial features: Use center-relative coordinates only
            x_center = (x - self.w/2) / (self.w/2)  # Center-relative [-1, 1]
            y_center = (y - self.h/2) / (self.h/2)  # Center-relative [-1, 1]
            
            # Time delta features
            if i > 0:
                dt = t - raw_events[i-1, 2]
                dt_norm = np.log(dt + 1e-6)  # Log-scaled delta time
            else:
                dt_norm = 0
                
            # Polarity-aware time since last event at this pixel
            if p > 0:
                dt_pixel = t - p_time_map[iy, ix] if p_time_map[iy, ix] > 0 else 0
                p_time_map[iy, ix] = t
            else:
                dt_pixel = t - n_time_map[iy, ix] if n_time_map[iy, ix] > 0 else 0
                n_time_map[iy, ix] = t
            
            dt_pixel_norm = np.log(dt_pixel + 1e-6) if dt_pixel > 0 else 0
            
            # === PFD-inspired Features ===
            
            # Update polarity change detection
            if polarity_map[iy, ix] == 0:  # First event at this pixel
                polarity_map[iy, ix] = p
            else:
                # Check for polarity change (same logic as C++ code)
                if polarity_map[iy, ix] * p == -1:  # Polarity changed
                    polarity_change_map[iy, ix] += 1
            
            polarity_map[iy, ix] = p  # Update latest polarity
            event_count_map[iy, ix] += 1  # Increment event count
            
            # Clean up old events outside time window for Mf calculation
            current_time = t
            recent_events = [(ex, ey, et, ep, ei) for (ex, ey, et, ep, ei) in recent_events 
                           if current_time - et <= self.pfd_time_window]
            recent_events.append((ix, iy, t, p, i))
            
            # Calculate Mf (polarity frequency) within time window for current pixel
            mf_current = 0
            last_polarity = None
            for (ex, ey, et, ep, ei) in recent_events:
                if ex == ix and ey == iy:  # Same pixel
                    if last_polarity is not None and last_polarity * ep == -1:
                        mf_current += 1
                    last_polarity = ep
            
            # PFD features - parameterized neighborhood size
            ma_neighborhood, ne_neighborhood, d_neighborhood = self._compute_pfd_neighborhood_features(
                ix, iy, polarity_change_map, event_count_map
            )
            
            # PFD-A score: |Mf_current - Ma_neighborhood/Ne_neighborhood|
            pfd_a_score = abs(mf_current - d_neighborhood)
            
            # PFD-B score: Ma_neighborhood/Ne_neighborhood (for flicker detection)
            pfd_b_score = d_neighborhood
            
            # Additional PFD-inspired features
            current_polarity_changes = polarity_change_map[iy, ix]
            current_event_count = event_count_map[iy, ix]
            
            # Create enhanced feature vector with PFD features
            feature_vector = np.array([
                x_center, y_center,           # 0,1: Center-relative spatial coordinates [-1,1]
                p,                            # 2: Polarity {-1, 1}
                dt_norm,                      # 3: Log-scaled delta time
                dt_pixel_norm,                # 4: Log-scaled pixel-wise delta time
                mf_current,                   # 5: Polarity frequency in time window (Mf)
                ma_neighborhood,              # 6: Neighborhood polarity changes (Ma)
                ne_neighborhood,              # 7: Active neighbors count (Ne)
                d_neighborhood,               # 8: Polarity change density D(x,y) = Ma/Ne
                pfd_a_score,                  # 9: PFD-A denoising score
                pfd_b_score,                  # 10: PFD-B flicker detection score
                current_polarity_changes,     # 11: Total polarity changes at this pixel
                current_event_count,          # 12: Total event count at this pixel
                *np.zeros(output_dim - 13)    # 13+: Placeholder for additional features
            ])
            
            feature_sequence[i] = feature_vector

        return feature_sequence
    
    def _compute_pfd_neighborhood_features(self, x, y, polarity_change_map, event_count_map):
        """
        Compute PFD neighborhood features with configurable neighborhood size.
        
        Args:
            x, y: Current pixel coordinates (int)
            polarity_change_map: Map of polarity changes per pixel
            event_count_map: Map of event counts per pixel
            
        Returns:
            tuple: (ma_neighborhood, ne_neighborhood, d_neighborhood)
        """
        if self.pfd_neighborhood_size == 1:
            # 1x1 version: only current pixel (no neighbors)
            ma_neighborhood = 0  # No neighbors for 1x1
            ne_neighborhood = 1 if event_count_map[y, x] > 0 else 1  # Always 1 for current event
            d_neighborhood = 0   # 0 for 1x1 (no neighborhood changes)
            
        else:
            # NxN version: compute neighborhood statistics
            ma_neighborhood = 0  # Sum of polarity changes in neighborhood
            ne_neighborhood = 0  # Count of active neighbors
            
            radius = self.pfd_neighborhood_size // 2
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip center pixel
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        ma_neighborhood += polarity_change_map[ny, nx]
                        if event_count_map[ny, nx] > 0:
                            ne_neighborhood += 1
            
            # Calculate density
            d_neighborhood = ma_neighborhood / ne_neighborhood if ne_neighborhood > 0 else 0
        
        return ma_neighborhood, ne_neighborhood, d_neighborhood