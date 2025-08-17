#!/usr/bin/env python3
"""
H5StreamReader for EventMamba-FX
A robust reader for streaming large H5 event files in blocks to prevent memory exhaustion.
"""
import h5py
import hdf5plugin
import numpy as np
from typing import Generator, Tuple, Optional

class H5StreamReader:
    """
    A robust reader for streaming large H5 event files in blocks
    to prevent memory exhaustion.
    """
    def __init__(self, h5_file_path: str, block_size_events: int = 5_000_000, time_limit_us: Optional[int] = None):
        """
        Args:
            h5_file_path (str): Path to the input H5 event file.
            block_size_events (int): The number of events to read in each block.
                                     Adjust based on available RAM.
            time_limit_us (Optional[int]): If provided, only process events within this time limit (microseconds).
        """
        self.h5_file_path = h5_file_path
        self.block_size = block_size_events
        self.time_limit_us = time_limit_us
        
        with h5py.File(self.h5_file_path, 'r') as f:
            # Check file format - DSEC vs Generated Training Data
            if 'events/t' in f:
                self.file_format = 'dsec'
            elif 'features' in f and 'labels' in f:
                self.file_format = 'generated'
                print("⚠️  Warning: This file contains pre-extracted features, not raw events.")
                print("   Generated training files cannot be used for inference as they lack raw event data.")
                raise ValueError("Generated training H5 files are not supported for inference. Please use DSEC format raw event files.")
            else:
                raise ValueError("Input H5 file must contain either 'events/t' (DSEC format) or 'features' (generated format).")
            
            if self.time_limit_us is not None:
                # Find first event timestamp
                first_timestamp = f['events/t'][0]
                cutoff_timestamp = first_timestamp + self.time_limit_us
                
                # Binary search to find cutoff index
                timestamps = f['events/t']
                cutoff_idx = self._find_cutoff_index(timestamps, cutoff_timestamp)
                self.total_events = cutoff_idx
                print(f"  - Time limit: {self.time_limit_us/1e6:.1f} seconds")
                print(f"  - First timestamp: {first_timestamp}")
                print(f"  - Cutoff timestamp: {cutoff_timestamp}")
            else:
                self.total_events = len(f['events/t'])
                
            self.num_blocks = (self.total_events + self.block_size - 1) // self.block_size

        print(f"H5StreamReader Initialized:")
        print(f"  - File: {h5_file_path}")
        print(f"  - Total Events: {self.total_events:,}")
        print(f"  - Block Size: {self.block_size:,} events")
        print(f"  - Total Blocks to Process: {self.num_blocks}")
    
    def _find_cutoff_index(self, timestamps, cutoff_timestamp):
        """Binary search to find the index where timestamp exceeds cutoff."""
        left, right = 0, len(timestamps) - 1
        result = len(timestamps)  # default to all events if no cutoff found
        
        while left <= right:
            mid = (left + right) // 2
            if timestamps[mid] <= cutoff_timestamp:
                left = mid + 1
            else:
                result = mid
                right = mid - 1
        
        return result

    def stream_blocks(self) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        A generator that yields one block of events at a time as a NumPy array.
        Yields:
            - A tuple of (events_block, start_index, end_index).
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            events_group = f['events']
            for i in range(0, self.total_events, self.block_size):
                start_idx = i
                end_idx = min(i + self.block_size, self.total_events)
                
                # Read only the necessary slice from the H5 file
                x = events_group['x'][start_idx:end_idx]
                y = events_group['y'][start_idx:end_idx]
                t = events_group['t'][start_idx:end_idx]
                p = events_group['p'][start_idx:end_idx]

                p = np.where(p > 0, 1, -1)

                # Stack and yield. Use float64 for precision.
                block = np.column_stack([x, y, t, p]).astype(np.float64)
                yield block, start_idx, end_idx