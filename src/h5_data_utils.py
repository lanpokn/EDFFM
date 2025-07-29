"""
H5-based Event Data Processing Utilities
========================================

This module provides utilities for handling event data in H5 format,
supporting multiple resolutions and efficient data operations.

Key Features:
- Resolution-independent event data storage
- Efficient H5 format I/O with compression
- Event data combining (original + flare)  
- Metadata management for different datasets
- Compatible with existing feature extraction pipeline

Author: Generated with Claude Code
"""

import h5py
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
import hdf5plugin  # Required for compressed H5 files


class H5EventData:
    """
    H5-based event data container with metadata support.
    """
    
    def __init__(self, h5_path: str, mode: str = 'r'):
        """
        Initialize H5 event data container.
        
        Args:
            h5_path: Path to H5 file
            mode: File access mode ('r', 'w', 'a')
        """
        self.h5_path = h5_path
        self.mode = mode
        self._file = None
        
    def __enter__(self):
        self._file = h5py.File(self.h5_path, self.mode)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            
    def save_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                   resolution: Tuple[int, int], dataset_name: str = "events",
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Save event data to H5 format with compression.
        
        Args:
            x, y, t, p: Event data arrays
            resolution: (height, width) of the sensor
            dataset_name: Name of the dataset (e.g., "original", "flare", "mixed")
            metadata: Additional metadata to store
        """
        if self.mode == 'r':
            raise ValueError("Cannot save in read-only mode")
            
        # Create group for this dataset
        if dataset_name in self._file:
            del self._file[dataset_name]
        group = self._file.create_group(dataset_name)
        
        # Save event data with compression
        group.create_dataset('x', data=x.astype(np.uint16), 
                           compression='gzip', compression_opts=9)
        group.create_dataset('y', data=y.astype(np.uint16),
                           compression='gzip', compression_opts=9)
        group.create_dataset('t', data=t.astype(np.uint64),
                           compression='gzip', compression_opts=9)
        group.create_dataset('p', data=p.astype(np.uint8),
                           compression='gzip', compression_opts=9)
        
        # Save metadata
        group.attrs['resolution_height'] = resolution[0]
        group.attrs['resolution_width'] = resolution[1]
        group.attrs['num_events'] = len(x)
        group.attrs['time_start'] = int(t.min()) if len(t) > 0 else 0
        group.attrs['time_end'] = int(t.max()) if len(t) > 0 else 0
        group.attrs['polarity_ratio'] = float(np.mean(p)) if len(p) > 0 else 0.5
        
        if metadata:
            for key, value in metadata.items():
                group.attrs[key] = value
                
    def load_events(self, dataset_name: str = "events") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load event data from H5 format.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            x, y, t, p arrays and metadata dict
        """
        if dataset_name not in self._file:
            raise KeyError(f"Dataset '{dataset_name}' not found in {self.h5_path}")
            
        group = self._file[dataset_name]
        
        # Load event data
        x = group['x'][:]
        y = group['y'][:]
        t = group['t'][:]
        p = group['p'][:]
        
        # Load metadata
        metadata = dict(group.attrs)
        
        return x, y, t, p, metadata
        
    def list_datasets(self):
        """List all available datasets in the H5 file."""
        return list(self._file.keys())


def convert_txt_to_h5(txt_path: str, h5_path: str, resolution: Tuple[int, int],
                     dataset_name: str = "events", metadata: Optional[Dict[str, Any]] = None):
    """
    Convert TXT event data to H5 format.
    
    Args:
        txt_path: Path to input TXT file [x, y, t, p] format
        h5_path: Path to output H5 file
        resolution: (height, width) of the sensor
        dataset_name: Name for the dataset
        metadata: Additional metadata
    """
    print(f"Converting {txt_path} to H5 format...")
    
    # Load TXT data
    data = np.loadtxt(txt_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    x = data[:, 0].astype(np.uint16)
    y = data[:, 1].astype(np.uint16)
    t = data[:, 2].astype(np.uint64)
    p = data[:, 3].astype(np.uint8)
    
    print(f"Loaded {len(x):,} events from TXT file")
    
    # Save to H5
    with H5EventData(h5_path, 'w') as h5_data:
        h5_data.save_events(x, y, t, p, resolution, dataset_name, metadata)
        
    # Check compression ratio
    txt_size = os.path.getsize(txt_path)
    h5_size = os.path.getsize(h5_path)
    compression_ratio = txt_size / h5_size
    
    print(f"✓ Conversion complete!")
    print(f"  TXT size: {txt_size:,} bytes ({txt_size/1024/1024:.2f} MB)")
    print(f"  H5 size:  {h5_size:,} bytes ({h5_size/1024/1024:.2f} MB)")
    print(f"  Compression: {compression_ratio:.2f}x smaller")


def combine_h5_events(original_h5: str, flare_h5: str, output_h5: str,
                     original_dataset: str = "events", flare_dataset: str = "events"):
    """
    Combine original and flare events into a mixed dataset.
    
    Args:
        original_h5: Path to original events H5 file
        flare_h5: Path to flare events H5 file  
        output_h5: Path to output mixed events H5 file
        original_dataset: Dataset name in original H5
        flare_dataset: Dataset name in flare H5
    """
    print(f"Combining events from {original_h5} and {flare_h5}...")
    
    # Load original events
    with H5EventData(original_h5, 'r') as h5_orig:
        x_orig, y_orig, t_orig, p_orig, meta_orig = h5_orig.load_events(original_dataset)
        
    # Load flare events
    with H5EventData(flare_h5, 'r') as h5_flare:
        x_flare, y_flare, t_flare, p_flare, meta_flare = h5_flare.load_events(flare_dataset)
        
    print(f"Original events: {len(x_orig):,}")
    print(f"Flare events: {len(x_flare):,}")
    
    # Combine events
    x_mixed = np.concatenate([x_orig, x_flare])
    y_mixed = np.concatenate([y_orig, y_flare])
    t_mixed = np.concatenate([t_orig, t_flare])
    p_mixed = np.concatenate([p_orig, p_flare])
    
    # Create labels (1=original/clean, 0=flare/noise)
    labels = np.concatenate([
        np.ones(len(x_orig), dtype=np.uint8),    # Original events = 1 (clean)
        np.zeros(len(x_flare), dtype=np.uint8)   # Flare events = 0 (noise)
    ])
    
    # Sort by timestamp
    sort_idx = np.argsort(t_mixed)
    x_mixed = x_mixed[sort_idx]
    y_mixed = y_mixed[sort_idx]
    t_mixed = t_mixed[sort_idx]
    p_mixed = p_mixed[sort_idx]
    labels = labels[sort_idx]
    
    print(f"Mixed events: {len(x_mixed):,} (sorted by timestamp)")
    
    # Save mixed events with labels
    resolution = (meta_orig['resolution_height'], meta_orig['resolution_width'])
    metadata = {
        'original_events': len(x_orig),
        'flare_events': len(x_flare),
        'total_events': len(x_mixed),
        'clean_ratio': len(x_orig) / len(x_mixed),
        'flare_ratio': len(x_flare) / len(x_mixed),
        'source_original': original_h5,
        'source_flare': flare_h5
    }
    
    with H5EventData(output_h5, 'w') as h5_mixed:
        # Save event data
        h5_mixed.save_events(x_mixed, y_mixed, t_mixed, p_mixed, 
                           resolution, "events", metadata)
        
        # Save labels separately
        group = h5_mixed._file["events"]
        group.create_dataset('labels', data=labels, 
                           compression='gzip', compression_opts=9)
        
    print(f"✓ Mixed events saved to {output_h5}")
    print(f"  Clean events: {len(x_orig):,} ({len(x_orig)/len(x_mixed)*100:.1f}%)")
    print(f"  Flare events: {len(x_flare):,} ({len(x_flare)/len(x_mixed)*100:.1f}%)")


def load_h5_for_training(h5_path: str, dataset_name: str = "events") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load H5 event data in format compatible with existing training pipeline.
    
    Args:
        h5_path: Path to H5 file
        dataset_name: Dataset name to load
        
    Returns:
        events: [N, 4] array in [x, y, t, p] format
        labels: [N] array of labels (if available)
    """
    with H5EventData(h5_path, 'r') as h5_data:
        x, y, t, p, metadata = h5_data.load_events(dataset_name)
        
        # Convert to standard format [x, y, t, p]
        events = np.column_stack([x, y, t, p])
        
        # Load labels if available
        group = h5_data._file[dataset_name]
        if 'labels' in group:
            labels = group['labels'][:]
        else:
            # Default to all clean events if no labels
            labels = np.ones(len(events), dtype=np.uint8)
            
    return events, labels


def analyze_h5_file(h5_path: str):
    """
    Analyze and display information about an H5 event file.
    
    Args:
        h5_path: Path to H5 file to analyze
    """
    print(f"Analyzing H5 file: {h5_path}")
    print("=" * 50)
    
    file_size = os.path.getsize(h5_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    with H5EventData(h5_path, 'r') as h5_data:
        datasets = h5_data.list_datasets()
        print(f"Datasets: {datasets}")
        
        for dataset_name in datasets:
            print(f"\n--- Dataset: {dataset_name} ---")
            try:
                x, y, t, p, metadata = h5_data.load_events(dataset_name)
                
                print(f"Events: {len(x):,}")
                print(f"Resolution: {metadata.get('resolution_height', 'unknown')} x {metadata.get('resolution_width', 'unknown')}")
                print(f"Time range: {metadata.get('time_start', 'unknown')} - {metadata.get('time_end', 'unknown')} μs")
                print(f"Spatial range: x=[{x.min()}, {x.max()}], y=[{y.min()}, {y.max()}]")
                print(f"Polarity ratio: {metadata.get('polarity_ratio', 'unknown'):.3f}")
                
                # Check for labels
                group = h5_data._file[dataset_name]
                if 'labels' in group:
                    labels = group['labels'][:]
                    clean_events = np.sum(labels == 1)
                    flare_events = np.sum(labels == 0)
                    print(f"Labels: {clean_events:,} clean ({clean_events/len(labels)*100:.1f}%), {flare_events:,} flare ({flare_events/len(labels)*100:.1f}%)")
                    
                # Show additional metadata
                print("Metadata:")
                for key, value in metadata.items():
                    if key not in ['resolution_height', 'resolution_width', 'time_start', 'time_end', 'polarity_ratio']:
                        print(f"  {key}: {value}")
                        
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")


if __name__ == "__main__":
    # Example usage
    print("H5 Event Data Utilities")
    print("Usage examples:")
    print("1. convert_txt_to_h5('events.txt', 'events.h5', (480, 640))")
    print("2. combine_h5_events('original.h5', 'flare.h5', 'mixed.h5')")
    print("3. analyze_h5_file('data.h5')")