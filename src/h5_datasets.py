import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import hdf5plugin
import os
from .feature_extractor import FeatureExtractor

class EventDenoisingH5Dataset(Dataset):
    """
    H5-based Event Denoising Dataset using DSEC format.
    
    DSEC H5 Format:
    - events/x: uint16 array of x coordinates
    - events/y: uint16 array of y coordinates  
    - events/t: uint32 array of timestamps (microseconds)
    - events/p: uint8 array of polarities (0/1)
    - t_offset: int64 scalar (can be ignored for our use)
    - ms_to_idx: uint64 array (can be ignored for our use)
    
    Additional for training:
    - events/labels: uint8 array of labels (1=clean, 0=flare)
    """
    
    def __init__(self, file_path, config):
        super().__init__()
        self.file_path = file_path
        self.sequence_length = config['data']['sequence_length']
        self.config = config
        
        # Load H5 data
        self._load_h5_data(file_path, config)
        
        # Create feature extractor with fixed resolution from config
        self.feature_extractor = FeatureExtractor(config)
        
    def _load_h5_data(self, file_path, config):
        """Load H5 data and extract resolution information."""
        print(f"Loading H5 data from {file_path}...")
        with h5py.File(file_path, 'r') as f:
            # Load event data
            self.events_x = f['events/x'][:]
            self.events_y = f['events/y'][:]  
            self.events_t = f['events/t'][:]
            self.events_p = f['events/p'][:]
            
            # Load labels if available
            if 'events/labels' in f:
                self.labels = f['events/labels'][:]
                print(f"Found labels: {np.sum(self.labels==1):,} clean, {np.sum(self.labels==0):,} flare events")
            else:
                # Default to all clean events if no labels
                self.labels = np.ones(len(self.events_x), dtype=np.uint8)
                print("No labels found, assuming all events are clean")
                
            # Get resolution from attributes for logging only
            if 'events' in f and 'resolution_height' in f['events'].attrs:
                file_resolution_h = int(f['events'].attrs['resolution_height'])
                file_resolution_w = int(f['events'].attrs['resolution_width'])
                print(f"H5 file resolution: {file_resolution_h} x {file_resolution_w}")
            else:
                print("No resolution metadata found in H5 file")
                
        print(f"Loaded {len(self.events_x):,} events from {file_path}")
        print(f"Using config resolution: {config['data']['resolution_h']} x {config['data']['resolution_w']}")
        print(f"Time range: {self.events_t.min()} - {self.events_t.max()} μs")
        
        # Convert polarity format: DSEC uses 0/1, our model expects -1/1
        self.events_p_converted = np.where(self.events_p == 0, -1, 1).astype(np.int8)

    def __len__(self):
        return len(self.events_x) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Extract sequence of raw events
        end_idx = idx + self.sequence_length
        
        # Create raw events array in [x, y, t, p] format
        raw_events = np.column_stack([
            self.events_x[idx:end_idx].astype(np.float32),
            self.events_y[idx:end_idx].astype(np.float32), 
            self.events_t[idx:end_idx].astype(np.float32),
            self.events_p_converted[idx:end_idx].astype(np.float32)
        ])
        
        # Extract corresponding labels
        sequence_labels = self.labels[idx:end_idx].astype(np.float32)
        
        # Apply feature extraction
        feature_sequence = self.feature_extractor.process_sequence(raw_events)
        
        # Convert to tensors
        features_tensor = torch.tensor(feature_sequence, dtype=torch.float32)
        labels_tensor = torch.tensor(sequence_labels, dtype=torch.float32).unsqueeze(-1)
        
        return features_tensor, labels_tensor


def create_h5_dataloaders(config):
    """
    Creates train, validation, and test dataloaders for H5 format.
    """
    # Update paths to use .h5 extension
    train_path = config['data']['train_path'].replace('.txt', '.h5')
    val_path = config['data']['val_path'].replace('.txt', '.h5')
    test_path = config['data']['test_path'].replace('.txt', '.h5')
    
    train_dataset = EventDenoisingH5Dataset(train_path, config)
    val_dataset = EventDenoisingH5Dataset(val_path, config)
    test_dataset = EventDenoisingH5Dataset(test_path, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader, test_loader


def convert_txt_to_h5_dsec_format(txt_path, h5_path, resolution=(260, 346)):
    """
    Convert TXT event data to DSEC H5 format.
    
    Args:
        txt_path: Path to TXT file in format [x, y, t, p, label]
        h5_path: Output H5 file path
        resolution: (height, width) tuple
    """
    print(f"Converting {txt_path} to DSEC H5 format...")
    
    # Load TXT data
    data = np.loadtxt(txt_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
        
    # Extract event data
    events_x = data[:, 0].astype(np.uint16)
    events_y = data[:, 1].astype(np.uint16)
    events_t = data[:, 2].astype(np.uint32) 
    events_p_raw = data[:, 3].astype(np.int8)
    
    # Convert polarity: -1/1 -> 0/1 for DSEC format
    events_p = np.where(events_p_raw == -1, 0, 1).astype(np.uint8)
    
    # Extract labels if available
    if data.shape[1] >= 5:
        labels = data[:, 4].astype(np.uint8)
    else:
        labels = np.ones(len(events_x), dtype=np.uint8)  # Default to clean
        
    print(f"Processing {len(events_x):,} events...")
    print(f"Time range: {events_t.min()} - {events_t.max()} μs")
    
    # Generate ms_to_idx lookup table
    if len(events_t) > 0:
        last_timestamp_us = events_t[-1]
        total_duration_ms = int(np.ceil(last_timestamp_us / 1000.0))
        ms_timestamps_us = np.arange(total_duration_ms) * 1000
        ms_to_idx = np.searchsorted(events_t, ms_timestamps_us, side='left').astype(np.uint64)
        print(f"Generated ms_to_idx table with {len(ms_to_idx)} entries")
    else:
        ms_to_idx = np.array([], dtype=np.uint64)
    
    # Save to H5 format
    with h5py.File(h5_path, 'w') as f:
        # Create events group
        events_group = f.create_group('events')
        events_group.create_dataset('x', data=events_x, compression='gzip', compression_opts=9)
        events_group.create_dataset('y', data=events_y, compression='gzip', compression_opts=9)
        events_group.create_dataset('t', data=events_t, compression='gzip', compression_opts=9)
        events_group.create_dataset('p', data=events_p, compression='gzip', compression_opts=9)
        events_group.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        
        # Add resolution metadata
        events_group.attrs['resolution_height'] = resolution[0]
        events_group.attrs['resolution_width'] = resolution[1]
        events_group.attrs['num_events'] = len(events_x)
        
        # DSEC format requirements
        f.create_dataset('t_offset', data=np.int64(0))  # Not used in our case
        f.create_dataset('ms_to_idx', data=ms_to_idx, compression='gzip', compression_opts=9)
        
    # Check compression ratio
    txt_size = os.path.getsize(txt_path)
    h5_size = os.path.getsize(h5_path)
    compression_ratio = txt_size / h5_size
    
    print(f"✓ Conversion complete!")
    print(f"  TXT size: {txt_size:,} bytes ({txt_size/1024/1024:.2f} MB)")
    print(f"  H5 size:  {h5_size:,} bytes ({h5_size/1024/1024:.2f} MB)")
    print(f"  Compression: {compression_ratio:.2f}x smaller")


if __name__ == "__main__":
    # Test conversion
    import sys
    if len(sys.argv) > 1:
        txt_file = sys.argv[1]
        h5_file = txt_file.replace('.txt', '.h5')
        convert_txt_to_h5_dsec_format(txt_file, h5_file)