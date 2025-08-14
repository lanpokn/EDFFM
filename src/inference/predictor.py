#!/usr/bin/env python3
"""
Robust Predictor for EventMamba-FX
Core inference logic that processes H5 files in blocks to prevent OOM, denoises them, and saves the result.
"""
import torch
import numpy as np
import h5py
import hdf5plugin
import os
from tqdm import tqdm
import gc  # For garbage collection
import time

from .h5_stream_reader import H5StreamReader
from ..feature_extractor import FeatureExtractor

class Predictor:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        
        # Use larger chunk size for inference since no backpropagation is needed
        self.chunk_size = config['training']['chunk_size'] * 10
        self.denoise_threshold = config['inference']['denoise_threshold']
        self.block_size_events = config['inference']['block_size_events']
        
        self.feature_extractor = FeatureExtractor(config)
        print("Robust Predictor Initialized.")
        print(f" - Denoising Threshold: {self.denoise_threshold}")
        print(f" - Model Chunk Size (for GPU): {self.chunk_size:,}")
        print(f" - H5 Block Size (for RAM): {self.block_size_events:,}")

    def denoise_file(self, input_h5_path: str, output_h5_path: str, time_limit_us: int = None):
        """
        Processes a single H5 file using a streaming approach to handle large files.
        
        Args:
            input_h5_path: Path to input H5 file
            output_h5_path: Path to output H5 file
            time_limit_us: If provided, only process events within this time limit (microseconds)
        """
        print(f"\n🚀 Starting robust denoising for: {input_h5_path}")
        
        stream_reader = H5StreamReader(input_h5_path, self.block_size_events, time_limit_us)
        
        # Prepare the output H5 file for writing
        output_dir = os.path.dirname(output_h5_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # --- File-level statistics ---
        total_events_processed = 0
        total_events_kept = 0
        
        with h5py.File(output_h5_path, 'w') as out_f:
            # Create resizable datasets in the output file
            # This allows us to append data block by block
            out_f.create_dataset('events/x', (0,), maxshape=(None,), dtype=np.uint16, compression=hdf5plugin.FILTERS['blosc'])
            out_f.create_dataset('events/y', (0,), maxshape=(None,), dtype=np.uint16, compression=hdf5plugin.FILTERS['blosc'])
            out_f.create_dataset('events/t', (0,), maxshape=(None,), dtype=np.int64, compression=hdf5plugin.FILTERS['blosc'])
            out_f.create_dataset('events/p', (0,), maxshape=(None,), dtype=np.bool_, compression=hdf5plugin.FILTERS['blosc'])
            
            # Reset model's hidden state ONCE for the entire file
            self.model.reset_hidden_state()

            # --- Main Processing Loop: Iterate through large blocks from the H5 file ---
            block_pbar = tqdm(total=stream_reader.total_events, desc="Processing H5 Blocks")
            total_feature_time = 0
            total_inference_time = 0
            
            for block_idx, (raw_events_block, start_idx, end_idx) in enumerate(stream_reader.stream_blocks()):
                block_start_time = time.time()
                
                # 1. Feature Extraction (on CPU)
                # This is a memory-heavy step, so we do it per block
                feature_start = time.time()
                features_block = self.feature_extractor.process_sequence(raw_events_block)
                features_tensor = torch.from_numpy(features_block)
                feature_time = time.time() - feature_start
                total_feature_time += feature_time
                
                block_predictions = []
                
                # 2. Inference Loop: Iterate through small chunks within the block
                inference_start = time.time()
                with torch.no_grad():
                    for i in range(0, features_tensor.shape[0], self.chunk_size):
                        # Move only the small chunk to GPU
                        chunk_features = features_tensor[i : i + self.chunk_size].to(self.device)
                        chunk_features = chunk_features.unsqueeze(0)  # Add batch dim
                        
                        predictions_logits = self.model(chunk_features)
                        predictions_probs = torch.sigmoid(predictions_logits).squeeze().cpu().numpy()
                        
                        # Handle single-element case
                        if predictions_probs.ndim == 0:
                            predictions_probs = [predictions_probs.item()]
                        
                        block_predictions.extend(predictions_probs)

                        # Clean up GPU memory
                        del chunk_features
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                
                inference_time = time.time() - inference_start
                total_inference_time += inference_time

                # 3. Filter and Save the processed block
                predictions_array = np.array(block_predictions)
                keep_mask = predictions_array < self.denoise_threshold
                
                clean_events_in_block = raw_events_block[keep_mask]
                
                if clean_events_in_block.shape[0] > 0:
                    self._append_to_h5(out_f, clean_events_in_block)
                
                # 4. Update stats and clean up RAM
                total_events_processed += len(raw_events_block)
                total_events_kept += len(clean_events_in_block)
                block_pbar.update(len(raw_events_block))

                # Print timing info every 5 blocks
                if block_idx % 5 == 0 and block_idx > 0:
                    avg_feature_time = total_feature_time / (block_idx + 1)
                    avg_inference_time = total_inference_time / (block_idx + 1)
                    total_time = time.time() - block_start_time
                    print(f"\n  Block {block_idx+1}: Feature={feature_time:.2f}s, Inference={inference_time:.2f}s, Total={total_time:.2f}s")
                    print(f"  Avg times: Feature={avg_feature_time:.2f}s, Inference={avg_inference_time:.2f}s")

                del raw_events_block, features_block, features_tensor, block_predictions, predictions_array, keep_mask
                gc.collect()  # Force garbage collection

            block_pbar.close()
            
            # Print final timing statistics
            print(f"\n⏱️  Performance Summary:")
            print(f"  Total feature extraction time: {total_feature_time:.2f}s")
            print(f"  Total inference time: {total_inference_time:.2f}s")
            print(f"  Feature/Inference ratio: {total_feature_time/total_inference_time:.2f}x")

        # --- Final Summary ---
        percent_removed = ((total_events_processed - total_events_kept) / total_events_processed * 100) if total_events_processed > 0 else 0
        print("\n--- ✅ Denoising Complete ---")
        print(f"Total Original Events: {total_events_processed:,}")
        print(f"Total Clean Events:    {total_events_kept:,}")
        print(f"Total Removed Events:  {total_events_processed - total_events_kept:,} ({percent_removed:.2f}%)")
        print(f"Clean file saved to:   {output_h5_path}")

    def _append_to_h5(self, h5_file_handle, events_to_append: np.ndarray):
        """Appends a block of events to the resizable datasets in an open H5 file."""
        num_to_append = events_to_append.shape[0]
        
        datasets = {
            'x': h5_file_handle['events/x'],
            'y': h5_file_handle['events/y'],
            't': h5_file_handle['events/t'],
            'p': h5_file_handle['events/p'],
        }

        # Resize all datasets
        for ds in datasets.values():
            ds.resize(ds.shape[0] + num_to_append, axis=0)
            
        # Append data
        current_size = datasets['x'].shape[0] - num_to_append
        datasets['x'][current_size:] = events_to_append[:, 0].astype(np.uint16)
        datasets['y'][current_size:] = events_to_append[:, 1].astype(np.uint16)
        datasets['t'][current_size:] = events_to_append[:, 2].astype(np.int64)
        datasets['p'][current_size:] = np.where(events_to_append[:, 3] > 0, True, False)