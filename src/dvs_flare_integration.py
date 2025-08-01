"""
DVS-Flare Integration Module for EventMamba-FX

This module integrates the flare synthesis with DVS-Voltmeter simulator to generate
realistic flare events that can be combined with background DSEC events.

Key features:
- Generate flare video sequences and convert to events using DVS simulator  
- Handle temporary file management for simulator pipeline
- Output events in format compatible with EventMamba-FX training
- Benchmark timing for each processing step
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional

# Add simulator path to import DVS-Voltmeter modules
simulator_path = "/mnt/e/2025/event_flick_flare/main/simulator/DVS-Voltmeter-main"
if simulator_path not in sys.path:
    sys.path.append(simulator_path)

# Try different import paths
try:
    from src.flare_synthesis import FlareFlickeringSynthesizer
except ImportError:
    try:
        from flare_synthesis import FlareFlickeringSynthesizer
    except ImportError:
        # Add current directory to path
        import sys
        sys.path.append(os.path.dirname(__file__))
        from flare_synthesis import FlareFlickeringSynthesizer


class DVSFlareEventGenerator:
    """Generates flare events using DVS simulator integration."""
    
    def __init__(self, config: Dict):
        """Initialize the DVS flare event generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.flare_synthesizer = FlareFlickeringSynthesizer(config)
        self.simulator_path = simulator_path
        
        # 使用DSEC分辨率而非默认DVS分辨率 (关键修复!)
        self.dvs_resolution = (
            config['data']['resolution_w'],  # 640
            config['data']['resolution_h']   # 480
        )
        print(f"DVS simulator resolution set to: {self.dvs_resolution[0]}x{self.dvs_resolution[1]}")
        
    def generate_flare_events(self, temp_dir: Optional[str] = None, 
                            cleanup: bool = True) -> Tuple[np.ndarray, Dict]:
        """Generate flare events using the complete pipeline.
        
        Args:
            temp_dir: Optional temporary directory, creates one if None
            cleanup: Whether to cleanup temporary files
            
        Returns:
            Tuple of (events_array, timing_info)
            Events format: [timestamp_us, x, y, polarity]
        """
        timing_info = {}
        total_start = time.time()
        
        # Create temporary directory if not provided
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="flare_events_")
            cleanup_temp = cleanup
        else:
            cleanup_temp = False
            
        try:
            # Step 1: Generate flickering flare video
            print("Step 1: Generating flickering flare video...")
            step1_start = time.time()
            
            video_frames, flare_metadata = self.flare_synthesizer.create_flare_event_sequence(
                target_resolution=self.dvs_resolution
            )
            
            timing_info['flare_synthesis_sec'] = time.time() - step1_start
            print(f"  Generated {len(video_frames)} frames in {timing_info['flare_synthesis_sec']:.3f}s")
            
            # Step 2: Save video frames for DVS simulator
            print("Step 2: Saving video frames for DVS simulator...")
            step2_start = time.time()
            
            sequence_dir = self._save_video_for_dvs_simulator(video_frames, temp_dir, flare_metadata)
            
            timing_info['frame_saving_sec'] = time.time() - step2_start
            print(f"  Saved frames in {timing_info['frame_saving_sec']:.3f}s")
            
            # Step 3: Run DVS simulator
            print("Step 3: Running DVS simulator...")
            step3_start = time.time()
            
            events_array = self._run_dvs_simulator(temp_dir)
            
            timing_info['dvs_simulation_sec'] = time.time() - step3_start
            print(f"  Generated {len(events_array)} events in {timing_info['dvs_simulation_sec']:.3f}s")
            
            # Combine metadata
            timing_info.update(flare_metadata)
            timing_info['total_pipeline_sec'] = time.time() - total_start
            
            return events_array, timing_info
            
        finally:
            # Cleanup temporary directory if requested
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    def _run_dvs_simulator(self, input_dir: str) -> np.ndarray:
        """Run the DVS-Voltmeter simulator on the prepared video frames.
        
        Args:
            input_dir: Directory containing video frames and info.txt
            
        Returns:
            Events array in format [timestamp_us, x, y, polarity]
        """
        # Change to simulator directory
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.simulator_path)
            
            # Prepare simulator config (modify paths dynamically)
            self._prepare_simulator_config(input_dir)
            
            # Run simulator
            result = subprocess.run([
                sys.executable, "main.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(f"DVS simulator failed: {result.stderr}")
            
            # Load generated events (DVS simulator outputs to OUT_PATH/video_name.txt)
            # The simulator outputs to OUT_PATH/flare_sequence.txt based on the directory name
            output_file = os.path.join(input_dir, "flare_sequence.txt")  
            if not os.path.exists(output_file):
                # Also check for other possible output locations
                alt_output_file = os.path.join(input_dir, "flare_sequence", "flare_sequence.txt")
                if os.path.exists(alt_output_file):
                    output_file = alt_output_file
                else:
                    # List all .txt files in the output directory for debugging
                    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
                    raise FileNotFoundError(f"DVS simulator output not found: {output_file}. Available files: {txt_files}")
            
            # Parse events
            events = self._parse_event_file(output_file)
            
            return events
            
        finally:
            os.chdir(original_cwd)
            
    def _save_video_for_dvs_simulator(self, video_frames: List[np.ndarray], 
                                    temp_dir: str, metadata: Dict = None) -> str:
        """Save video frames in the structure expected by DVS simulator.
        
        The DVS simulator expects:
        - temp_dir/flare_sequence/info.txt  (frame list with timestamps)
        - temp_dir/flare_sequence/*.png     (frame images)
        
        Args:
            video_frames: List of RGB video frames
            temp_dir: Temporary directory for storage
            
        Returns:
            Path to the sequence directory
        """
        sequence_name = "flare_sequence"
        sequence_dir = os.path.join(temp_dir, sequence_name)
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Calculate timestamps based on video metadata (dynamic FPS)
        if metadata and 'fps' in metadata:
            fps = metadata['fps']
        else:
            # Fallback: estimate from frame count and duration
            duration_sec = self.config['data']['flare_synthesis']['duration_sec']
            fps = len(video_frames) / duration_sec
        
        frame_duration_us = int(1e6 / fps)  # microseconds per frame
        
        frame_paths = []
        
        # Save frames and collect info
        for i, frame in enumerate(video_frames):
            frame_filename = f"{i:06d}.png"
            frame_path = os.path.join(sequence_dir, frame_filename)
            
            # Convert RGB to BGR for OpenCV and save
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
            
            # DVS simulator expects absolute paths or paths relative to IN_PATH
            relative_path = frame_path  # Use absolute path for now
            timestamp_us = i * frame_duration_us
            frame_paths.append((relative_path, timestamp_us))
        
        # Create info.txt in the sequence directory
        info_path = os.path.join(sequence_dir, "info.txt")
        with open(info_path, 'w') as f:
            for frame_path, timestamp in frame_paths:
                f.write(f"{frame_path} {timestamp:012d}\n")
        
        return sequence_dir
            
    def _prepare_simulator_config(self, input_dir: str):
        """Prepare DVS simulator configuration for the input directory.
        
        Args:
            input_dir: Directory containing video frames
        """
        config_path = os.path.join(self.simulator_path, "src/config.py")
        
        # Read current config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Create backup
        backup_path = config_path + ".backup"
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(config_content)
        
        # Fix the path replacement - need to match the actual format in config.py
        modified_content = config_content.replace(
            "__C.DIR.IN_PATH = '/tmp/flare_events_4kx_xqe4/'",
            f"__C.DIR.IN_PATH = '{input_dir}'"
        ).replace(
            "__C.DIR.OUT_PATH = '/tmp/flare_events_4kx_xqe4/'", 
            f"__C.DIR.OUT_PATH = '{input_dir}'"
        )
        
        # Also handle any remaining old paths
        if "data_samples/interp/" in modified_content:
            modified_content = modified_content.replace(
                "data_samples/interp/",
                f"{input_dir}/"
            )
        if "data_samples/output/" in modified_content:
            modified_content = modified_content.replace(
                "data_samples/output/", 
                f"{input_dir}/"
            )
        
        # Write modified config
        with open(config_path, 'w') as f:
            f.write(modified_content)
            
    def _parse_event_file(self, event_file_path: str) -> np.ndarray:
        """Parse DVS simulator output file to events array.
        
        Args:
            event_file_path: Path to events text file
            
        Returns:
            Events array [timestamp_us, x, y, polarity]
        """
        events = []
        
        with open(event_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:
                    # Parse: timestamp_us, x, y, polarity
                    timestamp = int(float(parts[0]))
                    x = int(parts[1])
                    y = int(parts[2])
                    polarity = int(parts[3])
                    
                    events.append([timestamp, x, y, polarity])
        
        return np.array(events) if events else np.empty((0, 4))
    
    def restore_simulator_config(self):
        """Restore original DVS simulator configuration."""
        config_path = os.path.join(self.simulator_path, "src/config.py")
        backup_path = config_path + ".backup"
        
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, config_path)
            os.remove(backup_path)


def test_dvs_flare_integration(config_path: str = "configs/config.yaml"):
    """Test the complete DVS-flare integration pipeline."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize generator
    generator = DVSFlareEventGenerator(config)
    
    print("Testing DVS-Flare integration...")
    print("=" * 50)
    
    try:
        # Generate flare events
        events, timing_info = generator.generate_flare_events()
        
        print(f"Results:")
        print(f"  Generated events: {len(events)}")
        if len(events) > 0:
            print(f"  Time range: {events[0, 0]} - {events[-1, 0]} μs")
            print(f"  Duration: {(events[-1, 0] - events[0, 0]) / 1000:.1f} ms")
            print(f"  Event rate: {len(events) / ((events[-1, 0] - events[0, 0]) / 1e6):.1f} events/s")
            
            # Analyze polarity distribution
            pos_events = np.sum(events[:, 3] == 1)
            neg_events = np.sum(events[:, 3] == 0)
            print(f"  Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), "
                  f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)")
        
        print(f"\nTiming breakdown:")
        for key, value in timing_info.items():
            if key.endswith('_sec'):
                print(f"  {key}: {value:.3f}s")
        
        return events, timing_info
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return None, None
    
    finally:
        # Restore simulator config
        generator.restore_simulator_config()


if __name__ == "__main__":
    # Run test
    events, timing = test_dvs_flare_integration()
    if events is not None:
        print("\nDVS-Flare integration test completed successfully!")
    else:
        print("\nDVS-Flare integration test failed!")