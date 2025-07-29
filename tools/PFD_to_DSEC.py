import h5py
import numpy as np
import argparse
from tqdm import tqdm
import hdf5plugin # 确保导入
import math

def pfd_to_dsec_absolute_time(pfd_txt_path, output_h5_path, original_h5_path):
    """
    Converts event data from a PFD TXT file back to a DSEC-compatible HDF5 file.
    This version correctly re-generates the 'ms_to_idx' lookup table.
    """
    print(f"Loading PFD data from: {pfd_txt_path}")
    
    try:
        # 1. 从原始HDF5文件获取 t_offset
        print(f"Reading t_offset from original file: {original_h5_path}")
        with h5py.File(original_h5_path, 'r') as f_orig:
            t_offset = f_orig['t_offset'][()] if 't_offset' in f_orig else 0
        print(f"Using t_offset: {t_offset}")

        # 2. 加载PFD的输出数据
        data = np.loadtxt(pfd_txt_path, delimiter=' ')
        num_events = data.shape[0]
        print(f"Found {num_events:,} denoised events.")
        
        # 分离各列并进行类型转换
        events_x = data[:, 0].astype(np.uint16)
        events_y = data[:, 1].astype(np.uint16)
        events_p_raw = data[:, 2]
        events_t_absolute = data[:, 3].astype(np.int64)
        
        # 3. 将绝对时间戳转换回DSEC的相对时间戳
        events_t_relative = events_t_absolute.astype(np.uint32)
        # 确保时间戳是排序的
        if not np.all(np.diff(events_t_relative) >= 0):
             print("Warning: Timestamps in denoised file are not sorted. Sorting them now.")
             sort_indices = np.argsort(events_t_relative)
             events_x = events_x[sort_indices]
             events_y = events_y[sort_indices]
             events_p_raw = events_p_raw[sort_indices]
             events_t_relative = events_t_relative[sort_indices]

        # 4. 将极性从-1/1转换回0/1
        events_p = np.where(events_p_raw == -1, 0, 1).astype(np.uint8)
        
        # ========================= 核心修复: 重新生成 ms_to_idx =========================
        print("\nRe-generating 'ms_to_idx' lookup table for the new event stream...")
        if num_events > 0:
            # 获取最后一个事件的时间戳 (单位: µs)
            last_timestamp_us = events_t_relative[-1]
            # 计算总时长 (单位: ms), 向上取整
            total_duration_ms = math.ceil(last_timestamp_us / 1000.0)
            print(f"Total duration of denoised events: {total_duration_ms} ms.")

            # 创建一个从 0ms, 1ms, 2ms, ... 对应的微秒时间戳数组
            ms_timestamps_us = np.arange(total_duration_ms) * 1000
            
            # 使用np.searchsorted高效地查找每个毫秒对应的事件索引
            # 'left'表示找到第一个 t >= target_t 的位置
            new_ms_to_idx = np.searchsorted(events_t_relative, ms_timestamps_us, side='left')
            
            print(f"New 'ms_to_idx' generated with shape: {new_ms_to_idx.shape}")
        else:
            new_ms_to_idx = np.array([], dtype=np.uint64)
            print("No events found, creating an empty 'ms_to_idx'.")
        # ==============================================================================

        print(f"\nWriting to DSEC HDF5 format at: {output_h5_path}")
        with h5py.File(output_h5_path, 'w') as f_out:
            events_group = f_out.create_group('events')
            
            events_group.create_dataset('x', data=events_x, compression='gzip')
            events_group.create_dataset('y', data=events_y, compression='gzip')
            events_group.create_dataset('t', data=events_t_relative, compression='gzip')
            events_group.create_dataset('p', data=events_p, compression='gzip')
            
            f_out.create_dataset('t_offset', data=t_offset)
            f_out.create_dataset('ms_to_idx', data=new_ms_to_idx.astype(np.uint64), compression='gzip')
        
        print("Conversion successful! Denoised HDF5 file with correct 'ms_to_idx' created.")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. Details: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PFD TXT event data back to DSEC HDF5 format.")
    parser.add_argument("--input_txt",default="E:/2025/event_flick_flare/object_detection/dsec-det-master/data/test/zurich_city_12_a/events/left/denoised_25000_3.txt",help="Path to the input PFD .txt file (e.g., denoised.txt).")
    parser.add_argument("--output_h5",default="E:/2025/event_flick_flare/object_detection/dsec-det-master/data/test/zurich_city_12_a/events/left/denoised_25000_3.h5", help="Path for the output .h5 file.")
    parser.add_argument("--original_h5", default="E:/2025/event_flick_flare/object_detection/dsec-det-master/data/test/zurich_city_12_a/events/left/events.h5",help="Path to the original DSEC .h5 file to get t_offset from.")

    args = parser.parse_args()
    
    pfd_to_dsec_absolute_time(args.input_txt, args.output_h5, args.original_h5)