"""
100%完整数据处理脚本
专门处理全部3,869,579条数据
"""

import argparse
import gc
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
REVIEW_CODE_ROOT = Path(__file__).resolve().parents[2]
FOR_REVIEW_ROOT = REVIEW_CODE_ROOT.parent
sys.path.insert(0, str(REVIEW_CODE_ROOT))

warnings.filterwarnings('ignore')

class Full100PercentProcessor:
    """100%完整数据处理器"""

    def __init__(
        self,
        input_file=FOR_REVIEW_ROOT / "ods_sq_flight_dynamic_data.csv",
        output_dir=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m",
    ):
        self.data = None
        self.trajectories = None
        self.trajectory_stats = None
        self.processed_chunks = []
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)

    def process_full_dataset(self, chunk_size=200000):
        """处理完整数据集（分块处理）"""
        print("=" * 70)
        print("开始处理100%完整数据集")
        print("=" * 70)

        data_file = self.input_file

        if not data_file.exists():
            print(f"错误: 数据文件不存在: {data_file}")
            return None

        # 获取文件大小和估计行数
        file_size_gb = data_file.stat().st_size / (1024**3)
        print(f"数据文件: {data_file}")
        print(f"文件大小: {file_size_gb:.2f} GB")
        print(f"分块大小: {chunk_size:,} 行")

        # 分块读取和处理
        total_rows = 0
        chunk_count = 0
        start_time = time.time()

        # 创建输出目录
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "temp").mkdir(parents=True, exist_ok=True)

        for chunk_idx, chunk in enumerate(pd.read_csv(
            data_file,
            chunksize=chunk_size,
            low_memory=False
        )):
            chunk_count += 1
            total_rows += len(chunk)

            print(f"\n处理块 {chunk_count}: {len(chunk):,} 行")
            print(f"累计进度: {total_rows:,} / 3,869,579 行 ({total_rows/3869579*100:.1f}%)")

            # 处理当前块
            processed_chunk = self._process_chunk(chunk, chunk_idx)

            # 保存处理后的块
            chunk_file = output_dir / "temp" / f"processed_chunk_{chunk_idx:03d}.pkl"
            with open(chunk_file, 'wb') as f:
                pickle.dump(processed_chunk, f)

            self.processed_chunks.append(chunk_file)

            # 清理内存
            del chunk, processed_chunk
            gc.collect()

            # 显示估计剩余时间
            elapsed_time = time.time() - start_time
            rows_per_second = total_rows / elapsed_time
            remaining_rows = 3869579 - total_rows
            if rows_per_second > 0:
                remaining_time = remaining_rows / rows_per_second
                print(f"处理速度: {rows_per_second:.0f} 行/秒")
                print(f"估计剩余时间: {remaining_time/60:.1f} 分钟")

            # 每处理10个块保存一次进度
            if chunk_count % 10 == 0:
                self._save_progress(total_rows, chunk_count, elapsed_time)

        # 合并所有块
        print(f"\n{'='*70}")
        print("合并所有数据块...")
        print(f"{'='*70}")

        self._merge_all_chunks()

        total_elapsed_time = time.time() - start_time
        print(f"\n数据处理完成!")
        print(f"总行数: {len(self.data):,}")
        print(f"总时间: {total_elapsed_time/60:.1f} 分钟")
        print(f"平均速度: {total_rows/total_elapsed_time:.0f} 行/秒")

        # 保存最终数据
        self._save_final_data()

        return self.data

    def _process_chunk(self, chunk, chunk_idx):
        """处理单个数据块"""
        # 复制数据避免修改原始块
        chunk = chunk.copy()

        print(f"  块 {chunk_idx}: 原始形状 {chunk.shape}")

        # 1. 基本数据清洗
        initial_rows = len(chunk)

        # 移除完全空的行
        chunk = chunk.dropna(how='all')
        print(f"  移除空行后: {len(chunk):,} 行 (移除 {initial_rows - len(chunk):,})")

        # 2. 解析时间
        if 'time' in chunk.columns:
            # 确保时间是字符串格式
            chunk['time_str'] = chunk['time'].astype(str).apply(lambda x: x.zfill(14) if pd.notna(x) else x)

            # 解析日期时间
            chunk['datetime'] = pd.to_datetime(chunk['time_str'], format='%Y%m%d%H%M%S', errors='coerce')

            # 移除无效的日期时间
            invalid_dt = chunk['datetime'].isna().sum()
            if invalid_dt > 0:
                print(f"  移除无效日期时间: {invalid_dt:,}")
                chunk = chunk[chunk['datetime'].notna()]

            # 提取时间特征
            chunk['date'] = chunk['datetime'].dt.date
            chunk['hour'] = chunk['datetime'].dt.hour
            chunk['minute'] = chunk['datetime'].dt.minute
            chunk['second'] = chunk['datetime'].dt.second
            chunk['day_of_week'] = chunk['datetime'].dt.dayofweek
            chunk['is_weekend'] = chunk['day_of_week'] >= 5
        else:
            print(f"  警告: 块 {chunk_idx} 没有 'time' 列")

        # 3. 数值列清洗
        numeric_columns = ['speed', 'altitude', 'yaw', 'flight_time', 'longitude', 'latitude']

        for col in numeric_columns:
            if col in chunk.columns:
                # 转换为数值类型
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

                # 统计无效值
                invalid_count = chunk[col].isna().sum()
                if invalid_count > 0:
                    print(f"  {col}: {invalid_count:,} 个无效值")

        # 4. 速度清洗 (0-40 m/s)
        if 'speed' in chunk.columns:
            valid_speed = (chunk['speed'] >= 0) & (chunk['speed'] <= 40)
            invalid_speed = (~valid_speed).sum()
            if invalid_speed > 0:
                print(f"  移除无效速度: {invalid_speed:,}")
                chunk = chunk[valid_speed]

        # 5. 高度清洗 (0-500 m)
        if 'altitude' in chunk.columns:
            valid_altitude = (chunk['altitude'] >= 0) & (chunk['altitude'] <= 500)
            invalid_altitude = (~valid_altitude).sum()
            if invalid_altitude > 0:
                print(f"  移除无效高度: {invalid_altitude:,}")
                chunk = chunk[valid_altitude]

        # 6. 航向清洗
        if 'yaw' in chunk.columns:
            # 标记999.0为无效
            chunk['yaw_valid'] = chunk['yaw'] != 999.0
            chunk['yaw_clean'] = chunk['yaw'].where(chunk['yaw'] != 999.0, np.nan)

            invalid_yaw = (~chunk['yaw_valid']).sum()
            if invalid_yaw > 0:
                print(f"  无效航向值: {invalid_yaw:,}")

        # 7. 飞行时间清洗
        if 'flight_time' in chunk.columns:
            valid_flight_time = chunk['flight_time'] >= 0
            invalid_flight_time = (~valid_flight_time).sum()
            if invalid_flight_time > 0:
                print(f"  移除无效飞行时间: {invalid_flight_time:,}")
                chunk = chunk[valid_flight_time]

        # 8. 经纬度清洗 (深圳范围)
        if 'longitude' in chunk.columns and 'latitude' in chunk.columns:
            # 深圳大致范围: 113.31-114.25°E, 22.31-22.87°N
            valid_longitude = (chunk['longitude'] >= 113.31) & (chunk['longitude'] <= 114.25)
            valid_latitude = (chunk['latitude'] >= 22.31) & (chunk['latitude'] <= 22.87)
            valid_coords = valid_longitude & valid_latitude

            invalid_coords = (~valid_coords).sum()
            if invalid_coords > 0:
                print(f"  移除无效经纬度: {invalid_coords:,}")
                chunk = chunk[valid_coords]

        print(f"  块 {chunk_idx}: 处理后形状 {chunk.shape}")
        print(f"  保留率: {len(chunk)/initial_rows*100:.1f}%")

        return chunk

    def _merge_all_chunks(self):
        """合并所有处理后的块"""
        print(f"合并 {len(self.processed_chunks)} 个数据块...")

        chunks_data = []
        total_rows = 0

        for i, chunk_file in enumerate(self.processed_chunks):
            if Path(chunk_file).exists():
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)

                chunks_data.append(chunk_data)
                total_rows += len(chunk_data)

                print(f"  加载块 {i+1}/{len(self.processed_chunks)}: {len(chunk_data):,} 行")

                # 每加载5个块清理一次内存
                if (i + 1) % 5 == 0:
                    gc.collect()

        # 合并所有数据
        if chunks_data:
            print("合并数据...")
            self.data = pd.concat(chunks_data, ignore_index=True)
            print(f"合并完成: {len(self.data):,} 行")

            # 清理内存
            del chunks_data
            gc.collect()
        else:
            print("错误: 没有数据块可合并")
            self.data = pd.DataFrame()

    def _save_final_data(self):
        """保存最终数据"""
        if self.data is None or len(self.data) == 0:
            print("错误: 没有数据可保存")
            return

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存清洗后的数据
        data_file = output_dir / "cleaned_data.csv"
        print(f"保存清洗后数据到: {data_file}")

        # 只保存必要的列以节省空间
        columns_to_save = [
            'datetime', 'date', 'hour', 'minute', 'second',
            'day_of_week', 'is_weekend',
            'speed', 'altitude', 'yaw_clean', 'flight_time',
            'longitude', 'latitude',  # 添加经纬度列
            'order_id', 'sn', 'vendor'
        ]

        # 只保留实际存在的列
        existing_columns = [col for col in columns_to_save if col in self.data.columns]
        data_to_save = self.data[existing_columns]

        # 分块保存CSV
        chunk_size = 500000
        for i in range(0, len(data_to_save), chunk_size):
            chunk = data_to_save.iloc[i:i+chunk_size]
            mode = 'w' if i == 0 else 'a'
            header = (i == 0)
            chunk.to_csv(data_file, mode=mode, header=header, index=False)

            print(f"  保存块 {i//chunk_size + 1}: {len(chunk):,} 行")

        print(f"清洗后数据已保存: {data_file} ({len(data_to_save):,} 行)")

        # 2. 重建轨迹并保存轨迹数据
        print("\n重建轨迹...")
        self.reconstruct_trajectories()

        # 3. 保存轨迹统计
        if self.trajectory_stats is not None:
            stats_file = output_dir / "trajectory_stats.csv"
            self.trajectory_stats.to_csv(stats_file, index=False)
            print(f"轨迹统计已保存: {stats_file}")

        # 4. 保存轨迹数据
        if self.trajectories is not None:
            trajectories_file = output_dir / "trajectories.pkl"
            with open(trajectories_file, 'wb') as f:
                pickle.dump(self.trajectories, f)
            print(f"轨迹数据已保存: {trajectories_file}")

        # 5. 生成数据摘要
        self._generate_data_summary()

    def reconstruct_trajectories(self):
        """重建轨迹"""
        if self.data is None or len(self.data) == 0:
            print("错误: 没有数据可重建轨迹")
            return

        print("重建轨迹中...")

        # 确保数据按时间和order_id排序
        self.data = self.data.sort_values(['order_id', 'datetime'])

        # 按order_id分组
        trajectories = []
        stats_records = []

        for order_id, group in self.data.groupby('order_id'):
            if len(group) >= 3:  # 至少3个点才算是有效轨迹
                trajectories.append({
                    'order_id': order_id,
                    'points': group[['datetime', 'speed', 'altitude', 'yaw_clean']].to_dict('records'),
                    'vendor': group['vendor'].iloc[0] if 'vendor' in group.columns else None,
                    'start_time': group['datetime'].iloc[0],
                    'end_time': group['datetime'].iloc[-1],
                    'duration': (group['datetime'].iloc[-1] - group['datetime'].iloc[0]).total_seconds(),
                    'num_points': len(group),
                    'avg_speed': group['speed'].mean(),
                    'avg_altitude': group['altitude'].mean()
                })

                stats_records.append({
                    'order_id': order_id,
                    'num_points': len(group),
                    'duration_seconds': (group['datetime'].iloc[-1] - group['datetime'].iloc[0]).total_seconds(),
                    'avg_speed': group['speed'].mean(),
                    'avg_altitude': group['altitude'].mean(),
                    'start_time': group['datetime'].iloc[0],
                    'end_time': group['datetime'].iloc[-1],
                    'vendor': group['vendor'].iloc[0] if 'vendor' in group.columns else None
                })

        self.trajectories = trajectories
        self.trajectory_stats = pd.DataFrame(stats_records)

        print(f"轨迹重建完成: {len(trajectories):,} 条有效轨迹")

    def _save_progress(self, total_rows, chunk_count, elapsed_time):
        """保存处理进度"""
        progress_file = self.output_dir / "processing_progress.json"

        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'total_rows_processed': total_rows,
            'chunks_processed': chunk_count,
            'elapsed_time_seconds': elapsed_time,
            'rows_per_second': total_rows / elapsed_time if elapsed_time > 0 else 0,
            'completion_percentage': total_rows / 3869579 * 100
        }

        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)

        print(f"进度已保存: {progress_file}")

    def _generate_data_summary(self):
        """生成数据摘要"""
        if self.data is None:
            return

        summary_file = self.output_dir / "data_summary.txt"

        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("100%完整数据摘要报告")
        summary_lines.append("=" * 60)
        summary_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")

        # 基本统计
        summary_lines.append("基本统计:")
        summary_lines.append(f"  总行数: {len(self.data):,}")
        summary_lines.append(f"  时间范围: {self.data['datetime'].min()} 到 {self.data['datetime'].max()}")
        summary_lines.append(f"  天数: {(self.data['datetime'].max() - self.data['datetime'].min()).days + 1} 天")
        summary_lines.append("")

        # 轨迹统计
        if self.trajectory_stats is not None:
            summary_lines.append("轨迹统计:")
            summary_lines.append(f"  有效轨迹数: {len(self.trajectory_stats):,}")
            summary_lines.append(f"  平均轨迹点数: {self.trajectory_stats['num_points'].mean():.1f}")
            summary_lines.append(f"  平均轨迹时长: {self.trajectory_stats['duration_seconds'].mean():.1f} 秒")
            summary_lines.append(f"  平均速度: {self.trajectory_stats['avg_speed'].mean():.2f} m/s")
            summary_lines.append(f"  平均高度: {self.trajectory_stats['avg_altitude'].mean():.2f} m")
            summary_lines.append("")

        # 时间分布
        summary_lines.append("时间分布:")
        summary_lines.append(f"  小时分布:")
        for hour in range(24):
            hour_count = (self.data['hour'] == hour).sum()
            hour_percent = hour_count / len(self.data) * 100
            summary_lines.append(f"    {hour:02d}:00-{hour:02d}:59: {hour_count:,} ({hour_percent:.1f}%)")

        # 保存摘要
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        print(f"数据摘要已保存: {summary_file}")

# 全局导入
import time
import json

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Preprocess the proprietary raw trajectory archive.")
    parser.add_argument("--input", default=FOR_REVIEW_ROOT / "ods_sq_flight_dynamic_data.csv", help="Path to the raw trajectory CSV.")
    parser.add_argument("--output-dir", default=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m", help="Directory for cleaned outputs.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="CSV chunk size for preprocessing.")
    args = parser.parse_args()

    print("100%完整数据处理")
    print("=" * 60)

    processor = Full100PercentProcessor(input_file=args.input, output_dir=args.output_dir)

    try:
        # 处理完整数据集
        data = processor.process_full_dataset(chunk_size=args.chunk_size)

        if data is not None:
            print("\n" + "=" * 60)
            print("数据处理成功完成!")
            print("=" * 60)
            return True
        else:
            print("\n数据处理失败")
            return False

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

