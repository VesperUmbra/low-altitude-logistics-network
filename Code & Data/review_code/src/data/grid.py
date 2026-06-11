"""
100%完整数据网格化处理
将3,869,563条轨迹点数据网格化到100m×2min的时空网格
"""

import argparse
import gc
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# 添加项目根目录到Python路径
REVIEW_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REVIEW_CODE_ROOT))

warnings.filterwarnings('ignore')

class FullDataGridProcessor:
    """100%完整数据网格处理器"""

    def __init__(
        self,
        cleaned_data_file=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m" / "cleaned_data.csv",
        output_dir=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m" / "grid",
        grid_size=100,
    ):
        self.data = None
        self.grid_data = None
        self.cell_stats = None
        self.spatiotemporal_stats = None
        self.cleaned_data_file = Path(cleaned_data_file)
        self.output_dir = Path(output_dir)

        # 网格参数
        self.grid_size = grid_size  # 米
        self.time_window = 120  # 秒 (2分钟)
        self.min_lon = 113.31
        self.max_lon = 114.25
        self.min_lat = 22.31
        self.max_lat = 22.87

    def load_cleaned_data(self):
        """加载清洗后的数据"""
        print("加载清洗后的数据...")

        data_file = self.cleaned_data_file

        if not data_file.exists():
            print(f"错误: 数据文件不存在: {data_file}")
            return False

        # 获取文件大小
        file_size = data_file.stat().st_size / (1024**2)  # MB
        print(f"数据文件: {data_file}")
        print(f"文件大小: {file_size:.1f} MB")

        # 分块读取数据
        chunk_size = 500000
        chunks = []
        total_rows = 0

        print("分块读取数据...")
        for chunk_idx, chunk in enumerate(pd.read_csv(data_file, chunksize=chunk_size)):
            # 解析日期时间
            if 'datetime' in chunk.columns:
                chunk['datetime'] = pd.to_datetime(chunk['datetime'])

            chunks.append(chunk)
            total_rows += len(chunk)

            print(f"  读取块 {chunk_idx + 1}: {len(chunk):,} 行")

            # 测试读取：先读取前3个块
            if chunk_idx >= 2 and len(chunks) * chunk_size >= 1000000:
                print(f"  已读取 {total_rows:,} 行，继续完整读取...")

        # 合并数据
        self.data = pd.concat(chunks, ignore_index=True)
        print(f"数据加载完成: {len(self.data):,} 行")

        # 显示数据信息
        print(f"时间范围: {self.data['datetime'].min()} 到 {self.data['datetime'].max()}")
        print(f"天数: {(self.data['datetime'].max() - self.data['datetime'].min()).days + 1} 天")

        return True

    def create_spatial_grid(self):
        """创建空间网格"""
        print("\n创建空间网格...")

        # 计算网格参数
        # 使用近似转换：1度纬度 ≈ 111km，1度经度 ≈ 111km * cos(latitude)
        lat_center = (self.min_lat + self.max_lat) / 2
        meters_per_degree_lat = 111000  # 1度纬度 ≈ 111km
        meters_per_degree_lon = 111000 * np.cos(np.radians(lat_center))

        # 计算范围（米）
        lat_range_m = (self.max_lat - self.min_lat) * meters_per_degree_lat
        lon_range_m = (self.max_lon - self.min_lon) * meters_per_degree_lon

        # 计算网格维度
        n_rows = int(np.ceil(lat_range_m / self.grid_size))
        n_cols = int(np.ceil(lon_range_m / self.grid_size))

        print(f"空间范围: {lat_range_m:.0f}m × {lon_range_m:.0f}m")
        print(f"网格大小: {self.grid_size}m")
        print(f"网格维度: {n_rows} × {n_cols}")
        print(f"总网格单元: {n_rows * n_cols:,}")

        # 创建网格映射函数
        def lat_lon_to_grid(lat, lon):
            """将经纬度转换为网格坐标"""
            row = int(((lat - self.min_lat) * meters_per_degree_lat) / self.grid_size)
            col = int(((lon - self.min_lon) * meters_per_degree_lon) / self.grid_size)

            # 确保在网格范围内
            row = max(0, min(row, n_rows - 1))
            col = max(0, min(col, n_cols - 1))

            return row, col

        def grid_to_cell_id(row, col):
            """将网格坐标转换为单元ID"""
            return f"{row}_{col}"

        # 应用网格化
        print("应用网格化到数据点...")

        # 分块处理以避免内存问题
        chunk_size = 100000
        grid_records = []

        for i in range(0, len(self.data), chunk_size):
            chunk = self.data.iloc[i:i+chunk_size]

            # 计算网格坐标
            rows, cols = zip(*chunk.apply(
                lambda x: lat_lon_to_grid(x['latitude'], x['longitude']),
                axis=1
            ))

            # 创建网格记录
            for idx, (row, col) in enumerate(zip(rows, cols)):
                original_idx = i + idx
                if original_idx < len(self.data):
                    grid_records.append({
                        'cell_id': grid_to_cell_id(row, col),
                        'row': row,
                        'col': col,
                        'original_index': original_idx,
                        'datetime': self.data.iloc[original_idx]['datetime'],
                        'speed': self.data.iloc[original_idx]['speed'] if 'speed' in self.data.columns else np.nan,
                        'altitude': self.data.iloc[original_idx]['altitude'] if 'altitude' in self.data.columns else np.nan
                    })

            print(f"  处理块 {i//chunk_size + 1}/{(len(self.data)-1)//chunk_size + 1}: {len(chunk):,} 点")

        # 创建网格数据DataFrame
        self.grid_data = pd.DataFrame(grid_records)
        print(f"网格数据创建完成: {len(self.grid_data):,} 条记录")

        # 保存网格信息
        grid_info = {
            'grid_size_m': self.grid_size,
            'time_window_s': self.time_window,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'total_cells': n_rows * n_cols,
            'min_lat': self.min_lat,
            'max_lat': self.max_lat,
            'min_lon': self.min_lon,
            'max_lon': self.max_lon,
            'meters_per_degree_lat': meters_per_degree_lat,
            'meters_per_degree_lon': meters_per_degree_lon,
            'creation_time': datetime.now().isoformat()
        }

        grid_info_dir = self.output_dir
        grid_info_dir.mkdir(parents=True, exist_ok=True)

        with open(grid_info_dir / "grid_info.json", 'w', encoding='utf-8') as f:
            json.dump(grid_info, f, indent=2, ensure_ascii=False)

        print(f"网格信息已保存: {grid_info_dir / 'grid_info.json'}")

        return True

    def create_temporal_windows(self):
        """创建时间窗口"""
        print("\n创建时间窗口...")

        if self.grid_data is None or len(self.grid_data) == 0:
            print("错误: 没有网格数据")
            return False

        # 确定时间范围
        min_time = self.grid_data['datetime'].min()
        max_time = self.grid_data['datetime'].max()

        # 创建时间窗口，并按当前窗口长度对齐到小时内边界。
        window_start = min_time.replace(second=0, microsecond=0)
        hour_start = window_start.replace(minute=0)
        seconds_from_hour = (window_start - hour_start).total_seconds()
        aligned_seconds = int(seconds_from_hour // self.time_window) * self.time_window
        window_start = hour_start + timedelta(seconds=aligned_seconds)

        windows = []
        current_window = window_start

        while current_window <= max_time:
            windows.append(current_window)
            current_window += timedelta(seconds=self.time_window)

        print(f"时间范围: {min_time} 到 {max_time}")
        print(f"时间窗口大小: {self.time_window} 秒 ({self.time_window/60:g}分钟)")
        print(f"时间窗口数: {len(windows):,}")
        print(f"第一个窗口: {windows[0]}")
        print(f"最后一个窗口: {windows[-1]}")

        # 将数据点分配到时间窗口
        print("分配数据点到时间窗口...")

        def find_time_window(dt):
            """找到数据点所属的时间窗口"""
            # 计算从起始时间开始的窗口索引
            seconds_from_start = (dt - window_start).total_seconds()
            window_idx = int(seconds_from_start // self.time_window)
            return window_idx if window_idx >= 0 and window_idx < len(windows) else -1

        # 应用时间窗口分配
        self.grid_data['window_idx'] = self.grid_data['datetime'].apply(find_time_window)
        self.grid_data['window_start'] = self.grid_data['window_idx'].apply(
            lambda idx: windows[idx] if idx >= 0 and idx < len(windows) else None
        )

        # 移除无法分配时间窗口的数据点
        valid_mask = self.grid_data['window_idx'] >= 0
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"移除无法分配时间窗口的数据点: {invalid_count:,}")
            self.grid_data = self.grid_data[valid_mask]

        print(f"时间窗口分配完成: {len(self.grid_data):,} 条有效记录")

        return True

    def compute_cell_statistics(self):
        """计算单元统计"""
        print("\n计算单元统计...")

        if self.grid_data is None or len(self.grid_data) == 0:
            print("错误: 没有网格数据")
            return False

        # 按空间单元分组统计
        print("按空间单元统计...")

        cell_stats_records = []

        # 获取所有唯一的空间单元
        unique_cells = self.grid_data['cell_id'].unique()
        print(f"唯一空间单元数: {len(unique_cells):,}")

        # 分块处理
        chunk_size = 1000
        for i in range(0, len(unique_cells), chunk_size):
            cell_chunk = unique_cells[i:i+chunk_size]
            chunk_data = self.grid_data[self.grid_data['cell_id'].isin(cell_chunk)]

            for cell_id in cell_chunk:
                cell_data = chunk_data[chunk_data['cell_id'] == cell_id]

                if len(cell_data) > 0:
                    # 解析行列号
                    if '_' in cell_id:
                        row_str, col_str = cell_id.split('_')
                        row = int(row_str)
                        col = int(col_str)
                    else:
                        row = col = -1

                    cell_stats_records.append({
                        'cell_id': cell_id,
                        'row': row,
                        'col': col,
                        'total_points': len(cell_data),
                        'avg_speed': cell_data['speed'].mean() if 'speed' in cell_data.columns else np.nan,
                        'std_speed': cell_data['speed'].std() if 'speed' in cell_data.columns else np.nan,
                        'avg_altitude': cell_data['altitude'].mean() if 'altitude' in cell_data.columns else np.nan,
                        'std_altitude': cell_data['altitude'].std() if 'altitude' in cell_data.columns else np.nan,
                        'first_seen': cell_data['datetime'].min(),
                        'last_seen': cell_data['datetime'].max(),
                        'active_hours': len(cell_data['datetime'].dt.hour.unique()),
                        'active_days': len(cell_data['datetime'].dt.date.unique())
                    })

            print(f"  处理单元块 {i//chunk_size + 1}/{(len(unique_cells)-1)//chunk_size + 1}: {len(cell_chunk):,} 单元")

        self.cell_stats = pd.DataFrame(cell_stats_records)
        print(f"单元统计计算完成: {len(self.cell_stats):,} 个单元")

        # 排序并添加排名
        self.cell_stats = self.cell_stats.sort_values('total_points', ascending=False).reset_index(drop=True)
        self.cell_stats['rank'] = range(1, len(self.cell_stats) + 1)

        return True

    def compute_spatiotemporal_statistics(self):
        """计算时空统计"""
        print("\n计算时空统计...")

        if self.grid_data is None or len(self.grid_data) == 0:
            print("错误: 没有网格数据")
            return False

        # 按时空单元（cell_id + window_start）分组
        print("按时空单元统计...")

        # 创建时空单元ID
        self.grid_data['spatiotemporal_id'] = self.grid_data['cell_id'] + '_' + self.grid_data['window_start'].astype(str)

        # 分组统计
        spatiotemporal_groups = self.grid_data.groupby(['cell_id', 'window_start'])

        st_records = []
        total_groups = len(spatiotemporal_groups)

        print(f"时空单元数: {total_groups:,}")

        # 分块处理
        chunk_size = 50000
        group_keys = list(spatiotemporal_groups.groups.keys())

        for i in range(0, len(group_keys), chunk_size):
            chunk_keys = group_keys[i:i+chunk_size]

            for cell_id, window_start in chunk_keys:
                group_data = spatiotemporal_groups.get_group((cell_id, window_start))

                st_records.append({
                    'cell_id': cell_id,
                    'window_start': window_start,
                    'n_points': len(group_data),
                    'mean_speed': group_data['speed'].mean() if 'speed' in group_data.columns else np.nan,
                    'std_speed': group_data['speed'].std() if 'speed' in group_data.columns else np.nan,
                    'min_speed': group_data['speed'].min() if 'speed' in group_data.columns else np.nan,
                    'max_speed': group_data['speed'].max() if 'speed' in group_data.columns else np.nan,
                    'mean_altitude': group_data['altitude'].mean() if 'altitude' in group_data.columns else np.nan,
                    'hour': window_start.hour if hasattr(window_start, 'hour') else -1,
                    'day_of_week': window_start.weekday() if hasattr(window_start, 'weekday') else -1,
                    'is_weekend': window_start.weekday() >= 5 if hasattr(window_start, 'weekday') else False
                })

            print(f"  处理时空单元块 {i//chunk_size + 1}/{(len(group_keys)-1)//chunk_size + 1}: {len(chunk_keys):,} 单元")

        self.spatiotemporal_stats = pd.DataFrame(st_records)
        print(f"时空统计计算完成: {len(self.spatiotemporal_stats):,} 个时空单元")

        return True

    def save_results(self):
        """保存结果"""
        print("\n保存网格化结果...")

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存网格数据（采样，因为文件可能很大）
        if self.grid_data is not None:
            # 保存所有数据
            grid_data_file = output_dir / "gridded_data.csv"
            print(f"保存网格数据到: {grid_data_file}")

            # 只保存必要的列
            columns_to_save = ['cell_id', 'row', 'col', 'datetime', 'window_start', 'speed', 'altitude']
            existing_columns = [col for col in columns_to_save if col in self.grid_data.columns]

            # 分块保存
            chunk_size = 500000
            for i in range(0, len(self.grid_data), chunk_size):
                chunk = self.grid_data.iloc[i:i+chunk_size][existing_columns]
                mode = 'w' if i == 0 else 'a'
                header = (i == 0)
                chunk.to_csv(grid_data_file, mode=mode, header=header, index=False)

                print(f"  保存块 {i//chunk_size + 1}: {len(chunk):,} 行")

            print(f"网格数据已保存: {grid_data_file} ({len(self.grid_data):,} 行)")

        # 2. 保存单元统计
        if self.cell_stats is not None:
            cell_stats_file = output_dir / "cell_stats.csv"
            self.cell_stats.to_csv(cell_stats_file, index=False)
            print(f"单元统计已保存: {cell_stats_file} ({len(self.cell_stats):,} 行)")

        # 3. 保存时空统计
        if self.spatiotemporal_stats is not None:
            spatiotemporal_file = output_dir / "spatiotemporal_stats.csv"
            self.spatiotemporal_stats.to_csv(spatiotemporal_file, index=False)
            print(f"时空统计已保存: {spatiotemporal_file} ({len(self.spatiotemporal_stats):,} 行)")

        # 4. 生成摘要报告
        self._generate_summary_report(output_dir)

        print("所有网格化结果已保存")

    def _generate_summary_report(self, output_dir):
        """生成摘要报告"""
        report_file = Path(output_dir) / "grid_summary.txt"

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("100%完整数据网格化摘要报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 基本统计
        report_lines.append("基本统计:")
        if self.data is not None:
            report_lines.append(f"  原始数据点数: {len(self.data):,}")
        if self.grid_data is not None:
            report_lines.append(f"  网格化数据点数: {len(self.grid_data):,}")
            report_lines.append(f"  网格化保留率: {len(self.grid_data)/len(self.data)*100:.1f}%")
        report_lines.append("")

        # 空间网格统计
        report_lines.append("空间网格统计:")
        if self.cell_stats is not None:
            report_lines.append(f"  空间单元总数: {len(self.cell_stats):,}")
            report_lines.append(f"  有数据的空间单元: {len(self.cell_stats[self.cell_stats['total_points'] > 0]):,}")
            report_lines.append(f"  平均每单元点数: {self.cell_stats['total_points'].mean():.1f}")
            report_lines.append(f"  最大单元点数: {self.cell_stats['total_points'].max():,}")
            report_lines.append(f"  前10%单元承载流量: {self.cell_stats.nlargest(int(len(self.cell_stats)*0.1), 'total_points')['total_points'].sum()/self.cell_stats['total_points'].sum()*100:.1f}%")
        report_lines.append("")

        # 时空网格统计
        report_lines.append("时空网格统计:")
        if self.spatiotemporal_stats is not None:
            report_lines.append(f"  时空单元总数: {len(self.spatiotemporal_stats):,}")
            report_lines.append(f"  平均密度（点/单元）: {self.spatiotemporal_stats['n_points'].mean():.2f}")
            report_lines.append(f"  最大密度: {self.spatiotemporal_stats['n_points'].max():.0f}")
            report_lines.append(f"  平均速度: {self.spatiotemporal_stats['mean_speed'].mean():.2f} m/s")
            report_lines.append(f"  时间窗口数: {self.spatiotemporal_stats['window_start'].nunique():,}")
        report_lines.append("")

        # 网格参数
        report_lines.append("网格参数:")
        report_lines.append(f"  空间网格大小: {self.grid_size} 米")
        report_lines.append(f"  时间窗口大小: {self.time_window} 秒 ({self.time_window/60} 分钟)")
        report_lines.append(f"  空间范围: {self.min_lat:.3f}°-{self.max_lat:.3f}°N, {self.min_lon:.3f}°-{self.max_lon:.3f}°E")

        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"摘要报告已保存: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Discretize cleaned trajectories into a spatial-temporal grid.")
    parser.add_argument("--input", default=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m" / "cleaned_data.csv", help="Path to cleaned trajectory CSV.")
    parser.add_argument("--output-dir", default=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m" / "grid", help="Directory for grid outputs.")
    parser.add_argument("--grid-size", type=int, default=100, help="Spatial grid size in meters.")
    args = parser.parse_args()

    print("100%完整数据网格化处理")
    print("=" * 60)

    processor = FullDataGridProcessor(
        cleaned_data_file=args.input,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
    )

    try:
        # 1. 加载清洗后的数据
        if not processor.load_cleaned_data():
            print("加载数据失败")
            return False

        # 2. 创建空间网格
        if not processor.create_spatial_grid():
            print("创建空间网格失败")
            return False

        # 3. 创建时间窗口
        if not processor.create_temporal_windows():
            print("创建时间窗口失败")
            return False

        # 4. 计算单元统计
        if not processor.compute_cell_statistics():
            print("计算单元统计失败")
            return False

        # 5. 计算时空统计
        if not processor.compute_spatiotemporal_statistics():
            print("计算时空统计失败")
            return False

        # 6. 保存结果
        processor.save_results()

        print("\n" + "=" * 60)
        print("网格化处理成功完成!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
