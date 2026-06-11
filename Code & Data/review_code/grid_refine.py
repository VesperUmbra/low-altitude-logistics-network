"""
改进网格化精度的脚本
添加经纬度精度控制和OD聚合功能
"""

import argparse
import os
from math import cos, radians
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

class ImprovedSpatialGrid:
    """改进的空间网格化处理器"""

    def __init__(self, grid_size=100, lon_precision=4, lat_precision=4):
        """
        初始化

        参数:
        grid_size: 网格大小（米）
        lon_precision: 经度精度（小数点后位数）
        lat_precision: 纬度精度（小数点后位数）
        """
        self.grid_size = grid_size
        self.lon_precision = lon_precision
        self.lat_precision = lat_precision
        self.data = None
        self.trajectories = None
        self.grid = None
        self.od_pairs = None

    def load_data(self, data_path):
        """加载数据"""
        print(f"加载数据: {data_path}")
        self.data = pd.read_csv(data_path)

        # 确保必要的列存在
        required_cols = ['longitude', 'latitude', 'order_id', 'datetime']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        # 解析时间
        if 'datetime' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])

        print(f"数据加载完成: {len(self.data):,} 行")
        return self.data

    def apply_coordinate_precision(self):
        """应用坐标精度控制"""
        print(f"应用坐标精度控制: 经度精度={self.lon_precision}, 纬度精度={self.lat_precision}")

        # 舍入坐标
        self.data['lon_rounded'] = np.round(self.data['longitude'], self.lon_precision)
        self.data['lat_rounded'] = np.round(self.data['latitude'], self.lat_precision)

        # 计算精度对应的实际距离
        # 1度纬度 ≈ 111km
        lat_resolution = 111000 / (10 ** self.lat_precision)  # 米
        # 1度经度 ≈ 111km * cos(latitude)
        lat_mid = self.data['latitude'].mean()
        cos_lat = cos(radians(lat_mid))
        lon_resolution = 111000 * cos_lat / (10 ** self.lon_precision)  # 米

        print(f"坐标精度对应的空间分辨率:")
        print(f"  纬度: {lat_resolution:.1f} 米")
        print(f"  经度: {lon_resolution:.1f} 米 (cos({lat_mid:.1f}°)={cos_lat:.3f})")
        print(f"  唯一坐标点: {self.data[['lon_rounded', 'lat_rounded']].drop_duplicates().shape[0]:,}")

        return self.data

    def create_precision_based_grid(self):
        """创建基于精度的网格"""
        print(f"创建基于精度的网格 (网格大小: {self.grid_size}米)")

        # 使用舍入后的坐标
        lon_min = self.data['lon_rounded'].min()
        lon_max = self.data['lon_rounded'].max()
        lat_min = self.data['lat_rounded'].min()
        lat_max = self.data['lat_rounded'].max()

        print(f"舍入后坐标范围:")
        print(f"  经度: {lon_min:.6f} 到 {lon_max:.6f}")
        print(f"  纬度: {lat_min:.6f} 到 {lat_max:.6f}")

        # 转换为平面坐标（近似）
        lat_mid = (lat_min + lat_max) / 2
        cos_lat = cos(radians(lat_mid))

        self.data['easting'] = (self.data['lon_rounded'] - lon_min) * 111000 * cos_lat
        self.data['northing'] = (self.data['lat_rounded'] - lat_min) * 111000

        # 创建网格索引
        min_easting = self.data['easting'].min()
        max_easting = self.data['easting'].max()
        min_northing = self.data['northing'].min()
        max_northing = self.data['northing'].max()

        n_cols = int(np.ceil((max_easting - min_easting) / self.grid_size))
        n_rows = int(np.ceil((max_northing - min_northing) / self.grid_size))

        self.data['col_idx'] = ((self.data['easting'] - min_easting) / self.grid_size).astype(int)
        self.data['row_idx'] = ((self.data['northing'] - min_northing) / self.grid_size).astype(int)

        # 创建网格ID（使用舍入后的坐标）
        self.data['cell_id'] = self.data['row_idx'].astype(str) + '_' + self.data['col_idx'].astype(str)

        # 存储网格信息
        self.grid = {
            'grid_size': self.grid_size,
            'lon_precision': self.lon_precision,
            'lat_precision': self.lat_precision,
            'n_cols': n_cols,
            'n_rows': n_rows,
            'total_cells': n_cols * n_rows,
            'used_cells': self.data['cell_id'].nunique(),
            'min_lon': lon_min,
            'max_lon': lon_max,
            'min_lat': lat_min,
            'max_lat': lat_max
        }

        print(f"网格创建完成:")
        print(f"  网格维度: {n_rows}行 × {n_cols}列 = {n_cols * n_rows:,}个单元")
        print(f"  使用的单元: {self.grid['used_cells']:,}")
        print(f"  网格使用率: {self.grid['used_cells']/(n_cols * n_rows)*100:.2f}%")

        return self.grid

    def extract_od_pairs(self):
        """提取OD对"""
        print("提取OD对...")

        if 'order_id' not in self.data.columns:
            print("警告: 没有order_id列，无法提取OD对")
            return None

        # 按轨迹分组
        trajectories = []
        od_data = []

        for order_id, group in self.data.groupby('order_id'):
            if len(group) >= 2:  # 至少需要起点和终点
                # 按时间排序
                group = group.sort_values('datetime')

                # 获取起点和终点
                start = group.iloc[0]
                end = group.iloc[-1]

                # 使用网格单元作为OD
                start_cell = start['cell_id'] if 'cell_id' in start else None
                end_cell = end['cell_id'] if 'cell_id' in end else None

                if start_cell and end_cell and start_cell != end_cell:
                    od_data.append({
                        'order_id': order_id,
                        'start_cell': start_cell,
                        'end_cell': end_cell,
                        'start_time': start['datetime'],
                        'end_time': end['datetime'],
                        'duration': (end['datetime'] - start['datetime']).total_seconds(),
                        'start_lon': start['lon_rounded'],
                        'start_lat': start['lat_rounded'],
                        'end_lon': end['lon_rounded'],
                        'end_lat': end['lat_rounded']
                    })

                trajectories.append(group)

        self.trajectories = trajectories
        self.od_pairs = pd.DataFrame(od_data)

        print(f"OD对提取完成:")
        print(f"  总轨迹数: {len(trajectories):,}")
        print(f"  OD对数量: {len(self.od_pairs):,}")

        if len(self.od_pairs) > 0:
            # OD对统计
            unique_od = self.od_pairs[['start_cell', 'end_cell']].drop_duplicates()
            print(f"  唯一OD对: {len(unique_od):,}")
            print(f"  平均OD距离: {self.od_pairs['duration'].mean():.1f} 秒")

        return self.od_pairs

    def analyze_od_flow(self):
        """分析OD流"""
        if self.od_pairs is None or len(self.od_pairs) == 0:
            print("没有OD对数据")
            return None

        print("分析OD流...")

        # OD流矩阵
        od_flow = self.od_pairs.groupby(['start_cell', 'end_cell']).size().reset_index(name='flow')

        # 统计
        total_flow = od_flow['flow'].sum()
        unique_od_pairs = len(od_flow)

        print(f"OD流分析:")
        print(f"  总OD流: {total_flow:,}")
        print(f"  唯一OD对: {unique_od_pairs:,}")
        print(f"  平均每OD对流: {total_flow/unique_od_pairs:.1f}")

        # 前10大OD流
        top_od = od_flow.sort_values('flow', ascending=False).head(10)
        print(f"  前10大OD流:")
        for i, row in top_od.iterrows():
            print(f"    {row['start_cell']} -> {row['end_cell']}: {row['flow']:,}")

        return od_flow

    def save_results(self, output_dir='data/processed/grid_refined'):
        """保存结果"""
        print(f"保存结果到 {output_dir}...")

        os.makedirs(output_dir, exist_ok=True)

        # 保存网格化数据
        if self.data is not None:
            grid_data_path = os.path.join(output_dir, 'gridded_data.csv')
            self.data.to_csv(grid_data_path, index=False)
            print(f"  网格化数据: {grid_data_path} ({len(self.data):,} 行)")

        # 保存OD对数据
        if self.od_pairs is not None and len(self.od_pairs) > 0:
            od_path = os.path.join(output_dir, 'od_pairs.csv')
            self.od_pairs.to_csv(od_path, index=False)
            print(f"  OD对数据: {od_path} ({len(self.od_pairs):,} 行)")

        # 保存网格信息
        if self.grid is not None:
            import json
            grid_info_path = os.path.join(output_dir, 'grid_info.json')
            with open(grid_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.grid, f, indent=2, ensure_ascii=False)
            print(f"  网格信息: {grid_info_path}")

        # 生成摘要报告
        self._generate_summary_report(output_dir)

        print("结果保存完成")

    def _generate_summary_report(self, output_dir):
        """生成摘要报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("改进的网格化分析摘要报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        if self.grid is not None:
            report_lines.append("网格参数:")
            report_lines.append("-" * 40)
            report_lines.append(f"  网格大小: {self.grid['grid_size']} 米")
            report_lines.append(f"  经度精度: 小数点后{self.grid['lon_precision']}位")
            report_lines.append(f"  纬度精度: 小数点后{self.grid['lat_precision']}位")
            report_lines.append(f"  网格维度: {self.grid['n_rows']}行 × {self.grid['n_cols']}列")
            report_lines.append(f"  总网格单元: {self.grid['total_cells']:,}")
            report_lines.append(f"  使用的网格单元: {self.grid['used_cells']:,}")
            report_lines.append(f"  网格使用率: {self.grid['used_cells']/self.grid['total_cells']*100:.2f}%")
            report_lines.append("")

        if self.od_pairs is not None and len(self.od_pairs) > 0:
            report_lines.append("OD分析:")
            report_lines.append("-" * 40)
            report_lines.append(f"  OD对数量: {len(self.od_pairs):,}")
            unique_od = self.od_pairs[['start_cell', 'end_cell']].drop_duplicates()
            report_lines.append(f"  唯一OD对: {len(unique_od):,}")
            report_lines.append(f"  平均OD距离: {self.od_pairs['duration'].mean():.1f} 秒")
            report_lines.append("")

        report_lines.append("改进说明:")
        report_lines.append("-" * 40)
        report_lines.append("1. 添加了坐标精度控制，减少浮点数计算误差")
        report_lines.append("2. 实现了OD对提取，支持基于流的分析")
        report_lines.append("3. 改进了网格化逻辑，提高空间分析准确性")

        report_path = os.path.join(output_dir, 'summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"  摘要报告: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate rounded-coordinate OD pairs for the review package.")
    parser.add_argument("--input", default=ROOT / "data" / "processed" / "full_100m" / "cleaned_data.csv", help="Path to cleaned trajectory CSV.")
    parser.add_argument("--output-dir", default=ROOT / "data" / "processed" / "grid_refined_100m", help="Directory for refined grid outputs.")
    parser.add_argument("--grid-size", type=int, default=100, help="Grid size in meters.")
    parser.add_argument("--lon-precision", type=int, default=4, help="Rounded longitude precision.")
    parser.add_argument("--lat-precision", type=int, default=4, help="Rounded latitude precision.")
    args = parser.parse_args()

    print("改进的网格化分析")
    print("=" * 60)

    # 创建改进的网格处理器
    grid_processor = ImprovedSpatialGrid(
        grid_size=args.grid_size,
        lon_precision=args.lon_precision,
        lat_precision=args.lat_precision,
    )

    # 加载数据
    data_file = args.input
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        return False

    try:
        grid_processor.load_data(data_file)

        # 应用坐标精度控制
        grid_processor.apply_coordinate_precision()

        # 创建基于精度的网格
        grid_processor.create_precision_based_grid()

        # 提取OD对
        grid_processor.extract_od_pairs()

        # 分析OD流
        grid_processor.analyze_od_flow()

        # 保存结果
        grid_processor.save_results(args.output_dir)

        print("\n" + "="*60)
        print("改进的网格化分析完成")
        print("主要改进:")
        print("1. 添加了经纬度精度控制（小数点后4位）")
        print("2. 实现了OD对提取和分析")
        print("3. 改进了网格化逻辑")
        print("="*60)

        return True

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

