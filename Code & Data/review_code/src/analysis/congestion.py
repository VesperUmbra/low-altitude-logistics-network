"""
100%完整数据拥挤相变分析
分析密度-速度关系，识别临界密度阈值
"""

import argparse
import gc
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import ruptures as rpt
except ModuleNotFoundError:
    rpt = None

# 添加项目根目录到Python路径
REVIEW_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REVIEW_CODE_ROOT))

warnings.filterwarnings('ignore')

class FullDataCongestionAnalyzer:
    """100%完整数据拥挤分析器"""

    def __init__(
        self,
        spatiotemporal_file=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m" / "grid" / "spatiotemporal_stats.csv",
        output_dir=REVIEW_CODE_ROOT / "data" / "results" / "full_100m" / "diagram",
    ):
        self.spatiotemporal_data = None
        self.fundamental_data = None
        self.breakpoint_results = None
        self.diagram_metrics = None
        self.spatiotemporal_file = Path(spatiotemporal_file)
        self.output_dir = Path(output_dir)

    def load_spatiotemporal_data(self):
        """加载时空统计数据"""
        print("加载时空统计数据...")

        data_file = self.spatiotemporal_file

        if not data_file.exists():
            print(f"错误: 时空统计数据文件不存在: {data_file}")
            return False

        # 读取数据
        file_size = data_file.stat().st_size / (1024**2)  # MB
        print(f"数据文件: {data_file}")
        print(f"文件大小: {file_size:.1f} MB")

        # 分块读取
        chunk_size = 500000
        chunks = []
        total_rows = 0

        for chunk_idx, chunk in enumerate(pd.read_csv(data_file, chunksize=chunk_size)):
            chunks.append(chunk)
            total_rows += len(chunk)

            print(f"  读取块 {chunk_idx + 1}: {len(chunk):,} 行")

            # 测试读取：先读取前2个块
            if chunk_idx >= 1 and len(chunks) * chunk_size >= 1000000:
                print(f"  已读取 {total_rows:,} 行，继续完整读取...")

        # 合并数据
        self.spatiotemporal_data = pd.concat(chunks, ignore_index=True)

        # 解析时间列
        if 'window_start' in self.spatiotemporal_data.columns:
            self.spatiotemporal_data['window_start'] = pd.to_datetime(self.spatiotemporal_data['window_start'])

        print(f"时空数据加载完成: {len(self.spatiotemporal_data):,} 行")

        # 显示基本统计
        print(f"密度范围: {self.spatiotemporal_data['n_points'].min():.0f} - {self.spatiotemporal_data['n_points'].max():.0f}")
        print(f"速度范围: {self.spatiotemporal_data['mean_speed'].min():.2f} - {self.spatiotemporal_data['mean_speed'].max():.2f} m/s")
        print(f"有效速度样本: {self.spatiotemporal_data['mean_speed'].notna().sum():,}")

        return True

    def prepare_diagram(self):
        """准备基本图数据"""
        print("\n准备基本图数据...")

        if self.spatiotemporal_data is None or len(self.spatiotemporal_data) == 0:
            print("错误: 没有时空数据")
            return False

        # 过滤无效数据
        valid_data = self.spatiotemporal_data[
            (self.spatiotemporal_data['n_points'] > 0) &
            (self.spatiotemporal_data['mean_speed'].notna()) &
            (self.spatiotemporal_data['mean_speed'] > 0)
        ].copy()

        print(f"有效数据: {len(valid_data):,} 行")
        print(f"密度范围: {valid_data['n_points'].min():.0f} - {valid_data['n_points'].max():.0f}")

        # 按密度分箱
        max_density = valid_data['n_points'].max()
        print(f"最大密度: {max_density:.0f}")

        # 创建密度分箱
        density_bins = np.arange(0, max_density + 2, 1)  # 1点间隔
        valid_data['density_bin'] = pd.cut(valid_data['n_points'], bins=density_bins, right=False)

        # 计算每个分箱的统计
        binned_data = valid_data.groupby('density_bin').agg({
            'n_points': ['count', 'mean'],
            'mean_speed': ['mean', 'std', 'count']
        }).reset_index()

        # 简化列名
        binned_data.columns = ['density_bin', 'sample_count', 'density_mean', 'speed_mean', 'speed_std', 'speed_count']

        # 计算分箱边界
        density_mins = []
        density_maxs = []

        for interval in binned_data['density_bin']:
            if hasattr(interval, 'left'):
                density_mins.append(float(interval.left))
                density_maxs.append(float(interval.right))
            else:
                # 如果不是区间，假设是数值
                val = float(interval)
                density_mins.append(val)
                density_maxs.append(val)

        binned_data['density_min'] = density_mins
        binned_data['density_max'] = density_maxs
        binned_data['density_center'] = (binned_data['density_min'] + binned_data['density_max']) / 2

        # 过滤样本量不足的分箱
        min_samples = 10
        binned_data = binned_data[binned_data['sample_count'] >= min_samples]

        print(f"分箱后数据: {len(binned_data):,} 个分箱")
        print(f"密度范围: {binned_data['density_center'].min():.1f} - {binned_data['density_center'].max():.1f}")

        self.fundamental_data = binned_data

        return True

    def estimate_breakpoint_piecewise(self):
        """使用分段回归估计临界点"""
        print("\n使用分段回归估计临界点...")

        if self.fundamental_data is None or len(self.fundamental_data) < 10:
            print("错误: 基本图数据不足")
            return None

        data = self.fundamental_data.copy()
        X = data['density_center'].values.reshape(-1, 1)
        y = data['speed_mean'].values

        best_breakpoint = None
        best_rss = float('inf')
        results = []
        best_result = None

        # 尝试不同的断点
        min_breakpoint = 1
        max_breakpoint = min(len(data) - 5, 50)  # 限制搜索范围

        print(f"搜索断点范围: {min_breakpoint} - {max_breakpoint}")

        for breakpoint_idx in range(min_breakpoint, max_breakpoint):
            # 分段回归
            X1 = X[:breakpoint_idx]
            y1 = y[:breakpoint_idx]
            X2 = X[breakpoint_idx:]
            y2 = y[breakpoint_idx:]

            if len(X1) < 3 or len(X2) < 3:
                continue

            # 拟合第一段
            coef1, intercept1 = np.polyfit(X1.ravel(), y1, 1)
            y1_pred = coef1 * X1.ravel() + intercept1

            # 拟合第二段
            coef2, intercept2 = np.polyfit(X2.ravel(), y2, 1)
            y2_pred = coef2 * X2.ravel() + intercept2

            # 计算总残差平方和
            rss = np.sum((y1 - y1_pred) ** 2) + np.sum((y2 - y2_pred) ** 2)

            # 记录结果
            breakpoint_density = data.iloc[breakpoint_idx]['density_center']
            speed_before = y[breakpoint_idx - 1] if breakpoint_idx > 0 else np.nan
            speed_after = y[breakpoint_idx] if breakpoint_idx < len(y) else np.nan
            speed_drop = ((speed_before - speed_after) / speed_before * 100) if speed_before > 0 else np.nan

            results.append({
                'breakpoint_idx': breakpoint_idx,
                'breakpoint_density': breakpoint_density,
                'rss': rss,
                'slope1': float(coef1),
                'intercept1': float(intercept1),
                'slope2': float(coef2),
                'intercept2': float(intercept2),
                'speed_before': speed_before,
                'speed_after': speed_after,
                'speed_drop_percent': speed_drop
            })

            # 更新最佳断点
            if rss < best_rss:
                best_rss = rss
                best_breakpoint = breakpoint_idx
                best_result = results[-1]

        if best_breakpoint is not None and best_result is not None:
            print(f"最佳断点找到:")
            print(f"  密度阈值: {best_result['breakpoint_density']:.2f}")
            print(f"  残差平方和: {best_result['rss']:.4f}")
            print(f"  第一段斜率: {best_result['slope1']:.4f}")
            print(f"  第二段斜率: {best_result['slope2']:.4f}")
            print(f"  速度下降: {best_result['speed_drop_percent']:.1f}%")

            return best_result
        else:
            print("未找到有效断点")
            return None

    def estimate_breakpoint_changepoint(self):
        """使用变化点检测估计临界点"""
        print("\n使用变化点检测估计临界点...")

        if self.fundamental_data is None or len(self.fundamental_data) < 10:
            print("错误: 基本图数据不足")
            return None

        data = self.fundamental_data.copy()

        # 准备数据
        y = data['speed_mean'].values
        n_samples = len(y)

        # 使用ruptures库检测变化点
        if rpt is None:
            print("变化点检测依赖 ruptures，当前环境缺失，跳过该方法")
            return None

        try:
            # 使用PELT算法
            algo = rpt.Pelt(model="rbf").fit(y.reshape(-1, 1))
            result = algo.predict(pen=10)  # 惩罚参数

            if len(result) > 1:
                # 第一个变化点（排除最后一个）
                change_point_idx = result[0] if result[0] < n_samples else n_samples - 1

                if change_point_idx > 0 and change_point_idx < n_samples:
                    change_point_density = data.iloc[change_point_idx]['density_center']
                    speed_before = y[change_point_idx - 1]
                    speed_after = y[change_point_idx]
                    speed_drop = ((speed_before - speed_after) / speed_before * 100) if speed_before > 0 else np.nan

                    result = {
                        'method': 'changepoint_pelt',
                        'change_point_idx': change_point_idx,
                        'change_point_density': change_point_density,
                        'speed_before': speed_before,
                        'speed_after': speed_after,
                        'speed_drop_percent': speed_drop,
                        'n_change_points': len(result) - 1
                    }

                    print(f"变化点检测结果:")
                    print(f"  密度阈值: {result['change_point_density']:.2f}")
                    print(f"  速度下降: {result['speed_drop_percent']:.1f}%")
                    print(f"  检测到的变化点数: {result['n_change_points']}")

                    return result

        except Exception as e:
            print(f"变化点检测错误: {e}")

        print("变化点检测未找到有效结果")
        return None

    def estimate_breakpoint_inflection(self):
        """使用拐点分析估计临界点"""
        print("\n使用拐点分析估计临界点...")

        if self.fundamental_data is None or len(self.fundamental_data) < 10:
            print("错误: 基本图数据不足")
            return None

        data = self.fundamental_data.copy()

        # 计算速度的一阶和二阶导数（数值差分）
        x = data['density_center'].values
        y = data['speed_mean'].values

        # 一阶导数
        dy = np.gradient(y, x)

        # 二阶导数
        d2y = np.gradient(dy, x)

        # 找到二阶导数的最大值（最大曲率点）
        # 只考虑前2/3的数据，避免边缘效应
        search_limit = int(len(x) * 2 / 3)
        if search_limit < 5:
            search_limit = len(x)

        valid_indices = np.where((x[:search_limit] > 0) & (np.isfinite(d2y[:search_limit])))[0]

        if len(valid_indices) > 0:
            # 找到二阶导数的最大值
            max_d2y_idx = valid_indices[np.argmax(d2y[valid_indices])]

            inflection_density = x[max_d2y_idx]
            speed_before = y[max_d2y_idx - 1] if max_d2y_idx > 0 else y[max_d2y_idx]
            speed_after = y[max_d2y_idx + 1] if max_d2y_idx < len(y) - 1 else y[max_d2y_idx]
            speed_drop = ((speed_before - speed_after) / speed_before * 100) if speed_before > 0 else np.nan

            result = {
                'method': 'inflection',
                'inflection_idx': max_d2y_idx,
                'inflection_density': inflection_density,
                'd2y_max': d2y[max_d2y_idx],
                'speed_before': speed_before,
                'speed_after': speed_after,
                'speed_drop_percent': speed_drop
            }

            print(f"拐点分析结果:")
            print(f"  密度阈值: {result['inflection_density']:.2f}")
            print(f"  最大二阶导数: {result['d2y_max']:.4f}")
            print(f"  速度下降: {result['speed_drop_percent']:.1f}%")

            return result
        else:
            print("拐点分析未找到有效结果")
            return None

    def analyze_congestion_metrics(self):
        """分析拥挤指标"""
        print("\n分析拥挤指标...")

        if self.spatiotemporal_data is None or len(self.spatiotemporal_data) == 0:
            print("错误: 没有时空数据")
            return False

        # 使用三种方法估计临界密度
        piecewise_result = self.estimate_breakpoint_piecewise()
        changepoint_result = self.estimate_breakpoint_changepoint()
        inflection_result = self.estimate_breakpoint_inflection()

        # 收集所有估计值
        estimates = []

        if piecewise_result:
            estimates.append({
                'method': 'piecewise_regression',
                'rho_star': piecewise_result['breakpoint_density'],
                'speed_drop': piecewise_result['speed_drop_percent']
            })

        if changepoint_result:
            estimates.append({
                'method': 'changepoint_detection',
                'rho_star': changepoint_result['change_point_density'],
                'speed_drop': changepoint_result['speed_drop_percent']
            })

        if inflection_result:
            estimates.append({
                'method': 'inflection_analysis',
                'rho_star': inflection_result['inflection_density'],
                'speed_drop': inflection_result['speed_drop_percent']
            })

        # 主文口径以 piecewise 断点为 operational threshold；其余方法保留为敏感性参考。
        if estimates:
            if piecewise_result is not None:
                consensus_rho_star = float(piecewise_result['breakpoint_density'])
            else:
                rho_star_values = [e['rho_star'] for e in estimates]
                consensus_rho_star = float(np.median(rho_star_values))

            # 计算速度-密度相关性
            valid_data = self.spatiotemporal_data[
                (self.spatiotemporal_data['n_points'] > 0) &
                (self.spatiotemporal_data['mean_speed'].notna())
            ]

            if len(valid_data) > 10:
                correlation, p_value = stats.pearsonr(
                    valid_data['n_points'],
                    valid_data['mean_speed']
                )
            else:
                correlation, p_value = np.nan, np.nan

            # 计算拥挤发生率
            congestion_mask = valid_data['n_points'] >= consensus_rho_star
            congestion_share = congestion_mask.mean() * 100

            # 保存结果
            self.breakpoint_results = {
                'estimates': estimates,
                'consensus_rho_star': float(consensus_rho_star),
                'correlation_coefficient': float(correlation),
                'correlation_p_value': float(p_value),
                'congestion_share_percent': float(congestion_share),
                'n_samples_total': int(len(valid_data)),
                'n_congested_samples': int(congestion_mask.sum()),
                'piecewise_settings': {
                    'input': 'binned_speed_density_means',
                    'bin_width_points': 1,
                    'min_bin_samples': 10,
                    'weighting': 'equal_weight_bins'
                },
                'analysis_time': datetime.now().isoformat()
            }

            print("\n拥挤分析结果汇总:")
            print(f"  共识临界密度 ρ*: {consensus_rho_star:.2f}")
            print(f"  速度-密度相关性: {correlation:.3f} (p={p_value:.3e})")
            print(f"  拥挤发生率: {congestion_share:.2f}%")
            print(f"  总样本数: {len(valid_data):,}")
            print(f"  拥挤样本数: {congestion_mask.sum():,}")

            # 各方法结果
            print("\n  各方法估计值:")
            for est in estimates:
                print(f"    {est['method']}: ρ*={est['rho_star']:.2f}, 速度下降={est['speed_drop']:.1f}%")

            return True
        else:
            print("错误: 所有方法都未能估计临界密度")
            return False

    def calculate_diagram_metrics(self):
        """计算基本图指标"""
        print("\n计算基本图指标...")

        if self.fundamental_data is None or len(self.fundamental_data) == 0:
            print("错误: 没有基本图数据")
            return False

        if self.breakpoint_results is None:
            print("错误: 没有断点分析结果")
            return False

        data = self.fundamental_data.copy()
        rho_star = self.breakpoint_results['consensus_rho_star']

        # 分离自由流和拥挤流
        free_flow = data[data['density_center'] < rho_star]
        congested_flow = data[data['density_center'] >= rho_star]

        # 计算指标
        metrics = {
            'rho_star': float(rho_star),
            'free_flow_samples': int(len(free_flow)),
            'congested_flow_samples': int(len(congested_flow)),
            'free_flow_density_range': [float(free_flow['density_center'].min()), float(free_flow['density_center'].max())],
            'congested_flow_density_range': [float(congested_flow['density_center'].min()), float(congested_flow['density_center'].max())],
            'free_flow_speed_mean': float(free_flow['speed_mean'].mean()),
            'congested_flow_speed_mean': float(congested_flow['speed_mean'].mean()),
            'free_flow_speed_std': float(free_flow['speed_mean'].std()),
            'congested_flow_speed_std': float(congested_flow['speed_mean'].std()),
            'speed_drop_at_rho_star': float(0.0),  # 需要计算
            'free_flow_slope': float(0.0),  # 需要计算
            'congested_flow_slope': float(0.0)  # 需要计算
        }

        # 计算速度下降
        if len(free_flow) > 0 and len(congested_flow) > 0:
            # 找到rho_star附近的速度
            near_rho_star = data[
                (data['density_center'] >= rho_star - 1) &
                (data['density_center'] <= rho_star + 1)
            ]

            if len(near_rho_star) >= 2:
                # 排序并获取前后的速度
                near_rho_star = near_rho_star.sort_values('density_center')
                pre_speed = near_rho_star[near_rho_star['density_center'] < rho_star]['speed_mean'].iloc[-1] if len(near_rho_star[near_rho_star['density_center'] < rho_star]) > 0 else np.nan
                post_speed = near_rho_star[near_rho_star['density_center'] >= rho_star]['speed_mean'].iloc[0] if len(near_rho_star[near_rho_star['density_center'] >= rho_star]) > 0 else np.nan

                if not np.isnan(pre_speed) and not np.isnan(post_speed) and pre_speed > 0:
                    speed_drop = (pre_speed - post_speed) / pre_speed * 100
                    metrics['speed_drop_at_rho_star'] = float(speed_drop)

        # 计算斜率
        if len(free_flow) >= 3:
            X = free_flow['density_center'].values.reshape(-1, 1)
            y = free_flow['speed_mean'].values
            coef, _ = np.polyfit(X.ravel(), y, 1)
            metrics['free_flow_slope'] = float(coef)

        if len(congested_flow) >= 3:
            X = congested_flow['density_center'].values.reshape(-1, 1)
            y = congested_flow['speed_mean'].values
            coef, _ = np.polyfit(X.ravel(), y, 1)
            metrics['congested_flow_slope'] = float(coef)

        self.diagram_metrics = metrics

        print("基本图指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list):
                print(f"  {key}: [{value[0]:.2f}, {value[1]:.2f}]")
            else:
                print(f"  {key}: {value}")

        return True

    def save_results(self):
        """保存结果"""
        print("\n保存拥挤分析结果...")

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存基本图数据
        if self.fundamental_data is not None:
            data_file = output_dir / "fundamental_data.csv"
            self.fundamental_data.to_csv(data_file, index=False)
            print(f"基本图数据已保存: {data_file} ({len(self.fundamental_data):,} 行)")

        # 2. 保存断点分析结果
        if self.breakpoint_results is not None:
            results_file = output_dir / "breakpoint_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.breakpoint_results, f, indent=2, ensure_ascii=False)
            print(f"断点分析结果已保存: {results_file}")

        # 3. 保存基本图指标
        if self.diagram_metrics is not None:
            metrics_file = output_dir / "diagram_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.diagram_metrics, f, indent=2, ensure_ascii=False)
            print(f"基本图指标已保存: {metrics_file}")

        # 4. 生成摘要报告
        self._generate_summary_report(output_dir)

        print("所有拥挤分析结果已保存")

    def _generate_summary_report(self, output_dir):
        """生成摘要报告"""
        report_file = Path(output_dir) / "diagram_summary.txt"

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("100%完整数据拥挤相变分析摘要报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 基本统计
        if self.spatiotemporal_data is not None:
            report_lines.append("基本统计:")
            report_lines.append(f"  时空样本总数: {len(self.spatiotemporal_data):,}")
            report_lines.append(f"  有效速度样本: {self.spatiotemporal_data['mean_speed'].notna().sum():,}")
            report_lines.append(f"  密度范围: {self.spatiotemporal_data['n_points'].min():.0f} - {self.spatiotemporal_data['n_points'].max():.0f}")
            report_lines.append(f"  速度范围: {self.spatiotemporal_data['mean_speed'].min():.2f} - {self.spatiotemporal_data['mean_speed'].max():.2f} m/s")
            report_lines.append("")

        # 临界密度结果
        if self.breakpoint_results is not None:
            report_lines.append("临界密度分析:")
            report_lines.append(f"  共识临界密度 ρ*: {self.breakpoint_results['consensus_rho_star']:.2f}")
            report_lines.append(f"  速度-密度相关性: {self.breakpoint_results['correlation_coefficient']:.3f} (p={self.breakpoint_results['correlation_p_value']:.3e})")
            report_lines.append(f"  拥挤发生率: {self.breakpoint_results['congestion_share_percent']:.2f}%")
            report_lines.append(f"  拥挤样本数: {self.breakpoint_results['n_congested_samples']:,}")
            report_lines.append("")

            report_lines.append("各方法估计值:")
            for est in self.breakpoint_results['estimates']:
                report_lines.append(f"  {est['method']}: ρ*={est['rho_star']:.2f}, 速度下降={est.get('speed_drop', 'N/A'):.1f}%")
            report_lines.append("")

        # 基本图指标
        if self.diagram_metrics is not None:
            report_lines.append("基本图指标:")
            report_lines.append(f"  自由流样本数: {self.diagram_metrics['free_flow_samples']:,}")
            report_lines.append(f"  拥挤流样本数: {self.diagram_metrics['congested_flow_samples']:,}")
            report_lines.append(f"  自由流平均速度: {self.diagram_metrics['free_flow_speed_mean']:.2f} m/s")
            report_lines.append(f"  拥挤流平均速度: {self.diagram_metrics['congested_flow_speed_mean']:.2f} m/s")
            report_lines.append(f"  临界点速度下降: {self.diagram_metrics['speed_drop_at_rho_star']:.1f}%")
            report_lines.append(f"  自由流斜率: {self.diagram_metrics['free_flow_slope']:.4f}")
            report_lines.append(f"  拥挤流斜率: {self.diagram_metrics['congested_flow_slope']:.4f}")
            report_lines.append("")

        # 解释
        report_lines.append("解释:")
        if self.breakpoint_results is not None:
            rho_star = self.breakpoint_results['consensus_rho_star']
            report_lines.append(f"  1. 临界密度 ρ*={rho_star:.2f} 表示当局部密度超过此阈值时，")
            report_lines.append(f"     系统从自由流状态进入拥挤状态")
            report_lines.append(f"  2. 速度-密度负相关 ({self.breakpoint_results['correlation_coefficient']:.3f}) 表明")
            report_lines.append(f"     密度增加会导致速度下降")
            report_lines.append(f"  3. {self.breakpoint_results['congestion_share_percent']:.1f}% 的样本处于拥挤状态，")
            report_lines.append(f"     表明当前运营频繁接近不稳定边界")

        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"摘要报告已保存: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Estimate the breakpoint on gridded spatiotemporal data.")
    parser.add_argument("--input", default=REVIEW_CODE_ROOT / "data" / "processed" / "full_100m" / "grid" / "spatiotemporal_stats.csv", help="Path to spatiotemporal stats CSV.")
    parser.add_argument("--output-dir", default=REVIEW_CODE_ROOT / "data" / "results" / "full_100m" / "diagram", help="Directory for breakpoint outputs.")
    args = parser.parse_args()

    print("100%完整数据拥挤相变分析")
    print("=" * 60)

    analyzer = FullDataCongestionAnalyzer(spatiotemporal_file=args.input, output_dir=args.output_dir)

    try:
        # 1. 加载时空数据
        if not analyzer.load_spatiotemporal_data():
            print("加载数据失败")
            return False

        # 2. 准备基本图数据
        if not analyzer.prepare_diagram():
            print("准备基本图数据失败")
            return False

        # 3. 分析拥挤指标
        if not analyzer.analyze_congestion_metrics():
            print("分析拥挤指标失败")
            return False

        # 4. 计算基本图指标
        if not analyzer.calculate_diagram_metrics():
            print("计算基本图指标失败")
            return False

        # 5. 保存结果
        analyzer.save_results()

        print("\n" + "=" * 60)
        print("拥挤相变分析成功完成!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

