#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现FL-market.pdf论文中的精度随迭代步骤变化图表
基于已有的RQ4实验数据生成高质量的论文风格图表

Usage:
    python reproduce_fl_accuracy.py
"""
from __future__ import division, print_function

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def load_rq4_data(json_path):
    """加载RQ4聚合数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_accuracy_comparison(data, output_dir='figures', figsize=(16, 5)):
    """
    绘制精度对比图：4个子图展示不同数据集和alpha设置
    
    图A: MNIST α=0.1  |  图B: MNIST α=0.5
    图C: CIFAR-10 α=0.1  |  图D: CIFAR-10 α=0.5
    """
    
    # 定义方法颜色和样式（与论文保持一致）
    method_styles = {
        'Ours': {'color': '#1b5e20', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'o', 'markersize': 5},
        'CSRA': {'color': '#1565c0', 'linestyle': '-', 'linewidth': 2.0, 'marker': 's', 'markersize': 5},
        'MFG-Pricing': {'color': '#6a1b9a', 'linestyle': '-', 'linewidth': 2.0, 'marker': '^', 'markersize': 5},
        'PAC': {'color': '#00838f', 'linestyle': '-', 'linewidth': 2.0, 'marker': 'D', 'markersize': 4},
        'RegretNet': {'color': '#e65100', 'linestyle': '--', 'linewidth': 2.0, 'marker': 'v', 'markersize': 5},
        'Uniform-DP': {'color': '#757575', 'linestyle': '-.', 'linewidth': 2.0, 'marker': 'x', 'markersize': 6},
        'No-DP (upper)': {'color': '#000000', 'linestyle': ':', 'linewidth': 2.5, 'marker': None, 'markersize': 0, 'alpha': 0.5}
    }
    
    # 方法显示顺序
    method_order = ['Ours', 'CSRA', 'MFG-Pricing', 'PAC', 'RegretNet', 'Uniform-DP', 'No-DP (upper)']
    
    # 创建4子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('FL-Market: Test Accuracy vs Training Rounds', fontsize=18, fontweight='bold', y=0.995)
    
    configs = [
        (("MNIST", 0.1), axes[0, 0], "MNIST (α=0.1, high heterogeneity)"),
        (("MNIST", 0.5), axes[0, 1], "MNIST (α=0.5, moderate heterogeneity)"),
        (("CIFAR10", 0.1), axes[1, 0], "CIFAR-10 (α=0.1, high heterogeneity)"),
        (("CIFAR10", 0.5), axes[1, 1], "CIFAR-10 (α=0.5, moderate heterogeneity)"),
    ]
    
    for (dataset_key, ax, title) in configs:
        if dataset_key not in data:
            print(f"Warning: {dataset_key} not found in data")
            continue
        
        block = data[dataset_key]
        rounds = np.array(block['rounds'])
        methods = block['methods']
        n_seeds = block.get('n_seeds', 3)
        
        # 绘制每个方法的曲线
        for method_name in method_order:
            if method_name not in methods:
                continue
            
            method_data = methods[method_name]
            mean_acc = np.array(method_data['test_acc_mean'])
            std_acc = np.array(method_data['test_acc_std'])
            
            if len(mean_acc) == 0:
                continue
            
            # 确保长度一致
            min_len = min(len(rounds), len(mean_acc))
            rounds_plot = rounds[:min_len]
            mean_plot = mean_acc[:min_len]
            std_plot = std_acc[:min_len]
            
            style = method_styles.get(method_name, {})
            
            # 绘制均值线
            line, = ax.plot(rounds_plot, mean_plot, 
                          label=method_name,
                          color=style.get('color', '#333333'),
                          linestyle=style.get('linestyle', '-'),
                          linewidth=style.get('linewidth', 2.0),
                          marker=style.get('marker'),
                          markersize=style.get('markersize', 5),
                          markevery=max(1, len(rounds_plot) // 8),  # 每8个点显示一个marker
                          alpha=style.get('alpha', 1.0),
                          zorder=10 if method_name == 'Ours' else 5)
            
            # 添加标准差阴影区域（除了No-DP上界）
            if method_name != 'No-DP (upper)' and n_seeds > 1:
                ax.fill_between(rounds_plot, 
                              mean_plot - std_plot, 
                              mean_plot + std_plot,
                              color=style.get('color', '#333333'),
                              alpha=0.15,
                              zorder=1)
        
        # 设置坐标轴
        ax.set_xlabel('Training Round', fontsize=13)
        ax.set_ylabel('Test Accuracy', fontsize=13)
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=0)
        
        # Y轴范围设置（根据数据自动调整，但保证最小0.05的range）
        if len(methods) > 0:
            all_means = []
            for m in methods.values():
                if len(m['test_acc_mean']) > 0:
                    all_means.extend(m['test_acc_mean'])
            if all_means:
                y_min = max(0, min(all_means) - 0.02)
                y_max = min(1, max(all_means) + 0.02)
                if y_max - y_min < 0.05:
                    y_mid = (y_max + y_min) / 2
                    y_min = max(0, y_mid - 0.025)
                    y_max = min(1, y_mid + 0.025)
                ax.set_ylim(y_min, y_max)
        
        # 添加图例（只在右上角添加一次）
        if ax == axes[0, 1]:
            ax.legend(loc='lower right', framealpha=0.9, 
                     edgecolor='gray', fancybox=True, 
                     ncol=1, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fl_accuracy_comparison_4panels.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved 4-panel comparison figure: {output_path}")
    
    plt.close()


def plot_single_dataset_comparison(data, dataset='MNIST', alpha=0.5, output_dir='figures'):
    """
    为单个数据集配置绘制独立的大图
    """
    method_styles = {
        'Ours': {'color': '#1b5e20', 'linestyle': '-', 'linewidth': 3.0, 'marker': 'o', 'markersize': 7},
        'CSRA': {'color': '#1565c0', 'linestyle': '-', 'linewidth': 2.5, 'marker': 's', 'markersize': 6},
        'MFG-Pricing': {'color': '#6a1b9a', 'linestyle': '-', 'linewidth': 2.5, 'marker': '^', 'markersize': 6},
        'PAC': {'color': '#00838f', 'linestyle': '-', 'linewidth': 2.5, 'marker': 'D', 'markersize': 5},
        'RegretNet': {'color': '#e65100', 'linestyle': '--', 'linewidth': 2.5, 'marker': 'v', 'markersize': 6},
        'Uniform-DP': {'color': '#757575', 'linestyle': '-.', 'linewidth': 2.5, 'marker': 'x', 'markersize': 7},
        'No-DP (upper)': {'color': '#000000', 'linestyle': ':', 'linewidth': 3.0, 'marker': None, 'alpha': 0.6}
    }
    
    method_order = ['Ours', 'CSRA', 'MFG-Pricing', 'PAC', 'RegretNet', 'Uniform-DP', 'No-DP (upper)']
    
    dataset_key = (dataset, alpha)
    if dataset_key not in data:
        print(f"Error: {dataset_key} not found in data")
        return
    
    block = data[dataset_key]
    rounds = np.array(block['rounds'])
    methods = block['methods']
    n_seeds = block.get('n_seeds', 3)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for method_name in method_order:
        if method_name not in methods:
            continue
        
        method_data = methods[method_name]
        mean_acc = np.array(method_data['test_acc_mean'])
        std_acc = np.array(method_data['test_acc_std'])
        
        if len(mean_acc) == 0:
            continue
        
        min_len = min(len(rounds), len(mean_acc))
        rounds_plot = rounds[:min_len]
        mean_plot = mean_acc[:min_len]
        std_plot = std_acc[:min_len]
        
        style = method_styles.get(method_name, {})
        
        ax.plot(rounds_plot, mean_plot, 
               label=method_name,
               color=style.get('color', '#333333'),
               linestyle=style.get('linestyle', '-'),
               linewidth=style.get('linewidth', 2.5),
               marker=style.get('marker'),
               markersize=style.get('markersize', 6),
               markevery=max(1, len(rounds_plot) // 10),
               alpha=style.get('alpha', 1.0),
               zorder=10 if method_name == 'Ours' else 5)
        
        if method_name != 'No-DP (upper)' and n_seeds > 1:
            ax.fill_between(rounds_plot, 
                          mean_plot - std_plot, 
                          mean_plot + std_plot,
                          color=style.get('color', '#333333'),
                          alpha=0.2,
                          zorder=1)
    
    ax.set_xlabel('Training Round', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    title = f'{dataset} FL Convergence (Dirichlet α={alpha})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray', 
             fancybox=True, fontsize=12)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'fl_accuracy_{dataset}_alpha{alpha}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {dataset} α={alpha} figure: {output_path}")
    
    plt.close()


def plot_final_accuracy_bar_chart(data, output_dir='figures'):
    """
    绘制最终精度对比柱状图（取最后一轮的精度）
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Final Test Accuracy Comparison', fontsize=16, fontweight='bold')
    
    datasets = ['MNIST', 'CIFAR10']
    alphas = [0.1, 0.5]
    method_order = ['Ours', 'CSRA', 'MFG-Pricing', 'PAC', 'RegretNet', 'Uniform-DP']
    
    colors_map = {
        'Ours': '#1b5e20',
        'CSRA': '#1565c0',
        'MFG-Pricing': '#6a1b9a',
        'PAC': '#00838f',
        'RegretNet': '#e65100',
        'Uniform-DP': '#757575',
    }
    
    for ax_i, dataset in enumerate(datasets):
        ax = axes[ax_i]
        
        # 收集每个alpha下每个方法的最终精度
        final_accs = {alpha: {} for alpha in alphas}
        
        for alpha in alphas:
            key = (dataset, alpha)
            if key not in data:
                continue
            block = data[key]
            methods = block['methods']
            
            for method_name in method_order:
                if method_name not in methods:
                    continue
                mean_acc = methods[method_name]['test_acc_mean']
                if len(mean_acc) > 0:
                    final_accs[alpha][method_name] = mean_acc[-1]
        
        # 绘制分组柱状图
        x = np.arange(len(method_order))
        width = 0.35
        
        alpha1_vals = [final_accs[0.1].get(m, 0) for m in method_order]
        alpha2_vals = [final_accs[0.5].get(m, 0) for m in method_order]
        
        ax.bar(x - width/2, alpha1_vals, width, label='α=0.1 (high heterogeneity)', 
               color='#d32f2f', alpha=0.8)
        ax.bar(x + width/2, alpha2_vals, width, label='α=0.5 (moderate heterogeneity)', 
               color='#1976d2', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Final Test Accuracy', fontsize=12)
        ax.set_title(f'{dataset}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=30, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(max(alpha1_vals), max(alpha2_vals)) * 1.1)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'final_accuracy_comparison_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved final accuracy bar chart: {output_path}")
    
    plt.close()


def print_statistics(data):
    """打印统计信息"""
    print("\n" + "="*80)
    print("实验数据统计摘要".center(80))
    print("="*80)
    
    for key in sorted(data.keys()):
        dataset, alpha = key
        block = data[key]
        n_seeds = block.get('n_seeds', 0)
        rounds = block.get('rounds', [])
        methods = block['methods']
        
        print(f"\n【{dataset}, α={alpha}】")
        print(f"  - Seeds数量: {n_seeds}")
        print(f"  - 训练轮数: {len(rounds)} ({rounds[0] if rounds else 0} ~ {rounds[-1] if rounds else 0})")
        print(f"  - 方法数量: {len(methods)}")
        print(f"  - 最终精度 (Round {rounds[-1] if rounds else 0}):")
        
        for method_name in sorted(methods.keys()):
            method_data = methods[method_name]
            mean_acc = method_data['test_acc_mean']
            std_acc = method_data['test_acc_std']
            if len(mean_acc) > 0:
                final_mean = mean_acc[-1]
                final_std = std_acc[-1] if len(std_acc) > 0 else 0
                print(f"      {method_name:20s}: {final_mean:.4f} ± {final_std:.4f}")


def main():
    """主函数：生成所有图表"""
    
    # 数据路径
    rq4_json = "run/privacy_paper/rq4/rq4_aggregated.json"
    output_dir = "run/privacy_paper/figures"
    
    print("="*80)
    print("FL-Market 论文实验复现：精度随迭代步骤变化".center(80))
    print("="*80)
    
    # 检查数据文件
    if not os.path.exists(rq4_json):
        print(f"\n❌ 错误: 找不到数据文件 {rq4_json}")
        print("\n请先运行 RQ4 实验生成数据:")
        print("  bash scripts/run_rq4_paper.sh")
        print("或:")
        print("  python exp_rq/rq4_fl_benchmark.py --dataset MNIST --alpha 0.5 --seed 0")
        return 1
    
    # 加载数据
    print(f"\n📂 加载数据: {rq4_json}")
    data = load_rq4_data(rq4_json)
    
    # 打印统计信息
    print_statistics(data)
    
    # 生成图表
    print("\n" + "="*80)
    print("开始生成图表...".center(80))
    print("="*80 + "\n")
    
    # 1. 四子图对比（主图）
    print("📊 [1/6] 生成4子图对比图...")
    plot_accuracy_comparison(data, output_dir)
    
    # 2-5. 每个配置的独立大图
    configs = [
        ('MNIST', 0.1),
        ('MNIST', 0.5),
        ('CIFAR10', 0.1),
        ('CIFAR10', 0.5),
    ]
    for i, (ds, alpha) in enumerate(configs, start=2):
        print(f"📊 [{i}/6] 生成 {ds} α={alpha} 独立图...")
        plot_single_dataset_comparison(data, ds, alpha, output_dir)
    
    # 6. 最终精度柱状图
    print(f"📊 [6/6] 生成最终精度柱状图...")
    plot_final_accuracy_bar_chart(data, output_dir)
    
    print("\n" + "="*80)
    print("✅ 所有图表生成完成！".center(80))
    print("="*80)
    print(f"\n输出目录: {output_dir}/")
    print("\n生成的图表:")
    print("  1. fl_accuracy_comparison_4panels.png - 4子图对比（主图）")
    print("  2. fl_accuracy_MNIST_alpha0.1.png - MNIST α=0.1")
    print("  3. fl_accuracy_MNIST_alpha0.5.png - MNIST α=0.5")
    print("  4. fl_accuracy_CIFAR10_alpha0.1.png - CIFAR-10 α=0.1")
    print("  5. fl_accuracy_CIFAR10_alpha0.5.png - CIFAR-10 α=0.5")
    print("  6. final_accuracy_comparison_bar.png - 最终精度对比")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
