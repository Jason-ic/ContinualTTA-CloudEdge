#!/usr/bin/env python3
"""
可视化训练损失曲线
用法: python plot_metrics.py
"""
import json
import matplotlib.pyplot as plt
import os

def plot_training_metrics(metrics_file='outputs/coco_base_r50/metrics.json'):
    """从 JSON 文件中读取指标并绘制曲线"""

    # 读取 metrics.json
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))

    if not metrics:
        print("No metrics found!")
        return

    # 提取数据
    iterations = []
    total_loss = []
    loss_cls = []
    loss_box_reg = []
    loss_rpn_cls = []
    loss_rpn_loc = []
    learning_rate = []

    for m in metrics:
        if 'iteration' in m and 'total_loss' in m:
            iterations.append(m['iteration'])
            total_loss.append(m['total_loss'])
            if 'loss_cls' in m:
                loss_cls.append(m['loss_cls'])
            if 'loss_box_reg' in m:
                loss_box_reg.append(m['loss_box_reg'])
            if 'loss_rpn_cls' in m:
                loss_rpn_cls.append(m['loss_rpn_cls'])
            if 'loss_rpn_loc' in m:
                loss_rpn_loc.append(m['loss_rpn_loc'])
            if 'lr' in m:
                learning_rate.append(m['lr'])

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics (Total: {max(iterations):,} iterations)', fontsize=16)

    # 1. 总损失
    axes[0, 0].plot(iterations, total_loss, linewidth=2, label='Total Loss', color='blue')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 分类损失和边界框回归损失
    axes[0, 1].plot(iterations, loss_cls, linewidth=2, label='Classification Loss', color='red')
    axes[0, 1].plot(iterations, loss_box_reg, linewidth=2, label='BBox Regression Loss', color='green')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. RPN 损失
    axes[1, 0].plot(iterations, loss_rpn_cls, linewidth=2, label='RPN Classification', color='orange')
    axes[1, 0].plot(iterations, loss_rpn_loc, linewidth=2, label='RPN Localization', color='purple')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('RPN Losses')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 学习率
    axes[1, 1].plot(iterations, learning_rate, linewidth=2, label='Learning Rate', color='brown')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()

    # 保存图像
    output_file = metrics_file.replace('.json', '_loss_curves.png')
    plt.savefig(output_file, dpi=150)
    print(f"Loss curves saved to: {output_file}")

    # 显示图像（如果在交互式环境中）
    try:
        plt.show()
    except:
        pass

if __name__ == '__main__':
    plot_training_metrics()
