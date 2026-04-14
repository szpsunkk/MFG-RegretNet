#!/bin/bash
# 一键重新训练MFG-RegretNet并运行RQ3实验

set -e  # 出错立即停止

cd /home/skk/FL/market/FL-Market

echo "================================================================================"
echo "MFG-RegretNet 完整优化流程"
echo "================================================================================"
echo ""

# ============================================================================
# 第一步：验证代码修复
# ============================================================================

echo "【第一步】验证代码修复"
echo "----------------------------------------------------------------------"

# 检查utils.py的修改
if grep -q "plosses = torch.sum(allocs \* items, dim=2)" utils.py; then
    echo "✓ utils.py 第181行已正确保留"
else
    echo "✗ 错误：utils.py 第181行缺失"
    exit 1
fi

if grep -q "plosses = torch.clamp(plosses, max=pbudgets)" utils.py; then
    echo "✓ utils.py 第182行已正确添加"
else
    echo "✗ 错误：utils.py 第182行缺失"
    exit 1
fi

echo ""

# ============================================================================
# 第二步：清除缓存
# ============================================================================

echo "【第二步】清除Python缓存"
echo "----------------------------------------------------------------------"

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

echo "✓ 缓存已清除"
echo ""

# ============================================================================
# 第三步：检查训练脚本
# ============================================================================

echo "【第三步】检查可用的训练脚本"
echo "----------------------------------------------------------------------"

TRAIN_SCRIPT=""

if [ -f "train_mfg_regretnet.py" ]; then
    TRAIN_SCRIPT="train_mfg_regretnet.py"
    echo "✓ 找到: train_mfg_regretnet.py"
elif [ -f "exp_rq/train_mfg_regretnet.py" ]; then
    TRAIN_SCRIPT="exp_rq/train_mfg_regretnet.py"
    echo "✓ 找到: exp_rq/train_mfg_regretnet.py"
elif [ -f "train.py" ]; then
    TRAIN_SCRIPT="train.py"
    echo "✓ 找到: train.py"
else
    echo "⚠️  警告：没有找到标准训练脚本"
    echo "   将跳过训练，直接进行RQ3评估"
    TRAIN_SCRIPT=""
fi

echo ""

# ============================================================================
# 第四步：选择运行模式
# ============================================================================

echo "【第四步】选择运行模式"
echo "----------------------------------------------------------------------"
echo ""
echo "方案A：仅运行RQ3评估（使用现有模型）- 快速，30分钟"
echo "方案B：重新训练MFG-RegretNet后运行RQ3 - 完整，1.5-2小时"
echo ""

# 根据参数选择
if [ "$1" = "train" ] || [ "$1" = "full" ]; then
    MODE="train"
    echo "✓ 已选择：完整训练模式"
else
    MODE="eval"
    echo "✓ 已选择：快速评估模式（使用现有模型）"
fi

echo ""

# ============================================================================
# 第五步：训练（可选）
# ============================================================================

if [ "$MODE" = "train" ] && [ ! -z "$TRAIN_SCRIPT" ]; then
    
    echo "【第五步】重新训练MFG-RegretNet"
    echo "----------------------------------------------------------------------"
    echo ""
    echo "开始训练，这可能需要1-2小时..."
    echo "（如果中断，可以使用现有的模型检查点继续）"
    echo ""
    
    python "$TRAIN_SCRIPT" \
        --n-agents 10 \
        --n-items 1 \
        --num-epochs 100 \
        --batch-size 64 \
        --seed 42
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ 训练完成"
    else
        echo ""
        echo "⚠️  训练过程中出错（代码 $TRAIN_EXIT_CODE），但继续进行RQ3评估"
    fi
    
    echo ""
else
    echo "【第五步】跳过训练"
    echo "----------------------------------------------------------------------"
    echo "使用现有的模型权重进行评估"
    echo ""
fi

# ============================================================================
# 第六步：运行RQ3实验
# ============================================================================

echo "【第六步】运行改进后的RQ3实验"
echo "----------------------------------------------------------------------"
echo ""
echo "参数："
echo "  客户端数: 10"
echo "  隐私项目数: 1"
echo "  总预算: 50"
echo "  评估轮数: 1000"
echo "  随机种子: 42,43,44,45,46"
echo ""
echo "预期结果（修复后）："
echo "  Welfare: 37.23 → 45-48 ✓"
echo "  成本: 12.77 → 2-5 ✓"
echo "  BF: 0.84 → 1.0 ✓"
echo ""

python exp_rq/rq3_paper_complete.py \
    --n-agents 10 \
    --n-items 1 \
    --budget 50.0 \
    --num-profiles 1000 \
    --seeds "42,43,44,45,46" \
    --out-dir "run/privacy_paper/rq3"

RQ3_EXIT_CODE=$?

echo ""
echo "============================================================================"

if [ $RQ3_EXIT_CODE -eq 0 ]; then
    echo "✓ RQ3实验完成"
else
    echo "✗ RQ3实验出错（代码 $RQ3_EXIT_CODE）"
fi

# ============================================================================
# 第七步：显示结果
# ============================================================================

echo ""
echo "【第七步】结果汇总"
echo "----------------------------------------------------------------------"
echo ""

if [ -f "run/privacy_paper/rq3/table_rq3_paper.md" ]; then
    echo "RQ3实验结果："
    echo ""
    cat run/privacy_paper/rq3/table_rq3_paper.md
    echo ""
    echo "✓ 完整结果已保存到: run/privacy_paper/rq3/"
else
    echo "⚠️  结果文件不存在"
fi

echo ""
echo "============================================================================"
echo "实验完成!"
echo "============================================================================"
echo ""
echo "后续步骤："
echo "1. 检查Ours行的 mean_social_welfare 值"
echo "   期望: 45+ (从37.23改善)"
echo ""
echo "2. 如果改进效果不理想，可以："
echo "   - 尝试方案B（完整训练）: bash run_rq3_complete.sh train"
echo "   - 运行诊断脚本: python debug_rq3.py"
echo ""
echo "3. 对比改进前后的图表："
echo "   - run/privacy_paper/rq3/figure_rq3_revenue_welfare_bars.png"
echo ""
