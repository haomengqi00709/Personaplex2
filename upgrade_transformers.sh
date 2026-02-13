#!/bin/bash
# 升级 Transformers 以支持 PersonaPlex

echo "🔄 升级 Transformers 库..."
echo ""

# 方法1: 升级到最新稳定版
echo "方法1: 升级到最新稳定版"
pip install --upgrade transformers

echo ""
echo "如果方法1失败，尝试方法2: 从源码安装"
echo "执行: pip install git+https://github.com/huggingface/transformers.git"
echo ""

# 检查版本
python3 -c "import transformers; print(f'当前版本: {transformers.__version__}')"

echo ""
echo "✅ 升级完成！请重新启动程序。"

