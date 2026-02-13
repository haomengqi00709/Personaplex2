#!/bin/bash
# RunPod 环境设置脚本

echo "🚀 设置 PersonaPlex 测试环境..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  警告: nvidia-smi 未找到"
fi

# 安装依赖
echo "📦 安装 Python 依赖..."
pip install -r requirements.txt

# 检查 Hugging Face Token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  警告: HF_TOKEN 环境变量未设置"
    echo "请在 RunPod Pod 设置中添加环境变量: HF_TOKEN=your_token"
else
    echo "✅ HF_TOKEN 已设置"
fi

echo "✅ 环境设置完成！"
echo ""
echo "运行测试: python3 test_personaplex.py"

