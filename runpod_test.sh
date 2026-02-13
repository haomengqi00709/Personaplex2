#!/bin/bash
# RunPod 完整测试脚本

echo "="*60
echo "PersonaPlex RunPod 完整测试"
echo "="*60

# 1. 检查环境
echo ""
echo "1. 检查环境..."
echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Python 环境:"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. 检查 Token
echo ""
echo "2. 检查 Hugging Face Token..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN 未设置"
    echo "   请设置: export HF_TOKEN=your_token"
    exit 1
else
    echo "✅ HF_TOKEN 已设置"
fi

# 3. 设置官方代码库
echo ""
echo "3. 设置官方 PersonaPlex 代码库..."
cd /workspace

if [ ! -d "personaplex" ]; then
    echo "📥 克隆官方仓库..."
    git clone https://github.com/NVIDIA/personaplex.git
else
    echo "✅ 官方仓库已存在"
fi

cd personaplex

# 4. 安装依赖
echo ""
echo "4. 安装依赖..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
else
    echo "⚠️  未找到 requirements.txt，安装基础依赖..."
    pip install -q torch transformers accelerate huggingface-hub soundfile librosa numpy
fi

# 5. 查看官方文档
echo ""
echo "5. 官方代码库信息:"
if [ -f "README.md" ]; then
    echo "✅ 找到 README.md"
    echo ""
    echo "前 50 行内容:"
    head -50 README.md
    echo ""
    echo "查看完整 README: cat /workspace/personaplex/README.md"
else
    echo "⚠️  未找到 README.md"
fi

# 6. 查找示例代码
echo ""
echo "6. 查找示例代码:"
find . -maxdepth 2 -name "*.py" -type f | grep -E "(example|test|demo|inference)" | head -10

# 7. 运行我们的基础测试
echo ""
echo "7. 运行基础模型加载测试..."
cd /workspace/Personaplex2
python3 quick_test.py

echo ""
echo "="*60
echo "✅ 测试完成！"
echo "="*60
echo ""
echo "下一步:"
echo "1. 查看官方 README: cat /workspace/personaplex/README.md"
echo "2. 查看示例代码: ls /workspace/personaplex/examples/ 2>/dev/null || find /workspace/personaplex -name '*example*.py'"
echo "3. 按照官方文档运行完整测试"

