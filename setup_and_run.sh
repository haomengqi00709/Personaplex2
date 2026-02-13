#!/bin/bash
# 设置官方 PersonaPlex 代码库并创建可用的前端

echo "🚀 设置官方 PersonaPlex 代码库..."

cd /workspace

# 1. 克隆官方仓库
if [ ! -d "personaplex" ]; then
    echo "📥 克隆官方仓库..."
    git clone https://github.com/NVIDIA/personaplex.git
else
    echo "✅ 官方仓库已存在"
    cd personaplex
    git pull
    cd ..
fi

cd personaplex

# 2. 安装依赖
echo "📦 安装依赖..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
else
    pip install -q torch transformers accelerate huggingface-hub soundfile librosa numpy gradio
fi

# 3. 查看官方文档
echo ""
echo "📖 官方代码库信息:"
if [ -f "README.md" ]; then
    echo "找到 README.md"
    echo "前 100 行:"
    head -100 README.md
fi

# 4. 查找示例代码
echo ""
echo "🔍 查找示例代码:"
find . -maxdepth 3 -name "*.py" -type f | grep -E "(example|demo|inference|chat)" | head -10

echo ""
echo "✅ 设置完成！"
echo ""
echo "下一步:"
echo "1. 查看官方 README: cat /workspace/personaplex/README.md"
echo "2. 查看示例代码: find /workspace/personaplex -name '*example*.py'"
echo "3. 按照官方文档运行"

