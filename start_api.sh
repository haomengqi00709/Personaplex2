#!/bin/bash
# 启动 API 服务器

echo "🚀 启动 PersonaPlex API 服务器..."

# 检查依赖
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 安装依赖..."
    pip install flask flask-cors
fi

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动服务器
python api_server.py

