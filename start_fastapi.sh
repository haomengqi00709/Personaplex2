#!/bin/bash
# 启动 FastAPI 服务器

echo "🚀 启动 PersonaPlex FastAPI 服务器..."

# 检查依赖
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 安装依赖..."
    pip install fastapi uvicorn[standard] python-multipart
fi

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动服务器
python api_server_fastapi.py

