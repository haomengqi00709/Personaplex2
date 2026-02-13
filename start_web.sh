#!/bin/bash
# 启动 Web 前端脚本

echo "🚀 启动 PersonaPlex Web 界面..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
pip install -q gradio

# 启动应用
echo "🌐 启动 Web 服务器..."
echo ""
echo "访问地址:"
echo "  - 本地: http://localhost:7860"
echo "  - RunPod: 查看 Pod 的公共 URL"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python3 app.py

