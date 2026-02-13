#!/bin/bash
# 启动前端服务器（端口 5001）

echo "🚀 启动 PersonaPlex Web 前端..."
echo ""

# 检查 Token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  警告: HF_TOKEN 未设置"
    echo "   请设置: export HF_TOKEN=your_token"
    echo ""
fi

# 进入项目目录
cd /workspace/Personaplex2

# 检查依赖
echo "📦 检查依赖..."
pip show gradio > /dev/null 2>&1 || pip install -q gradio

# 启动服务器
echo ""
echo "🌐 启动 Web 服务器在端口 5001..."
echo ""
echo "访问方式:"
echo "  - 在 RunPod Pod 详情页找到公共 URL"
echo "  - 或使用: http://localhost:5001 (如果配置了端口转发)"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python3 app.py

