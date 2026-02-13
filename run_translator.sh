#!/bin/bash
# 启动独立翻译机

echo "🚀 启动 PersonaPlex 实时翻译机..."
echo ""

cd /workspace/Personaplex2

# 检查 Token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  警告: HF_TOKEN 未设置"
    echo "   请设置: export HF_TOKEN=your_token"
    echo ""
fi

# 启动
echo "🌐 启动 Web 服务器在端口 5001..."
echo ""
echo "访问方式:"
echo "  - 在 RunPod Pod 详情页找到公共 URL"
echo "  - 端口: 5001"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python3 standalone_translator.py

