#!/bin/bash
# 快速设置 HF_TOKEN 脚本

# 请替换为你的实际 Token
TOKEN="YOUR_HF_TOKEN_HERE"

echo "🔐 设置 Hugging Face Token..."
export HF_TOKEN="$TOKEN"

echo "✅ Token 已设置"
echo ""
echo "验证 Token..."

python3 -c "
from huggingface_hub import login, whoami
import os
token = os.getenv('HF_TOKEN')
if token:
    try:
        login(token=token)
        user = whoami()
        print(f'✅ 认证成功! 用户: {user.get(\"name\", \"Unknown\")}')
        print(f'   邮箱: {user.get(\"email\", \"N/A\")}')
    except Exception as e:
        print(f'❌ 认证失败: {e}')
        print('')
        print('请检查:')
        print('1. Token 是否正确')
        print('2. 是否已接受模型许可协议: https://huggingface.co/nvidia/personaplex-7b-v1')
else:
    print('⚠️  HF_TOKEN 未设置')
"

echo ""
echo "📝 重要提示:"
echo "1. 确保已访问 https://huggingface.co/nvidia/personaplex-7b-v1"
echo "2. 确保已点击 'Agree and access repository' 接受许可协议"
echo ""
echo "现在可以运行测试:"
echo "  python quick_test.py"

