# 快速修复指南

## 🚀 在 RunPod 上快速设置

### 方法 1: 使用设置脚本（最简单）

```bash
cd /workspace/Personaplex2
git pull origin main
./setup_token.sh
python quick_test.py
```

### 方法 2: 手动设置

```bash
# 1. 拉取最新代码
cd /workspace/Personaplex2
git pull origin main

# 2. 设置 Token（替换为你的实际 token）
export HF_TOKEN=YOUR_HF_TOKEN_HERE

# 3. 验证 Token
python3 -c "
from huggingface_hub import login, whoami
import os
login(token=os.getenv('HF_TOKEN'))
print('✅ 认证成功:', whoami().get('name'))
"

# 4. 运行测试
python quick_test.py
```

## ⚠️ 重要：必须先接受许可协议

在运行测试前，**必须**完成以下步骤：

1. 访问: https://huggingface.co/nvidia/personaplex-7b-v1
2. **登录**你的 Hugging Face 账号（使用与 token 关联的账号）
3. 点击 **"Agree and access repository"** 按钮
4. 接受 NVIDIA Open Model License Agreement

如果不完成这一步，即使 token 正确，也会收到 401 错误。

## 🔍 验证步骤

```bash
# 检查环境变量
echo $HF_TOKEN

# 应该显示你的 token（以 hf_ 开头）

# 验证认证
python3 -c "
from huggingface_hub import login, whoami
import os
try:
    login(token=os.getenv('HF_TOKEN'))
    user = whoami()
    print('✅ 认证成功!')
    print(f'用户: {user.get(\"name\")}')
except Exception as e:
    print(f'❌ 失败: {e}')
"
```

## 📝 如果仍然失败

1. **确认已接受许可协议**
   - 访问 https://huggingface.co/nvidia/personaplex-7b-v1
   - 确认能看到模型文件列表（而不是 "You need to agree..." 提示）

2. **检查 Token 权限**
   - 访问 https://huggingface.co/settings/tokens
   - 确认 token 有 "Read" 权限

3. **尝试使用 huggingface-cli**
   ```bash
   huggingface-cli login
   # 输入你的 token（以 hf_ 开头）
   ```

4. **检查网络连接**
   ```bash
   ping huggingface.co
   ```

