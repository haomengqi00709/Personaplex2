# Hugging Face Token 设置指南

## 🔐 问题：认证失败

如果遇到以下错误：
- `Invalid user token`
- `401 Client Error`
- `Cannot access gated repo`

请按照以下步骤解决：

## 📋 步骤 1: 创建 Access Token

1. **访问 Token 设置页面**
   - 打开: https://huggingface.co/settings/tokens
   - 登录你的 Hugging Face 账号

2. **创建新 Token**
   - 点击 **"New token"** 或 **"Create new token"**
   - 选择 **"Read"** 权限（至少需要读取权限）
   - 给 Token 起个名字，例如: `personaplex-runpod`
   - 点击 **"Generate token"**

3. **复制 Token**
   - ⚠️ **重要**: Token 只会显示一次，请立即复制保存
   - Token 格式类似: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## 📋 步骤 2: 接受模型许可协议

**这是关键步骤！** PersonaPlex 是 gated repo，需要先接受许可协议。

1. **访问模型页面**
   - 打开: https://huggingface.co/nvidia/personaplex-7b-v1
   - **必须登录**你的 Hugging Face 账号

2. **接受许可协议**
   - 点击页面上的 **"Agree and access repository"** 按钮
   - 阅读并接受 NVIDIA Open Model License Agreement
   - 确认访问权限

3. **验证访问**
   - 刷新页面，应该能看到模型文件列表
   - 如果仍然看到 "You need to agree to share your contact information"，说明还没接受协议

## 📋 步骤 3: 在 RunPod 上设置 Token

### 方法 A: 使用环境变量（推荐）

在 RunPod Web Terminal 中执行：

```bash
# 设置 Token（替换 YOUR_TOKEN 为你的实际 token）
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 验证设置
echo $HF_TOKEN

# 应该显示你的 token（以 hf_ 开头）
```

### 方法 B: 在 RunPod Pod 设置中添加

1. 在 RunPod Pod 详情页
2. 找到 **"Environment Variables"** 或 **"Env"** 设置
3. 添加新变量：
   - **Key**: `HF_TOKEN`
   - **Value**: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
4. 保存并重启 Pod（如果需要）

### 方法 C: 使用 huggingface-cli 登录

```bash
# 安装 huggingface-hub（如果还没安装）
pip install huggingface-hub

# 登录（会提示输入 token）
huggingface-cli login

# 输入你的 token（以 hf_ 开头）
```

## ✅ 验证设置

运行以下命令验证：

```bash
# 方法 1: 检查环境变量
echo $HF_TOKEN

# 方法 2: 使用 Python 验证
python3 -c "
from huggingface_hub import login, whoami
import os
token = os.getenv('HF_TOKEN')
if token:
    try:
        login(token=token)
        user = whoami()
        print(f'✅ 认证成功! 用户: {user.get(\"name\", \"Unknown\")}')
    except Exception as e:
        print(f'❌ 认证失败: {e}')
else:
    print('⚠️  HF_TOKEN 未设置')
"
```

## 🔍 常见问题

### Q1: Token 格式错误

**错误**: `Invalid user token`

**解决**:
- 确保 Token 以 `hf_` 开头
- 确保复制了完整的 Token（没有遗漏字符）
- 重新生成 Token 并复制

### Q2: 401 Unauthorized

**错误**: `401 Client Error` 或 `Cannot access gated repo`

**解决**:
1. ✅ 确认已访问 https://huggingface.co/nvidia/personaplex-7b-v1
2. ✅ 确认已登录 Hugging Face 账号
3. ✅ 确认已点击 "Agree and access repository"
4. ✅ 确认 Token 有正确的权限（至少 Read）

### Q3: Token 已过期

**解决**:
- 生成新的 Token
- 更新环境变量或 Pod 设置

### Q4: 仍然无法访问

**检查清单**:
- [ ] Token 格式正确（以 `hf_` 开头）
- [ ] Token 有 Read 权限
- [ ] 已接受模型许可协议
- [ ] 使用正确的 Hugging Face 账号（接受协议的账号）
- [ ] 环境变量已正确设置
- [ ] 已重启终端或 Pod（如果修改了环境变量）

## 🎯 快速测试

设置完成后，运行：

```bash
cd /workspace/Personaplex2
python quick_test.py
```

应该看到：
```
✅ Hugging Face 认证成功
✅ 模型加载成功
```

## 📝 完整设置流程

```bash
# 1. 设置 Token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 2. 验证 Token
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN'); print('✅ Token 有效')"

# 3. 运行测试
cd /workspace/Personaplex2
python quick_test.py
```

## 🔗 相关链接

- [Hugging Face Tokens](https://huggingface.co/settings/tokens)
- [PersonaPlex 模型页面](https://huggingface.co/nvidia/personaplex-7b-v1)
- [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/quick-start#login)

