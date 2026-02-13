# RunPod 快速开始指南

## 🚀 在 RunPod 上快速设置和测试

### 步骤 1: 克隆或拉取代码

**如果目录不存在（首次使用）:**
```bash
cd /workspace
git clone https://github.com/haomengqi00709/Personaplex2.git
cd Personaplex2
```

**如果目录已存在:**
```bash
cd /workspace/Personaplex2
git pull origin main
```

### 步骤 2: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 3: 设置环境变量

```bash
# 设置 Hugging Face Token（替换为你的 token）
export HF_TOKEN=your_huggingface_token_here

# 验证设置
echo $HF_TOKEN
```

**或者在 RunPod Pod 设置中添加环境变量 `HF_TOKEN`**

### 步骤 4: 运行测试

**方式 1: 快速测试（推荐先运行）**
```bash
python quick_test.py
```

**方式 2: 启动 Web 前端**
```bash
python app.py
```
然后访问 RunPod 提供的公共 URL

**方式 3: 完整测试**
```bash
python test_personaplex.py
```

## 📋 完整命令序列（复制粘贴）

```bash
# 1. 进入工作目录
cd /workspace

# 2. 克隆仓库（如果不存在）
git clone https://github.com/haomengqi00709/Personaplex2.git
cd Personaplex2

# 3. 安装依赖
pip install -r requirements.txt

# 4. 设置环境变量（替换 YOUR_TOKEN）
export HF_TOKEN=YOUR_TOKEN_HERE

# 5. 运行快速测试
python quick_test.py
```

## 🔍 检查环境

```bash
# 检查 GPU
nvidia-smi

# 检查 Python
python3 --version

# 检查 CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查已安装的包
pip list | grep -E "torch|transformers|gradio"
```

## ⚠️ 常见问题

### 问题 1: git clone 失败
```bash
# 检查网络连接
ping github.com

# 如果在中国大陆，可能需要使用代理或镜像
```

### 问题 2: pip install 很慢
```bash
# 使用国内镜像（可选）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 3: 显存不足
```bash
# 检查显存使用
nvidia-smi

# 确保使用 float16（已在代码中配置）
```

## 🎯 下一步

测试成功后，可以：
1. 启动 Web 界面进行交互式测试
2. 修改 `config.yaml` 调整参数
3. 准备自己的音频文件进行测试

