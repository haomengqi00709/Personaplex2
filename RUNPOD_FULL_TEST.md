# RunPod 完整测试指南

## 🎯 目标：在 RunPod 上实际运行 PersonaPlex 模型

### 方案 1: 使用官方代码库（推荐）

#### 步骤 1: 设置官方代码库

```bash
cd /workspace

# 克隆官方仓库
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex

# 安装依赖
pip install -r requirements.txt

# 查看 README
cat README.md
```

#### 步骤 2: 设置环境变量

```bash
export HF_TOKEN=YOUR_HF_TOKEN_HERE
```

#### 步骤 3: 运行官方示例

```bash
# 查看示例代码
ls examples/
ls scripts/

# 按照官方 README 运行
# 通常会有类似这样的命令：
# python examples/basic_inference.py
# 或
# python -m personaplex.inference ...
```

### 方案 2: 从源码安装 transformers（如果官方代码不可用）

```bash
# 升级到最新版本
pip install --upgrade transformers

# 或从源码安装
pip install git+https://github.com/huggingface/transformers.git

# 然后尝试使用我们的测试代码
cd /workspace/Personaplex2
python quick_test.py
```

### 方案 3: 使用我们的设置脚本

```bash
cd /workspace/Personaplex2
chmod +x setup_official_personaplex.sh
./setup_official_personaplex.sh

# 然后运行测试
python test_official.py
```

## 📋 完整测试流程

### 1. 环境准备

```bash
# 检查 GPU
nvidia-smi

# 检查 Python 和 CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 设置 Token
export HF_TOKEN=YOUR_HF_TOKEN_HERE
```

### 2. 下载/验证模型

```bash
# 模型应该已经下载（16.7GB）
# 如果没下载，运行我们的测试会自动下载
cd /workspace/Personaplex2
python quick_test.py
```

### 3. 使用官方代码库运行

```bash
cd /workspace/personaplex

# 查看可用的脚本
find . -name "*.py" -type f | grep -E "(example|inference|test)" | head -10

# 运行官方示例（根据实际文件名调整）
# python examples/xxx.py
```

## 🔍 如果遇到问题

### 问题 1: 官方代码库结构不同

```bash
# 查看实际结构
cd /workspace/personaplex
find . -maxdepth 3 -type f -name "*.py" | head -20
cat README.md
```

### 问题 2: 依赖缺失

```bash
# 安装常见依赖
pip install torch transformers accelerate huggingface-hub soundfile librosa numpy scipy
```

### 问题 3: 模型路径问题

```bash
# 检查模型缓存
ls -lh ~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/
```

## ✅ 验证测试成功

成功的测试应该能够：
1. ✅ 加载模型到 GPU
2. ✅ 处理音频输入
3. ✅ 生成语音输出
4. ✅ 显示推理结果

## 📝 快速命令参考

```bash
# 完整流程
cd /workspace
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex
pip install -r requirements.txt
export HF_TOKEN=your_token
cat README.md  # 查看使用方法
# 然后按照 README 运行示例
```

