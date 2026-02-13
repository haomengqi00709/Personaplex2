# PersonaPlex 快速启动指南 - 实时翻译机

## 🎯 目标
1. 让 PersonaPlex 模型运行起来
2. 实现实时翻译功能

## 🚀 快速启动（在 RunPod 上）

### 步骤 1: 设置官方代码库

```bash
cd /workspace

# 克隆官方仓库
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex

# 安装依赖
pip install -r requirements.txt

# 设置 Token
export HF_TOKEN=YOUR_HF_TOKEN_HERE
```

### 步骤 2: 查看官方示例

```bash
# 查看 README
cat README.md

# 查找示例代码
find . -name "*.py" -type f | grep -E "(example|demo|inference)" | head -10

# 通常会有类似这样的文件：
# - examples/basic_inference.py
# - examples/streaming_demo.py
# - scripts/inference.py
```

### 步骤 3: 运行基础测试

```bash
# 按照官方文档运行示例
# 例如：
# python examples/basic_inference.py
# 或
# python -m personaplex.inference --model-id nvidia/personaplex-7b-v1
```

### 步骤 4: 使用我们的翻译界面

```bash
cd /workspace/Personaplex2
git pull origin main
python translator.py
```

## 📝 下一步
模型运行成功后，使用 `translator.py` 进行实时翻译测试。

