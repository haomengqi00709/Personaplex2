# PersonaPlex 模型测试项目

用于在 RunPod GPU 上测试 NVIDIA PersonaPlex-7b-v1 模型的最低配置代码。

## 📋 要求

### 硬件要求
- **GPU**: 至少 16GB VRAM（推荐 24GB+）
  - 最低: RTX 3090, A4000, V100 32GB
  - 推荐: A100, H100, RTX 4090
- **系统**: Linux（RunPod 默认环境）

### 软件要求
- Python 3.8+
- CUDA 11.8+ 或 12.0+
- PyTorch 2.0+

## 🚀 快速开始

### 1. 在 RunPod 上设置环境

#### 方法 A: 使用自定义模板
1. 在 RunPod 创建 Pod
2. 选择 GPU（至少 16GB VRAM）
3. 选择 PyTorch 2.0+ 模板
4. 上传项目文件

#### 方法 B: 使用 Docker 容器
```bash
# 在 RunPod 终端中
git clone <your-repo-url>
cd Personaplex2
pip install -r requirements.txt
```

### 2. 设置 Hugging Face Token

PersonaPlex 模型需要接受许可协议并认证：

1. 访问 https://huggingface.co/nvidia/personaplex-7b-v1
2. 登录并接受许可协议
3. 创建 Access Token: https://huggingface.co/settings/tokens
4. 在 RunPod 中设置环境变量：

```bash
export HF_TOKEN=your_huggingface_token_here
```

或者在 RunPod 的 Pod 设置中添加环境变量 `HF_TOKEN`。

### 3. 运行测试

**方式 1: Web 前端（推荐）**
```bash
python app.py
```
然后访问 RunPod 提供的公共 URL 或 http://localhost:7860

**方式 2: 命令行测试**
```bash
# 快速测试
python quick_test.py

# 完整测试
python test_personaplex.py
```

## 📁 项目结构

```
Personaplex2/
├── README.md              # 本文件
├── requirements.txt       # Python 依赖
├── config.yaml           # 配置文件
├── app.py                # Web 前端（Gradio）
├── test_personaplex.py   # 完整测试脚本
├── quick_test.py         # 快速测试脚本
├── start_web.sh          # Web 启动脚本
├── runpod_setup.sh       # RunPod 环境设置脚本
├── RUNPOD_GUIDE.md       # RunPod 详细部署指南
└── FRONTEND_GUIDE.md     # Web 前端使用指南
```

## ⚙️ 配置说明

编辑 `config.yaml` 可以调整：

- **模型配置**: 模型 ID、数据类型（float16 降低显存）
- **音频配置**: 采样率（24kHz）、块大小
- **推理配置**: 生成参数（temperature, top_p 等）

### 最低配置优化

当前配置已优化为最低性能要求：
- 使用 `float16` 降低显存占用
- 启用 `low_cpu_mem_usage` 减少内存使用
- 使用 `device_map="auto"` 自动分配 GPU

## 🔧 故障排除

### 问题 1: 显存不足 (OOM)
**解决方案**:
- 使用更大的 GPU（24GB+）
- 在 `config.yaml` 中确保使用 `float16`
- 减小 `max_new_tokens` 值

### 问题 2: Hugging Face 认证失败
**解决方案**:
- 确认已设置 `HF_TOKEN` 环境变量
- 确认已接受模型许可协议
- 检查网络连接

### 问题 3: 模型下载失败
**解决方案**:
- 检查 Hugging Face 访问权限
- 尝试手动下载模型到本地
- 使用镜像站点（如果在中国大陆）

### 问题 4: CUDA 错误
**解决方案**:
- 确认 RunPod 环境已安装 CUDA
- 检查 PyTorch CUDA 版本匹配
- 尝试重启 Pod

## 📊 性能参考

在最低配置（16GB VRAM）下的预期性能：
- 模型加载时间: ~2-5 分钟
- 首次推理延迟: ~5-10 秒
- 后续推理延迟: ~1-3 秒
- 显存占用: ~12-15 GB

## 🔗 相关链接

- [PersonaPlex 模型页面](https://huggingface.co/nvidia/personaplex-7b-v1)
- [GitHub 仓库](https://github.com/NVIDIA/personaplex)
- [RunPod 文档](https://docs.runpod.io/)

## 📝 许可证

本项目遵循 NVIDIA Open Model License Agreement。

