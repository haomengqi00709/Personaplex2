#!/bin/bash
# 安装所有依赖

echo "📦 安装 Python 依赖..."

# 基础依赖
pip install soundfile librosa numpy scipy torch torchaudio

# FastAPI 相关
pip install fastapi uvicorn[standard] python-multipart

# Hugging Face 相关
pip install huggingface-hub transformers accelerate safetensors sentencepiece

# 音频处理
pip install soundfile librosa

# 安装 ffmpeg（用于 WebM 支持）
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 安装 ffmpeg..."
    apt-get update && apt-get install -y ffmpeg
fi

echo "✅ 依赖安装完成！"

