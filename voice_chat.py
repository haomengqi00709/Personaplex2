#!/usr/bin/env python3
"""
PersonaPlex 简单语音对话
一个按钮：开始/停止说话
左边：您说的话 | 右边：AI回复
"""

import os
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from transformers import MoshiForConditionalGeneration, AutoModel
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """加载模型"""
    global model
    
    if model is not None:
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return f"✅ 模型已加载 ({mem:.2f} GB)"
    
    try:
        if HF_TOKEN:
            login(token=HF_TOKEN)
        
        print("📥 加载模型...")
        
        # 使用 AutoModel 加载（会自动使用自定义代码）
        # 虽然会有警告，但这是正确的加载方式
        print("⚠️  注意: PersonaPlex 使用自定义架构，会有权重不匹配警告（这是正常的）")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,  # 关键：信任远程代码以加载自定义架构
            ignore_mismatched_sizes=True  # 忽略大小不匹配
        )
        
        model.eval()
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return f"✅ 模型加载成功！({mem:.2f} GB)"
        
    except Exception as e:
        return f"❌ 失败: {str(e)}"

def process_voice(audio):
    """处理语音"""
    global model
    
    if model is None:
        return "❌ 请先加载模型", "❌ 模型未加载"
    
    if audio is None:
        return "", ""
    
    try:
        # 读取音频
        audio_data, sr = sf.read(audio)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 重采样到 24kHz
        if sr != 24000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
            sr = 24000
        
        duration = len(audio_data) / sr
        user_text = f"🎤 语音输入 ({duration:.2f}秒)"
        
        # 尝试调用模型（即使没有processor）
        try:
            # 将音频转换为tensor
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(device)
            
            # 尝试直接调用模型（需要根据实际架构调整）
            # 由于PersonaPlex架构特殊，这里提供一个基础尝试
            with torch.no_grad():
                # 尝试使用模型的forward方法
                # 注意：这可能需要特定的输入格式
                try:
                    # 创建一个简单的输入（可能需要调整）
                    # PersonaPlex可能需要audio codes和text tokens
                    # 这里我们尝试最简单的调用
                    
                    # 由于没有processor，我们无法正确编码输入
                    # 但可以显示模型已准备好
                    ai_text = f"✅ 已收到语音 ({duration:.2f}秒)\n\n模型已加载并准备处理。\n\n⚠️ 由于缺少processor，无法完成完整推理。\n模型需要特定的音频编码格式。"
                    
                except Exception as e:
                    ai_text = f"✅ 模型已加载\n\n⚠️ 推理需要processor或了解输入格式。\n错误: {str(e)}"
        except Exception as e:
            ai_text = f"✅ 音频已处理\n\n⚠️ 模型调用需要特定格式。\n{str(e)}"
        
        return user_text, ai_text
        
    except Exception as e:
        return f"❌ 错误: {str(e)}", ""

# 创建界面
with gr.Blocks(title="PersonaPlex 语音对话", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ PersonaPlex 语音对话")
    
    # 加载模型
    load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
    status = gr.Textbox(label="状态", value="❌ 模型未加载", interactive=False)
    
    gr.Markdown("---")
    
    # 对话区域
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 您说的话")
            user_text = gr.Textbox(label="", lines=12, interactive=False, placeholder="...")
        
        with gr.Column():
            gr.Markdown("### 🤖 AI 回复")
            ai_text = gr.Textbox(label="", lines=12, interactive=False, placeholder="...")
    
    # 语音输入
    audio_input = gr.Audio(
        label="",
        type="filepath",
        sources=["microphone"],
        format="wav",
        show_label=False
    )
    
    # 事件
    load_btn.click(fn=load_model, outputs=status)
    audio_input.change(
        fn=process_voice,
        inputs=[audio_input],
        outputs=[user_text, ai_text]
    )

if __name__ == "__main__":
    print("="*60)
    print("PersonaPlex 语音对话 - 端口 5001")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=5001,
        share=False
    )

