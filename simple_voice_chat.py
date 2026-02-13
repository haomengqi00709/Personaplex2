#!/usr/bin/env python3
"""
PersonaPlex 简单实时语音对话
只有一个按键：开始/停止说话
左边显示用户说的话，右边显示AI回复
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

# 全局变量
MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
is_recording = False

def load_model():
    """加载模型"""
    global model
    
    if model is not None:
        memory_info = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_info = f" (显存: {memory_used:.2f} GB)"
        return f"✅ 模型已加载{memory_info}"
    
    try:
        if HF_TOKEN:
            login(token=HF_TOKEN)
        
        print("📥 加载模型...")
        
        # 尝试加载模型
        try:
            from transformers import MoshiForConditionalGeneration
            model = MoshiForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except:
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        model.eval()
        
        memory_info = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_info = f" (显存: {memory_used:.2f} GB)"
        
        return f"✅ 模型加载成功！{memory_info}"
        
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"

def process_voice(audio):
    """处理语音输入并生成回复"""
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
        
        # 用户说的话（这里简化处理，实际应该用ASR）
        user_text = f"[音频输入: {len(audio_data)/sr:.2f}秒]"
        
        # AI回复（由于缺少processor，这里显示状态）
        # 实际使用时需要processor或手动实现推理
        ai_text = f"✅ 已收到您的语音\n\n⚠️ 由于缺少processor，无法完成完整推理。\n模型已加载（{torch.cuda.memory_allocated(0)/1e9:.2f} GB），\n但需要processor来处理音频。"
        
        return user_text, ai_text
        
    except Exception as e:
        return f"❌ 处理失败: {str(e)}", ""

# 创建简单界面
with gr.Blocks(title="PersonaPlex 语音对话", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ PersonaPlex 实时语音对话
    
    简单测试界面
    """)
    
    # 模型加载
    with gr.Row():
        load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
        status = gr.Textbox(label="状态", value="❌ 模型未加载", interactive=False)
    
    gr.Markdown("---")
    
    # 语音对话区域
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 您说的话")
            user_text = gr.Textbox(label="", lines=10, interactive=False, placeholder="您说的话会显示在这里...")
        
        with gr.Column():
            gr.Markdown("### 🤖 AI 回复")
            ai_text = gr.Textbox(label="", lines=10, interactive=False, placeholder="AI的回复会显示在这里...")
    
    # 语音输入
    audio_input = gr.Audio(
        label="",
        type="filepath",
        sources=["microphone"],
        format="wav",
        show_label=False
    )
    
    # 处理按钮
    process_btn = gr.Button("🚀 处理语音", variant="primary", size="lg")
    
    gr.Markdown("""
    ---
    ### 📝 使用说明
    
    1. 点击"加载模型"（首次需要几分钟）
    2. 点击麦克风图标录制语音
    3. 点击"处理语音"按钮
    4. 查看左侧（您说的话）和右侧（AI回复）
    """)
    
    # 事件绑定
    load_btn.click(fn=load_model, outputs=status)
    process_btn.click(
        fn=process_voice,
        inputs=[audio_input],
        outputs=[user_text, ai_text]
    )

if __name__ == "__main__":
    print("="*60)
    print("PersonaPlex 简单语音对话")
    print("端口: 5001")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=5001,
        share=False
    )

