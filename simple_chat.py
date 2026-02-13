#!/usr/bin/env python3
"""
PersonaPlex 简单实时语音对话测试界面
"""

import os
import torch
import numpy as np
import soundfile as sf
import gradio as gr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, MoshiForConditionalGeneration
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# 全局变量
MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """加载模型"""
    global model, processor
    
    if model is not None:
        return "✅ 模型已加载"
    
    try:
        # 认证
        if HF_TOKEN:
            login(token=HF_TOKEN)
        
        # 加载模型（使用 MoshiForConditionalGeneration）
        print("📥 加载模型...")
        model = MoshiForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.eval()
        
        # 尝试加载 processor（如果失败也没关系）
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True
            )
        except:
            processor = None
            print("⚠️  Processor 不可用，将使用基础功能")
        
        memory_info = ""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_info = f"\n显存: {memory_used:.2f} GB"
        
        return f"✅ 模型加载成功！{memory_info}"
        
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"

def chat(audio, text_prompt):
    """处理语音输入并生成响应"""
    global model, processor
    
    if model is None:
        return None, "❌ 请先加载模型"
    
    if processor is None:
        return None, "❌ Processor 不可用，无法进行推理。\n\n建议使用官方 PersonaPlex 代码库。"
    
    if audio is None:
        return None, "❌ 请录制或上传音频"
    
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
        
        # 设置文本提示
        if not text_prompt or text_prompt.strip() == "":
            text_prompt = "You are a helpful AI assistant."
        
        # 处理输入
        inputs = processor(
            audio=audio_data,
            sampling_rate=sr,
            text=text_prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 生成响应
        print("生成响应...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        # 解码文本
        if hasattr(processor, 'decode'):
            text_output = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            text_output = "响应已生成"
        
        # 尝试提取音频输出
        output_audio = None
        if hasattr(outputs, 'audio_values'):
            output_audio = outputs.audio_values.cpu().numpy()
        elif isinstance(outputs, dict) and 'audio_values' in outputs:
            output_audio = outputs['audio_values'].cpu().numpy()
        
        # 如果没有音频输出，生成占位音频
        if output_audio is None:
            sample_rate = 24000
            duration = 2.0
            output_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
            output_audio = output_audio.astype(np.float32)
            text_output += "\n\n⚠️ 音频输出不可用（已生成占位音频）"
        
        # 保存输出音频
        output_path = "/tmp/personaplex_response.wav"
        sf.write(output_path, output_audio, 24000)
        
        return output_path, text_output
        
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

# 创建简单界面
with gr.Blocks(title="PersonaPlex 语音对话", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ PersonaPlex 实时语音对话测试
    
    简单测试界面：说话 → AI 回复
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
            status = gr.Textbox(label="状态", value="❌ 模型未加载", interactive=False)
        
        with gr.Column(scale=2):
            text_prompt = gr.Textbox(
                label="角色设定（可选）",
                value="You are a helpful AI assistant.",
                placeholder="例如: You are a friendly assistant.",
                lines=2
            )
    
    with gr.Row():
        audio_input = gr.Audio(
            label="🎤 说话（点击录制）",
            type="filepath",
            sources=["microphone"],
            format="wav"
        )
    
    chat_btn = gr.Button("🚀 发送", variant="primary", size="lg")
    
    with gr.Row():
        audio_output = gr.Audio(label="🔊 AI 回复", type="filepath", format="wav")
        text_output = gr.Textbox(label="📝 文本回复", lines=3, interactive=False)
    
    # 事件
    load_btn.click(fn=load_model, outputs=status)
    chat_btn.click(fn=chat, inputs=[audio_input, text_prompt], outputs=[audio_output, text_output])

if __name__ == "__main__":
    print("="*60)
    print("启动 PersonaPlex 简单语音对话界面")
    print("端口: 5001")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=5001,
        share=False
    )

