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
model_loading = False
model_status = "未加载"

def load_model():
    """加载模型"""
    global model, model_status
    
    if model is not None:
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return f"✅ 模型已加载 ({mem:.2f} GB)"
    
    try:
        if HF_TOKEN:
            login(token=HF_TOKEN)
        
        print("📥 加载模型...")
        model_status = "加载中..."
        
        # 首先检查 Transformers 版本
        import transformers
        transformers_version = transformers.__version__
        print(f"Transformers 版本: {transformers_version}")
        
        # 尝试加载模型
        try:
            # 方法1: 使用 AutoModel + trust_remote_code
            print("尝试使用 AutoModel 加载...")
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✅ 使用 AutoModel 加载成功")
        except Exception as e1:
            print(f"⚠️  AutoModel 失败: {e1}")
            
            # 方法2: 尝试从源码安装的 Transformers
            error_msg = str(e1)
            if "does not recognize this architecture" in error_msg or "personaplex" in error_msg.lower():
                return f"""❌ Transformers 版本不支持 PersonaPlex 架构

当前版本: {transformers_version}

解决方案:
1. 升级 Transformers:
   pip install --upgrade transformers

2. 或从源码安装最新版本:
   pip install git+https://github.com/huggingface/transformers.git

3. 然后重新启动程序"""
            else:
                raise e1
        
        model.eval()
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        model_status = "已加载"
        return f"✅ 模型加载成功！({mem:.2f} GB)"
        
    except Exception as e:
        model_status = "加载失败"
        error_msg = str(e)
        if "does not recognize this architecture" in error_msg:
            return f"""❌ Transformers 不支持 PersonaPlex 架构

请执行以下命令升级 Transformers:
pip install --upgrade transformers

或从源码安装:
pip install git+https://github.com/huggingface/transformers.git

然后重新启动程序"""
        return f"❌ 失败: {error_msg}"

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

# 启动时自动加载模型
def auto_load_model():
    """程序启动时自动加载模型"""
    print("="*60)
    print("自动加载模型...")
    print("="*60)
    return load_model()

# 创建界面
with gr.Blocks(title="PersonaPlex 语音对话", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ PersonaPlex 语音对话")
    
    # 状态显示（自动加载）
    status = gr.Textbox(
        label="模型状态", 
        value="正在加载模型...", 
        interactive=False,
        lines=3
    )
    
    # 手动重新加载按钮（可选）
    load_btn = gr.Button("🔄 重新加载模型", variant="secondary", size="sm")
    
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
    # 启动时自动加载模型
    demo.load(fn=auto_load_model, outputs=status)
    
    # 手动重新加载
    load_btn.click(fn=load_model, outputs=status)
    
    # 语音处理
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

