#!/usr/bin/env python3
"""
PersonaPlex Web 测试界面
使用 Gradio 创建简易前端
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

def load_model_once():
    """加载模型（只加载一次）"""
    global model, processor
    
    if model is not None:
        return "✅ 模型已加载"
    
    try:
        # 认证
        auth_success = False
        if HF_TOKEN:
            if not HF_TOKEN.startswith('hf_'):
                print("⚠️  警告: Token 格式可能不正确（应以 'hf_' 开头）")
            try:
                login(token=HF_TOKEN)
                auth_success = True
            except Exception as e:
                print(f"⚠️  Token 认证失败: {e}")
        
        # 如果没有 token 或 token 失败，尝试使用 CLI 登录
        if not auth_success:
            try:
                from huggingface_hub import whoami
                user_info = whoami()
                if user_info:
                    print(f"✅ 使用 huggingface-cli 认证: {user_info.get('name', 'Unknown')}")
                    auth_success = True
            except:
                pass
        
        if not auth_success:
            print("⚠️  警告: 未找到有效的认证方式")
            print("   请设置 HF_TOKEN 或运行: huggingface-cli login")
        
        print("📥 加载处理器...")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        
        print("📥 加载模型...")
        try:
            model = MoshiForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
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
            memory_info = f"\n显存使用: {memory_used:.2f} GB"
        
        return f"✅ 模型加载成功！{memory_info}"
        
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"

def process_audio(audio_file, text_prompt, voice_prompt_file):
    """处理音频输入并生成响应"""
    global model, processor
    
    if model is None or processor is None:
        return None, "❌ 请先加载模型！", None
    
    try:
        # 处理用户音频
        if audio_file is None:
            # 如果没有上传音频，创建静音
            sample_rate = 24000
            duration = 1.0
            user_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            user_sr = sample_rate
        else:
            # 读取上传的音频
            user_audio, user_sr = sf.read(audio_file)
            if len(user_audio.shape) > 1:
                user_audio = np.mean(user_audio, axis=1)
            
            # 重采样到 24kHz
            if user_sr != 24000:
                import librosa
                user_audio = librosa.resample(
                    user_audio,
                    orig_sr=user_sr,
                    target_sr=24000
                )
                user_sr = 24000
        
        # 处理文本提示
        if not text_prompt or text_prompt.strip() == "":
            text_prompt = "You are a helpful AI assistant. Respond naturally and conversationally."
        
        # 处理语音提示（如果提供）
        voice_prompt_audio = None
        if voice_prompt_file:
            voice_prompt_audio, _ = sf.read(voice_prompt_file)
            if len(voice_prompt_audio.shape) > 1:
                voice_prompt_audio = np.mean(voice_prompt_audio, axis=1)
        
        print(f"处理输入: 音频长度={len(user_audio)/user_sr:.2f}秒, 文本='{text_prompt}'")
        
        # 准备输入
        inputs = processor(
            audio=user_audio,
            sampling_rate=user_sr,
            text=text_prompt,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 生成输出
        print("执行推理...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码文本输出
        if hasattr(processor, 'decode'):
            text_output = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            text_output = "推理完成（无法解码文本）"
        
        # 尝试提取音频输出
        output_audio = None
        if hasattr(outputs, 'audio_values'):
            output_audio = outputs.audio_values.cpu().numpy()
            if len(output_audio.shape) > 1:
                output_audio = output_audio[0] if output_audio.shape[0] == 1 else output_audio
        elif isinstance(outputs, dict) and 'audio_values' in outputs:
            output_audio = outputs['audio_values'].cpu().numpy()
        
        # 如果没有音频输出，生成一个占位音频
        if output_audio is None:
            # 创建一个简单的提示音
            sample_rate = 24000
            duration = 2.0
            output_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
            output_audio = output_audio.astype(np.float32)
            text_output += "\n\n⚠️ 注意: 音频输出不可用，已生成占位音频"
        
        # 保存输出音频到临时文件
        output_audio_path = "/tmp/personaplex_output.wav"
        sf.write(output_audio_path, output_audio, 24000)
        
        status = f"✅ 推理完成！\n文本输出: {text_output[:100]}..."
        
        return output_audio_path, status, text_output
        
    except Exception as e:
        error_msg = f"❌ 处理失败: {str(e)}"
        import traceback
        traceback.print_exc()
        return None, error_msg, None

def get_system_info():
    """获取系统信息"""
    info = "## 系统信息\n\n"
    
    if torch.cuda.is_available():
        info += f"**GPU**: {torch.cuda.get_device_name(0)}\n"
        info += f"**显存**: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n"
        if model is not None:
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            info += f"**已用显存**: {memory_used:.2f} GB / {memory_reserved:.2f} GB\n"
    else:
        info += "**设备**: CPU（不推荐）\n"
    
    info += f"\n**模型状态**: {'✅ 已加载' if model is not None else '❌ 未加载'}\n"
    info += f"**模型 ID**: {MODEL_ID}\n"
    
    return info

# 创建 Gradio 界面
def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="PersonaPlex 测试界面", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ PersonaPlex 模型测试界面
        
        这是一个用于测试 NVIDIA PersonaPlex-7b-v1 模型的简易 Web 界面。
        
        **使用说明:**
        1. 点击"加载模型"按钮（首次使用需要一些时间）
        2. 上传音频文件（可选，24kHz WAV 格式推荐）
        3. 输入文本提示（定义 AI 的角色和风格）
        4. 点击"生成响应"按钮
        5. 查看文本输出和播放音频输出
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 模型控制")
                load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
                model_status = gr.Textbox(
                    label="模型状态",
                    value="❌ 模型未加载",
                    interactive=False
                )
                system_info = gr.Markdown(get_system_info())
                
            with gr.Column(scale=2):
                gr.Markdown("### 输入设置")
                
                audio_input = gr.Audio(
                    label="用户音频输入 (可选)",
                    type="filepath",
                    sources=["upload", "microphone"],
                    format="wav"
                )
                
                text_prompt = gr.Textbox(
                    label="文本提示 (角色设定)",
                    placeholder="例如: You are a friendly customer service agent. Be helpful and professional.",
                    value="You are a helpful AI assistant. Respond naturally and conversationally.",
                    lines=3
                )
                
                voice_prompt = gr.Audio(
                    label="参考语音提示 (可选，用于控制音色)",
                    type="filepath",
                    sources=["upload"],
                    format="wav"
                )
                
                generate_btn = gr.Button("🚀 生成响应", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 输出结果")
                output_audio = gr.Audio(
                    label="AI 语音输出",
                    type="filepath",
                    format="wav"
                )
                output_text = gr.Textbox(
                    label="AI 文本输出",
                    lines=5,
                    interactive=False
                )
                status_output = gr.Textbox(
                    label="状态信息",
                    lines=3,
                    interactive=False
                )
        
        # 事件绑定
        load_btn.click(
            fn=load_model_once,
            outputs=model_status
        ).then(
            fn=get_system_info,
            outputs=system_info
        )
        
        generate_btn.click(
            fn=process_audio,
            inputs=[audio_input, text_prompt, voice_prompt],
            outputs=[output_audio, status_output, output_text]
        )
        
        gr.Markdown("""
        ---
        ### 📝 提示
        
        - **音频格式**: 推荐使用 24kHz WAV 格式
        - **文本提示**: 用英文描述 AI 的角色、性格和说话风格
        - **首次加载**: 模型较大，首次加载可能需要 2-5 分钟
        - **显存要求**: 至少需要 16GB VRAM（推荐 24GB+）
        """)
    
    return demo

def main():
    """主函数"""
    print("="*60)
    print("启动 PersonaPlex Web 界面")
    print("="*60)
    
    if not HF_TOKEN:
        print("⚠️  警告: HF_TOKEN 未设置")
        print("   某些功能可能无法使用")
    
    demo = create_interface()
    
    # 启动服务器
    # 在 RunPod 上，使用 share=False 并绑定到 0.0.0.0
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # Gradio 默认端口
        share=False,            # 不创建公共链接（RunPod 有自己的 URL）
        show_error=True
    )

if __name__ == "__main__":
    main()

