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
        print("="*60)
        print("开始加载模型")
        print("="*60)
        
        # 检查环境
        print("\n[DEBUG] 检查环境...")
        print(f"[DEBUG] MODEL_ID: {MODEL_ID}")
        print(f"[DEBUG] HF_TOKEN: {'已设置' if HF_TOKEN else '未设置'}")
        print(f"[DEBUG] Device: {device}")
        
        if HF_TOKEN:
            print("[DEBUG] 登录 Hugging Face...")
            login(token=HF_TOKEN)
            print("[DEBUG] 登录成功")
        else:
            print("[DEBUG] ⚠️  HF_TOKEN 未设置，可能无法访问 gated repo")
        
        model_status = "加载中..."
        
        # 检查 Transformers 版本和配置
        import transformers
        from transformers import AutoConfig
        transformers_version = transformers.__version__
        print(f"\n[DEBUG] Transformers 版本: {transformers_version}")
        print(f"[DEBUG] Transformers 路径: {transformers.__file__}")
        
        # 尝试加载配置（直接下载文件，不通过 AutoConfig）
        print("\n[DEBUG] 步骤1: 检查模型配置和自定义代码...")
        try:
            from huggingface_hub import hf_hub_download
            import json
            
            # 直接下载 config.json
            print("[DEBUG] 直接下载 config.json...")
            config_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename="config.json",
                token=HF_TOKEN
            )
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            print(f"[DEBUG] ✅ 配置文件下载成功")
            print(f"[DEBUG] - Model type: {config_data.get('model_type', 'N/A')}")
            print(f"[DEBUG] - Architectures: {config_data.get('architectures', 'N/A')}")
            print(f"[DEBUG] - Auto map: {config_data.get('auto_map', 'N/A')}")
            
            # 检查是否有自定义代码
            auto_map = config_data.get('auto_map', {})
            if auto_map:
                print(f"[DEBUG] ✅ 发现自定义代码映射: {auto_map}")
                
                # 检查是否有 modeling 文件
                if 'AutoModel' in auto_map or 'AutoModelForConditionalGeneration' in auto_map:
                    model_file = auto_map.get('AutoModel') or auto_map.get('AutoModelForConditionalGeneration')
                    print(f"[DEBUG] 自定义模型文件: {model_file}")
                    
                    # 尝试下载自定义代码文件
                    try:
                        custom_code_path = hf_hub_download(
                            repo_id=MODEL_ID,
                            filename=model_file,
                            token=HF_TOKEN
                        )
                        print(f"[DEBUG] ✅ 自定义代码文件下载成功: {custom_code_path}")
                    except Exception as e:
                        print(f"[DEBUG] ⚠️  自定义代码文件下载失败: {e}")
            else:
                print("[DEBUG] ⚠️  未找到 auto_map，可能需要手动处理")
                
        except Exception as e:
            print(f"[DEBUG] ⚠️  配置检查失败（继续尝试加载）: {e}")
            import traceback
            traceback.print_exc()
        
        # 尝试多种加载方式
        print("\n[DEBUG] 步骤2: 尝试加载模型...")
        
        # 方法1: 使用 AutoModel + trust_remote_code（绕过配置检查）
        print("[DEBUG] 方法1: 使用 AutoModel.from_pretrained + trust_remote_code=True")
        print("[DEBUG] 注意: 即使配置加载失败，也尝试直接加载模型（trust_remote_code 应该会处理自定义代码）")
        try:
            # 直接尝试加载，让 trust_remote_code 处理自定义代码
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,  # 关键：这会自动下载并执行自定义代码
                local_files_only=False  # 确保从远程下载自定义代码
            )
            print("[DEBUG] ✅ 方法1成功: AutoModel 加载成功")
        except Exception as e1:
            print(f"[DEBUG] ❌ 方法1失败: {type(e1).__name__}: {e1}")
            import traceback
            traceback.print_exc()
            
            # 方法2: 尝试手动加载自定义代码
            print("\n[DEBUG] 方法2: 尝试手动加载自定义代码...")
            try:
                # 检查是否有自定义代码文件
                from huggingface_hub import list_repo_files
                
                print("[DEBUG] 列出模型仓库文件...")
                repo_files = list_repo_files(
                    repo_id=MODEL_ID,
                    token=HF_TOKEN
                )
                print(f"[DEBUG] 仓库文件: {[f for f in repo_files if '.py' in f]}")
                
                # 查找 modeling 文件
                modeling_files = [f for f in repo_files if 'modeling' in f.lower() and f.endswith('.py')]
                if modeling_files:
                    print(f"[DEBUG] 找到建模文件: {modeling_files}")
                    # 尝试手动下载并导入
                    for model_file in modeling_files:
                        try:
                            print(f"[DEBUG] 尝试下载并导入: {model_file}")
                            custom_path = hf_hub_download(
                                repo_id=MODEL_ID,
                                filename=model_file,
                                token=HF_TOKEN
                            )
                            print(f"[DEBUG] 自定义代码路径: {custom_path}")
                            # 这里可以尝试动态导入，但比较复杂
                        except Exception as e:
                            print(f"[DEBUG] 下载 {model_file} 失败: {e}")
                
            except Exception as e2:
                print(f"[DEBUG] ⚠️  方法2失败: {e2}")
            
            # 方法3: 尝试使用 MoshiForConditionalGeneration（作为回退）
            print("\n[DEBUG] 方法3: 尝试使用 MoshiForConditionalGeneration...")
            try:
                from transformers import MoshiForConditionalGeneration
                model = MoshiForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                print("[DEBUG] ✅ 方法3成功: MoshiForConditionalGeneration 加载成功")
            except Exception as e3:
                print(f"[DEBUG] ❌ 方法3失败: {type(e3).__name__}: {e3}")
                import traceback
                traceback.print_exc()
                
                # 最终错误处理
                error_msg = str(e1)
                return f"""❌ 模型加载失败

当前 Transformers 版本: {transformers_version}
主要错误: {error_msg}

已尝试的方法:
1. AutoModel.from_pretrained + trust_remote_code=True
2. 手动检查自定义代码文件
3. MoshiForConditionalGeneration

问题分析:
PersonaPlex 使用自定义架构，需要从模型仓库加载自定义代码。
但 Transformers 在加载配置时就失败了，无法继续。

解决方案:
由于 PersonaPlex 架构太新，当前 Transformers 版本可能还不完全支持。
建议:
1. 等待 Transformers 更新支持 PersonaPlex
2. 或使用官方 PersonaPlex 代码库: https://github.com/NVIDIA/personaplex
3. 或手动实现模型加载逻辑"""
                
                raise e1
        
        # 验证模型
        print("\n[DEBUG] 步骤3: 验证模型...")
        if model is None:
            raise Exception("模型加载失败，model 为 None")
        
        print(f"[DEBUG] 模型类型: {type(model).__name__}")
        print(f"[DEBUG] 模型设备: {next(model.parameters()).device if hasattr(model, 'parameters') else 'N/A'}")
        
        model.eval()
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        model_status = "已加载"
        
        print(f"\n[DEBUG] ✅ 模型加载完成！显存: {mem:.2f} GB")
        print("="*60)
        
        return f"✅ 模型加载成功！({mem:.2f} GB)"
        
    except Exception as e:
        model_status = "加载失败"
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"\n[DEBUG] ❌ 最终错误: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        return f"""❌ 模型加载失败

错误类型: {error_type}
错误信息: {error_msg}

请查看控制台日志获取详细调试信息。"""

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
    
    # 文本提示（可选）
    text_prompt_input = gr.Textbox(
        label="文本提示（可选）",
        placeholder="例如: You are a helpful AI assistant.",
        lines=2,
        value="You are a helpful AI assistant. Respond naturally."
    )
    
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
    
    # 语音处理（包含文本提示）
    audio_input.change(
        fn=process_voice,
        inputs=[audio_input, text_prompt_input],
        outputs=[user_text, ai_text]
    )
    
    # 可选：如果文本提示改变，也触发处理（如果音频已存在）
    # text_prompt_input.change(
    #     fn=lambda prompt, audio: process_voice(audio, prompt) if audio else ("", ""),
    #     inputs=[text_prompt_input, audio_input],
    #     outputs=[user_text, ai_text]
    # )

if __name__ == "__main__":
    print("="*60)
    print("PersonaPlex 语音对话 - 端口 5001")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=5001,
        share=False
    )

