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

def encode_audio_manual(audio_data, sample_rate=24000):
    """手动编码音频（不依赖 processor）"""
    try:
        print("[DEBUG] 开始手动编码音频...")
        
        # 1. 确保音频是单声道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 2. 归一化到 [-1, 1]
        max_val = np.abs(audio_data).max()
        if max_val > 1.0 or max_val == 0:
            if max_val > 0:
                audio_data = audio_data / max_val
            else:
                audio_data = audio_data.astype(np.float32)
        else:
            audio_data = audio_data.astype(np.float32)
        
        # 3. 转换为 tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        print(f"[DEBUG] 音频编码完成: shape={audio_tensor.shape}, dtype={audio_tensor.dtype}")
        
        return audio_tensor
        
    except Exception as e:
        print(f"[DEBUG] 音频编码失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_voice(audio, text_prompt=None):
    """处理语音并生成回复"""
    global model
    
    if model is None:
        return "❌ 请先加载模型", "❌ 模型未加载"
    
    if audio is None:
        return "", ""
    
    # 处理音频路径（Gradio 可能传递字典）
    if isinstance(audio, dict):
        audio_path = audio.get('path', audio.get('url', None))
        if audio_path is None:
            return "", ""
    else:
        audio_path = audio
    
    # 设置默认文本提示
    if text_prompt is None or text_prompt.strip() == "":
        text_prompt = "You are a helpful AI assistant. Respond naturally."
    
    try:
        print("\n[DEBUG] 开始处理语音...")
        print(f"[DEBUG] 音频路径: {audio_path}")
        print(f"[DEBUG] 文本提示: {text_prompt}")
        
        # 读取音频
        audio_data, sr = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 重采样到 24kHz
        if sr != 24000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
            sr = 24000
        
        duration = len(audio_data) / sr
        user_text = f"🎤 语音输入 ({duration:.2f}秒, {len(audio_data)} 采样点)"
        print(f"[DEBUG] 处理后音频: {len(audio_data)} 采样点, {sr}Hz, {duration:.2f}秒")
        
        # 尝试调用模型进行推理
        try:
            # 编码音频
            print("[DEBUG] 编码音频...")
            audio_tensor = encode_audio_manual(audio_data, sr)
            
            if audio_tensor is None:
                raise Exception("音频编码失败")
            
            print(f"[DEBUG] 文本提示: {text_prompt}")
            print(f"[DEBUG] 音频 tensor 形状: {audio_tensor.shape}")
            
            # 尝试调用模型
            print("[DEBUG] 调用模型进行推理...")
            with torch.no_grad():
                # 检查模型是否有 generate 方法
                if hasattr(model, 'generate'):
                    print("[DEBUG] 使用 generate 方法...")
                    try:
                        # 尝试构建输入
                        audio_input = audio_tensor.unsqueeze(0).to(device)
                        
                        # 如果模型有 encode_audio 方法
                        if hasattr(model, 'encode_audio'):
                            print("[DEBUG] 使用 encode_audio 方法...")
                            encoded_audio = model.encode_audio(audio_input)
                            print(f"[DEBUG] 编码后形状: {encoded_audio.shape if hasattr(encoded_audio, 'shape') else 'N/A'}")
                        
                        # 尝试生成
                        print("[DEBUG] 尝试生成回复...")
                        
                        # 检查模型的实际结构
                        print("[DEBUG] 检查模型结构...")
                        print(f"[DEBUG] 模型类型: {type(model).__name__}")
                        print(f"[DEBUG] 模型属性: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")
                        
                        # 尝试查看模型的 forward 方法
                        import inspect
                        if hasattr(model, 'forward'):
                            sig = inspect.signature(model.forward)
                            print(f"[DEBUG] Forward 签名: {sig}")
                            print(f"[DEBUG] Forward 参数: {list(sig.parameters.keys())}")
                        
                        # 尝试直接调用模型进行推理
                        try:
                            # 检查模型是否有音频编码器
                            if hasattr(model, 'audio_encoder'):
                                print("[DEBUG] 发现 audio_encoder，尝试编码音频...")
                                try:
                                    # 使用 audio_encoder 编码音频
                                    # 注意：audio_encoder 可能需要特定的输入格式
                                    encoded_result = model.audio_encoder(audio_input)
                                    
                                    # 处理编码结果（可能是元组）
                                    if isinstance(encoded_result, tuple):
                                        print(f"[DEBUG] 编码结果: 元组，长度={len(encoded_result)}")
                                        for i, item in enumerate(encoded_result):
                                            if hasattr(item, 'shape'):
                                                print(f"[DEBUG] 编码结果[{i}]: shape={item.shape}")
                                        # 通常第一个是编码后的值，第二个是 codes
                                        user_audio_codes = encoded_result[0] if len(encoded_result) > 0 else None
                                        if len(encoded_result) > 1:
                                            user_audio_codes = encoded_result[1]  # codes 通常在第二个位置
                                    else:
                                        print(f"[DEBUG] 编码结果: shape={encoded_result.shape if hasattr(encoded_result, 'shape') else type(encoded_result)}")
                                        user_audio_codes = encoded_result
                                    
                                    print(f"[DEBUG] 使用 user_audio_codes: shape={user_audio_codes.shape if hasattr(user_audio_codes, 'shape') else 'N/A'}")
                                    
                                except Exception as encode_error:
                                    print(f"[DEBUG] audio_encoder 调用失败: {encode_error}")
                                    import traceback
                                    traceback.print_exc()
                                    # 如果 audio_encoder 失败，尝试直接使用原始音频
                                    user_audio_codes = None
                                    user_input_values = audio_input
                                    print("[DEBUG] 回退到使用 user_input_values")
                            else:
                                # 没有 audio_encoder，直接使用原始音频值
                                user_input_values = audio_input
                                user_audio_codes = None
                                print("[DEBUG] 没有 audio_encoder，使用 user_input_values")
                            
                            # 准备文本输入（如果需要）
                            # 对于 PersonaPlex，可能需要将文本提示转换为 input_ids
                            input_ids = None
                            if text_prompt:
                                print(f"[DEBUG] 处理文本提示: {text_prompt}")
                                # 尝试使用 tokenizer（如果模型有）
                                if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                                    try:
                                        input_ids = model.tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
                                        print(f"[DEBUG] 文本 input_ids: shape={input_ids.shape}")
                                    except:
                                        print("[DEBUG] tokenizer 不可用，跳过文本输入")
                            
                            # 尝试调用 generate 方法
                            print("[DEBUG] 尝试调用 generate 方法...")
                            try:
                                # 构建 generate 的输入
                                generate_kwargs = {}
                                
                                if user_audio_codes is not None:
                                    generate_kwargs['user_audio_codes'] = user_audio_codes
                                elif user_input_values is not None:
                                    generate_kwargs['user_input_values'] = user_input_values
                                
                                if input_ids is not None:
                                    generate_kwargs['input_ids'] = input_ids
                                
                                print(f"[DEBUG] Generate 参数: {list(generate_kwargs.keys())}")
                                
                                # 调用 generate
                                outputs = model.generate(
                                    **generate_kwargs,
                                    max_new_tokens=128,
                                    temperature=0.7,
                                    do_sample=True
                                )
                                
                                print(f"[DEBUG] Generate 成功！输出形状: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
                                
                                # 尝试解码输出
                                output_text = "推理完成"
                                if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                                    try:
                                        output_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                                        print(f"[DEBUG] 解码文本: {output_text[:100]}")
                                    except:
                                        pass
                                
                                ai_text = f"""✅ 推理成功！

📊 处理信息:
- 音频长度: {duration:.2f}秒
- 采样点数: {len(audio_data)}
- 文本提示: {text_prompt}

🤖 AI 回复:
{output_text}

✅ 模型推理完成！"""
                                
                            except Exception as gen_error:
                                print(f"[DEBUG] Generate 失败: {gen_error}")
                                import traceback
                                traceback.print_exc()
                                
                                # 如果 generate 失败，尝试 forward
                                print("[DEBUG] 尝试使用 forward 方法...")
                                try:
                                    forward_kwargs = {}
                                    if user_audio_codes is not None:
                                        forward_kwargs['user_audio_codes'] = user_audio_codes
                                    elif user_input_values is not None:
                                        forward_kwargs['user_input_values'] = user_input_values
                                    
                                    if input_ids is not None:
                                        forward_kwargs['input_ids'] = input_ids
                                    
                                    forward_output = model.forward(**forward_kwargs)
                                    print(f"[DEBUG] Forward 成功！输出类型: {type(forward_output)}")
                                    
                                    ai_text = f"""✅ 模型调用成功！

📊 处理信息:
- 音频长度: {duration:.2f}秒
- 模型类型: {type(model).__name__}

🔧 使用 forward 方法调用成功。
输出类型: {type(forward_output).__name__}

⚠️ 注意: 需要进一步处理输出以获取文本/音频回复。"""
                                    
                                except Exception as forward_error:
                                    print(f"[DEBUG] Forward 也失败: {forward_error}")
                                    import traceback
                                    traceback.print_exc()
                                    ai_text = f"""✅ 音频已处理

📊 处理信息:
- 音频长度: {duration:.2f}秒
- 模型类型: {type(model).__name__}

⚠️ 模型调用失败:
- Generate 错误: {str(gen_error)[:100]}
- Forward 错误: {str(forward_error)[:100]}

请查看控制台日志获取详细错误信息。"""
                            
                        except Exception as forward_error:
                            print(f"[DEBUG] 整体调用失败: {forward_error}")
                            import traceback
                            traceback.print_exc()
                            ai_text = f"""✅ 音频已处理

📊 处理信息:
- 音频长度: {duration:.2f}秒
- 模型类型: {type(model).__name__}
- 错误: {str(forward_error)}

⚠️ 需要根据模型文档构建正确的输入格式。"""
                        
                    except Exception as e:
                        print(f"[DEBUG] 推理尝试失败: {e}")
                        import traceback
                        traceback.print_exc()
                        ai_text = f"✅ 音频已处理\n\n⚠️ 推理失败: {str(e)}\n\n可能需要特定的输入格式。"
                else:
                    print("[DEBUG] 模型没有 generate 方法，尝试 forward...")
                    # 尝试使用 forward 方法
                    try:
                        # 检查模型的 forward 签名
                        import inspect
                        sig = inspect.signature(model.forward)
                        print(f"[DEBUG] Forward 方法签名: {sig}")
                        print(f"[DEBUG] Forward 参数: {list(sig.parameters.keys())}")
                        
                        # 尝试查看模型的主要组件
                        print("[DEBUG] 检查模型组件...")
                        model_components = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr, None))]
                        print(f"[DEBUG] 模型组件: {model_components[:15]}")
                        
                        ai_text = f"""✅ 音频已处理

📊 处理信息:
- 音频长度: {duration:.2f}秒
- 采样点数: {len(audio_data)}
- 采样率: {sr}Hz

🔧 模型信息:
- 模型类型: {type(model).__name__}
- Forward 参数: {list(sig.parameters.keys())}
- 模型组件: {', '.join(model_components[:10])}

⚠️ 需要根据 forward 方法的参数构建正确的输入。
请查看控制台日志获取详细调试信息。"""
                    except Exception as e:
                        print(f"[DEBUG] 检查模型结构失败: {e}")
                        import traceback
                        traceback.print_exc()
                        ai_text = f"✅ 音频已处理\n\n⚠️ 无法确定模型输入格式。\n错误: {str(e)}"
            
        except Exception as e:
            print(f"[DEBUG] 模型调用失败: {e}")
            import traceback
            traceback.print_exc()
            ai_text = f"✅ 音频已处理\n\n⚠️ 模型调用需要特定格式。\n错误: {str(e)}"
        
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

