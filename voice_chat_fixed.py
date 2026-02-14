#!/usr/bin/env python3
"""
PersonaPlex 简单语音对话 - 使用官方正确方式
只有一个按钮：说话
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# 尝试多种方式导入官方 moshi 包
OFFICIAL_MOSHI_AVAILABLE = False
moshi_paths = [
    os.path.join(os.path.dirname(__file__), 'personaplex', 'moshi'),
    os.path.join(os.path.dirname(__file__), '..', 'personaplex', 'moshi'),
    '/workspace/personaplex/moshi',
]

for moshi_path in moshi_paths:
    if os.path.exists(moshi_path):
        sys.path.insert(0, moshi_path)
        print(f"[INFO] 找到 moshi 包路径: {moshi_path}")
        break

try:
    from moshi.models import loaders, LMGen, MimiModel
    from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn
    from moshi.client_utils import make_log
    OFFICIAL_MOSHI_AVAILABLE = True
    print("[INFO] ✅ 成功导入官方 moshi 包")
except ImportError as e:
    print(f"⚠️ 无法导入官方 moshi 包: {e}")
    OFFICIAL_MOSHI_AVAILABLE = False

MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 验证 Token 是否设置
if not HF_TOKEN:
    print("⚠️ 警告: 未设置 HF_TOKEN 环境变量")
else:
    print(f"[INFO] HF_TOKEN 已设置 (长度: {len(HF_TOKEN)})")
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("[INFO] ✅ 预认证成功")
    except Exception as e:
        print(f"[INFO] ⚠️ 预认证警告: {e}")

# 全局变量
mimi = None
other_mimi = None
lm = None
lm_gen = None
text_tokenizer = None
model_status = "未加载"

def load_model():
    """加载模型 - 使用官方方式"""
    global mimi, other_mimi, lm, lm_gen, text_tokenizer, model_status
    
    if not OFFICIAL_MOSHI_AVAILABLE:
        return "❌ 官方 moshi 包不可用"
    
    if mimi is not None:
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return f"✅ 模型已加载 (显存: {mem:.2f} GB)"
    
    try:
        print("[INFO] 开始加载模型...")
        
        # 0. 确保 Hugging Face 认证
        if HF_TOKEN:
            from huggingface_hub import login
            try:
                login(token=HF_TOKEN, add_to_git_credential=False)
                print("[INFO] ✅ Hugging Face 认证成功")
            except Exception as e:
                print(f"[INFO] ⚠️ 认证警告: {e}")
        
        # 1. 加载 Mimi 编码器/解码器
        print("[INFO] 加载 Mimi...")
        from huggingface_hub import hf_hub_download
        mimi_weight = hf_hub_download(MODEL_ID, loaders.MIMI_NAME, token=HF_TOKEN)
        mimi = loaders.get_mimi(mimi_weight, device)
        other_mimi = loaders.get_mimi(mimi_weight, device)
        print("[INFO] Mimi 加载完成")
        
        # 2. 加载 tokenizer
        print("[INFO] 加载 tokenizer...")
        import sentencepiece
        tokenizer_path = hf_hub_download(MODEL_ID, loaders.TEXT_TOKENIZER_NAME, token=HF_TOKEN)
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print("[INFO] Tokenizer 加载完成")
        
        # 3. 加载 Moshi LM
        print("[INFO] 加载 Moshi LM...")
        moshi_weight = hf_hub_download(MODEL_ID, loaders.MOSHI_NAME, token=HF_TOKEN)
        lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=False)
        lm.eval()
        print("[INFO] Moshi LM 加载完成")
        
        # 4. 创建 LMGen
        print("[INFO] 创建 LMGen...")
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=device,
            frame_rate=mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        
        # 设置流式模式
        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)
        
        # 5. Warmup
        print("[INFO] 预热模型...")
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
            codes = mimi.encode(chunk)
            _ = other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c : c + 1])
                if tokens is not None:
                    _ = mimi.decode(tokens[:, 1:9])
                    _ = other_mimi.decode(tokens[:, 1:9])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[INFO] 预热完成")
        
        # 6. 初始化系统提示
        print("[INFO] 初始化系统提示...")
        text_prompt = "You enjoy having a good conversation."
        from moshi.offline import wrap_with_system_tags
        lm_gen.text_prompt_tokens = (
            text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
        )
        
        # 重置流式状态并运行系统提示
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        
        model_status = "已加载"
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return f"✅ 模型加载成功！\n显存使用: {mem:.2f} GB"
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 模型加载失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

def process_voice(audio):
    """处理语音并生成回复 - 使用官方方式"""
    global mimi, other_mimi, lm_gen, text_tokenizer
    
    if mimi is None or lm_gen is None:
        return "请先加载模型", "❌ 模型未加载"
    
    if audio is None:
        return "请说话", "❌ 没有检测到音频"
    
    try:
        # 1. 重置流式状态（重要：确保每次对话都是新的状态）
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        
        # 2. 读取音频
        audio_data, sr = sf.read(audio)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 3. 重采样到模型采样率 (24kHz)
        if sr != mimi.sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=mimi.sample_rate)
        
        # 4. 转换为 (C, T) 格式
        if len(audio_data.shape) == 1:
            audio_data = audio_data[np.newaxis, :]  # (1, T)
        
        # 5. 使用官方方式编码和处理
        user_audio = torch.tensor(audio_data, dtype=torch.float32, device=device)
        generated_frames = []
        generated_text = []
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        
        # 按照官方方式：处理用户输入的同时生成回复
        # 模型会在处理用户输入的过程中开始生成回复
        for user_encoded in encode_from_sphn(
            mimi,
            _iterate_audio(user_audio.cpu().numpy(), sample_interval_size=frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]  # [1, 8, 1]
                tokens = lm_gen.step(input_tokens=step_in)
                
                if tokens is None:
                    continue
                
                # 解码音频
                pcm = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
                pcm = pcm.detach().cpu().numpy()[0, 0]
                generated_frames.append(pcm)
                
                # 解码文本
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    generated_text.append(_text)
        
        # 继续生成回复（用户输入处理完后，继续生成直到有足够的回复）
        # 使用静音输入继续生成，直到模型停止生成或达到最大长度
        silence_count = 0
        max_silence = 30  # 最多允许30帧静音后停止
        max_additional_frames = 150  # 最多额外生成150帧
        
        for _ in range(max_additional_frames):
            # 使用静音输入继续生成
            silent_input = torch.zeros(1, 8, 1, dtype=torch.float32, device=device)
            tokens = lm_gen.step(input_tokens=silent_input)
            
            if tokens is None:
                silence_count += 1
                if silence_count > max_silence:
                    break
                continue
            
            # 解码音频
            pcm = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
            pcm = pcm.detach().cpu().numpy()[0, 0]
            generated_frames.append(pcm)
            
            # 检查是否是静音
            pcm_abs = np.abs(pcm)
            if np.max(pcm_abs) < 1e-6:
                silence_count += 1
                if silence_count > max_silence:
                    break
            else:
                silence_count = 0  # 有音频内容，重置静音计数
            
            # 解码文本
            text_token = tokens[0, 0, 0].item()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)
                _text = _text.replace("▁", " ")
                generated_text.append(_text)
        
        if len(generated_frames) == 0:
            return "未生成音频", "❌ 未生成任何音频帧"
        
        # 移除前面的静音帧
        # 找到第一个有实际音频内容的帧
        start_idx = 0
        for i, frame in enumerate(generated_frames):
            if np.max(np.abs(frame)) > 1e-6:
                start_idx = i
                break
        
        if start_idx > 0:
            generated_frames = generated_frames[start_idx:]
            print(f"[INFO] 移除了前 {start_idx} 帧静音")
        
        # 保存音频文件
        output_audio = np.concatenate(generated_frames, axis=-1)
        output_path = "/tmp/personaplex_output.wav"
        sf.write(output_path, output_audio, mimi.sample_rate)
        
        # 生成文本输出
        user_text = "🎤 您说了：" + (" ".join(generated_text) if generated_text else "（语音输入）")
        ai_text = "🤖 AI 回复：" + (" ".join(generated_text) if generated_text else "（处理中...）")
        
        return user_text, ai_text
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理失败: {str(e)}"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        return "处理失败", error_msg

# 极简界面 - 只有一个说话按钮
with gr.Blocks(title="PersonaPlex 语音对话") as demo:
    gr.Markdown("# 🎤 PersonaPlex 语音对话")
    gr.Markdown("点击下方按钮开始说话")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 您说的话")
            user_text = gr.Textbox(label="", value="等待您说话...", interactive=False, lines=8)
        
        with gr.Column(scale=1):
            gr.Markdown("### AI 回复")
            ai_text = gr.Textbox(label="", value="等待 AI 回复...", interactive=False, lines=8)
    
    # 只有一个音频输入组件（自动录音）
    audio_input = gr.Audio(
        label="", 
        type="filepath", 
        sources=["microphone"],
        show_label=False
    )
    
    status = gr.Textbox(label="状态", value="正在加载模型...", interactive=False, visible=False)
    
    # 自动加载模型
    demo.load(load_model, outputs=status)
    
    # 音频输入变化时自动处理
    audio_input.change(
        process_voice,
        inputs=[audio_input],
        outputs=[user_text, ai_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5001)
