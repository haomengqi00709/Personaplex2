#!/usr/bin/env python3
"""
PersonaPlex 简单语音对话 - 使用官方正确方式
基于官方代码库的实现
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
    # 方式1: 当前目录下的 personaplex/moshi
    os.path.join(os.path.dirname(__file__), 'personaplex', 'moshi'),
    # 方式2: 上级目录的 personaplex/moshi (RunPod 上克隆到 /workspace)
    os.path.join(os.path.dirname(__file__), '..', 'personaplex', 'moshi'),
    # 方式3: /workspace/personaplex/moshi (RunPod 常见路径)
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
    print("\n请按以下步骤安装:")
    print("1. 克隆官方代码库: git clone https://github.com/NVIDIA/personaplex.git")
    print("2. 安装 moshi 包: cd personaplex/moshi && pip install -e .")
    OFFICIAL_MOSHI_AVAILABLE = False

MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 全局变量
mimi = None
other_mimi = None
lm = None
lm_gen = None
text_tokenizer = None
model_loading = False
model_status = "未加载"

def load_model():
    """加载模型 - 使用官方方式"""
    global mimi, other_mimi, lm, lm_gen, text_tokenizer, model_status
    
    if not OFFICIAL_MOSHI_AVAILABLE:
        return "❌ 官方 moshi 包不可用。请运行: pip install personaplex/moshi/"
    
    if mimi is not None:
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return f"✅ 模型已加载 (显存: {mem:.2f} GB)"
    
    try:
        print("[INFO] 开始加载模型...")
        
        # 1. 加载 Mimi 编码器/解码器
        print("[INFO] 加载 Mimi...")
        from huggingface_hub import hf_hub_download
        mimi_weight = hf_hub_download(MODEL_ID, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device)
        other_mimi = loaders.get_mimi(mimi_weight, device)
        print("[INFO] Mimi 加载完成")
        
        # 2. 加载 tokenizer
        print("[INFO] 加载 tokenizer...")
        import sentencepiece
        tokenizer_path = hf_hub_download(MODEL_ID, loaders.TEXT_TOKENIZER_NAME)
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print("[INFO] Tokenizer 加载完成")
        
        # 3. 加载 Moshi LM
        print("[INFO] 加载 Moshi LM...")
        moshi_weight = hf_hub_download(MODEL_ID, loaders.MOSHI_NAME)
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

def process_voice(audio, text_prompt=None):
    """处理语音并生成回复 - 使用官方方式"""
    global mimi, other_mimi, lm_gen, text_tokenizer
    
    if mimi is None or lm_gen is None:
        return "请先加载模型", "❌ 模型未加载"
    
    if audio is None:
        return "请录制或上传音频", "❌ 无音频输入"
    
    try:
        # 1. 读取音频
        audio_data, sr = sf.read(audio)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 2. 重采样到模型采样率 (24kHz)
        if sr != mimi.sample_rate:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=mimi.sample_rate)
        
        # 3. 转换为 (C, T) 格式
        if len(audio_data.shape) == 1:
            audio_data = audio_data[np.newaxis, :]  # (1, T)
        
        # 4. 使用官方方式编码和处理
        sample_rate = mimi.sample_rate
        user_audio = torch.tensor(audio_data, dtype=torch.float32, device=device)
        
        # 5. 编码用户音频并逐步处理
        generated_frames = []
        generated_text = []
        
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        
        for user_encoded in encode_from_sphn(
            mimi,
            _iterate_audio(
                user_audio.cpu().numpy(), 
                sample_interval_size=frame_size, 
                pad=True
            ),
            max_batch=1,
        ):
            # user_encoded: [1, K=8, T=1]
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]  # [1, 8, 1]
                
                # 调用 LMGen.step() - 这是正确的方式！
                tokens = lm_gen.step(input_tokens=step_in)
                
                if tokens is None:
                    continue
                
                # 解码音频
                pcm = mimi.decode(tokens[:, 1:9])  # 解码 codebooks 1-8
                _ = other_mimi.decode(tokens[:, 1:9])
                pcm = pcm.detach().cpu().numpy()[0, 0]
                generated_frames.append(pcm)
                
                # 解码文本
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    generated_text.append(_text)
        
        # 6. 拼接生成的音频
        if len(generated_frames) == 0:
            return "未生成音频", "❌ 未生成任何音频帧"
        
        output_audio = np.concatenate(generated_frames, axis=-1)
        
        # 7. 保存输出音频
        output_path = "/tmp/personaplex_output.wav"
        try:
            import sphn
            sphn.write_wav(output_path, output_audio, mimi.sample_rate)
        except ImportError:
            # 如果没有 sphn，使用 soundfile
            sf.write(output_path, output_audio, mimi.sample_rate)
        
        # 8. 生成文本输出
        user_text = "用户语音输入"
        ai_text = " ".join(generated_text) if generated_text else "AI 回复（无文本）"
        
        return user_text, ai_text, output_path
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return "处理失败", error_msg, None

# Gradio 界面
with gr.Blocks(title="PersonaPlex 语音对话") as demo:
    gr.Markdown("# PersonaPlex 语音对话")
    gr.Markdown("使用官方实现方式的语音对话界面")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 控制")
            load_btn = gr.Button("加载模型", variant="primary")
            status_text = gr.Textbox(label="状态", value="未加载", interactive=False)
            
            gr.Markdown("### 输入")
            audio_input = gr.Audio(label="录制或上传音频", type="filepath")
            text_prompt_input = gr.Textbox(
                label="文本提示（可选）", 
                value="You enjoy having a good conversation.",
                placeholder="输入系统提示..."
            )
            process_btn = gr.Button("处理", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 输出")
            user_text_output = gr.Textbox(label="用户输入", interactive=False)
            ai_text_output = gr.Textbox(label="AI 回复", interactive=False)
            audio_output = gr.Audio(label="AI 音频回复", type="filepath")
    
    load_btn.click(load_model, outputs=status_text)
    process_btn.click(
        process_voice,
        inputs=[audio_input, text_prompt_input],
        outputs=[user_text_output, ai_text_output, audio_output]
    )
    
    # 自动加载模型
    demo.load(load_model, outputs=status_text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5001)

