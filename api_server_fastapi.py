#!/usr/bin/env python3
"""
PersonaPlex API 服务器 - 使用 FastAPI（替代 Flask）
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import warnings
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import traceback
import uvicorn

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

# 设置 PyTorch CUDA 内存分配策略
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
model_loaded = False

app = FastAPI(title="PersonaPlex API")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """加载模型 - 使用官方方式"""
    global mimi, other_mimi, lm, lm_gen, text_tokenizer, model_loaded
    
    if not OFFICIAL_MOSHI_AVAILABLE:
        return False, "官方 moshi 包不可用"
    
    if model_loaded:
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return True, f"模型已加载 (显存: {mem:.2f} GB)"
    
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
        
        model_loaded = True
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        print(f"[INFO] ✅ 模型加载成功！显存使用: {mem:.2f} GB")
        return True, f"模型加载成功！显存使用: {mem:.2f} GB"
        
    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}\n\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        return False, error_msg

def process_voice(audio_path):
    """处理语音并生成回复 - 使用官方方式"""
    global mimi, other_mimi, lm_gen, text_tokenizer
    
    if not model_loaded or mimi is None or lm_gen is None:
        return None, None, "模型未加载"
    
    try:
        # 0. 清理显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 1. 重置流式状态
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()
        
        # 2. 读取音频（使用多种方法尝试）
        import librosa
        audio_data = None
        sr = None
        
        # 方法1: 尝试使用 librosa（支持 WebM, OGG 等）
        try:
            audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
            print(f"[INFO] 使用 librosa 成功读取音频: {sr}Hz")
        except Exception as e1:
            print(f"[WARN] librosa 读取失败: {e1}")
            
            # 方法2: 尝试使用 soundfile
            try:
                audio_data, sr = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                print(f"[INFO] 使用 soundfile 成功读取音频: {sr}Hz")
            except Exception as e2:
                print(f"[WARN] soundfile 读取失败: {e2}")
                
                # 方法3: 尝试使用 pydub（如果可用）
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(audio_path)
                    audio = audio.set_channels(1)  # 转为单声道
                    audio = audio.set_frame_rate(24000)  # 转为 24kHz
                    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    audio_data = audio_data / (1 << 15)  # 归一化到 [-1, 1]
                    sr = 24000
                    print(f"[INFO] 使用 pydub 成功读取音频: {sr}Hz")
                except Exception as e3:
                    print(f"[WARN] pydub 读取失败: {e3}")
                    raise Exception(f"无法读取音频文件。尝试了 librosa, soundfile, pydub 都失败。最后错误: {e3}")
        
        if audio_data is None or sr is None:
            raise Exception("无法读取音频数据")
        
        # 3. 重采样到模型采样率 (24kHz)
        if sr != mimi.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=mimi.sample_rate)
        
        # 4. 转换为 (C, T) 格式
        if len(audio_data.shape) == 1:
            audio_data = audio_data[np.newaxis, :]  # (1, T)
        
        # 5. 使用官方方式编码和处理
        user_audio = torch.tensor(audio_data, dtype=torch.float32, device=device)
        generated_frames = []
        generated_text = []
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        
        # 处理用户输入的同时生成回复
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
                
                # 解码音频（立即移到 CPU 释放显存）
                pcm = mimi.decode(tokens[:, 1:9])
                _ = other_mimi.decode(tokens[:, 1:9])
                pcm = pcm.detach().cpu().numpy()[0, 0]
                generated_frames.append(pcm)
                
                # 定期清理显存（每10帧清理一次）
                if len(generated_frames) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 解码文本
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    generated_text.append(_text)
        
        # 继续生成回复
        silence_count = 0
        max_silence = 30
        max_additional_frames = 100  # 减少以节省显存
        
        for _ in range(max_additional_frames):
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
            
            # 定期清理显存
            if len(generated_frames) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 检查是否是静音
            pcm_abs = np.abs(pcm)
            if np.max(pcm_abs) < 1e-6:
                silence_count += 1
                if silence_count > max_silence:
                    break
            else:
                silence_count = 0
            
            # 解码文本
            text_token = tokens[0, 0, 0].item()
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)
                _text = _text.replace("▁", " ")
                generated_text.append(_text)
        
        if len(generated_frames) == 0:
            return None, None, "未生成任何音频帧"
        
        # 移除前面的静音帧
        start_idx = 0
        for i, frame in enumerate(generated_frames):
            if np.max(np.abs(frame)) > 1e-6:
                start_idx = i
                break
        
        if start_idx > 0:
            generated_frames = generated_frames[start_idx:]
            print(f"[INFO] 移除了前 {start_idx} 帧静音")
        
        # 最终清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 拼接音频
        output_audio = np.concatenate(generated_frames, axis=-1)
        
        # 保存到临时文件
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(output_file.name, output_audio, mimi.sample_rate)
        output_file.close()
        
        # 生成文本
        user_text = " ".join(generated_text) if generated_text else "（语音输入）"
        ai_text = " ".join(generated_text) if generated_text else "（处理中...）"
        
        return output_file.name, ai_text, None
        
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        return None, None, error_msg

@app.get("/api/status")
async def status():
    """检查模型状态"""
    if model_loaded:
        mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        return {
            'status': 'loaded',
            'memory_gb': round(mem, 2)
        }
    return {'status': 'not_loaded'}

@app.post("/api/load")
async def load():
    """加载模型"""
    success, message = load_model()
    return {
        'success': success,
        'message': message
    }

@app.post("/api/process")
async def process(audio: UploadFile = File(...)):
    """处理音频"""
    if not audio:
        raise HTTPException(status_code=400, detail="没有上传音频文件")
    
    # 读取上传的音频内容
    content = await audio.read()
    
    # 检测文件格式（通过 magic bytes）
    def detect_format(data):
        """通过文件头检测格式"""
        if data.startswith(b'RIFF') and b'WAVE' in data[:12]:
            return '.wav'
        elif data.startswith(b'OggS'):
            return '.ogg'
        elif data.startswith(b'fLaC'):
            return '.flac'
        elif data.startswith(b'\xff\xfb') or data.startswith(b'\xff\xf3'):
            return '.mp3'
        elif data.startswith(b'\x00\x00\x00\x20ftyp') or data.startswith(b'\x00\x00\x00\x18ftyp'):
            return '.m4a'
        elif data.startswith(b'\x1a\x45\xdf\xa3'):  # WebM/Matroska
            return '.webm'
        else:
            # 默认尝试 .webm（浏览器录音常用）
            return '.webm'
    
    # 检测格式
    file_ext = detect_format(content)
    filename = audio.filename or f"audio{file_ext}"
    print(f"[INFO] 检测到音频格式: {file_ext}, 文件名: {filename}")
    
    # 保存上传的音频到临时文件
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    try:
        input_file.write(content)
        input_file.close()
        
        output_path, ai_text, error = process_voice(input_file.name)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        # 返回音频文件路径和文本
        return {
            'success': True,
            'audio_url': f'/api/audio/{os.path.basename(output_path)}',
            'text': ai_text,
            'filename': os.path.basename(output_path)
        }
    finally:
        # 清理输入文件
        if os.path.exists(input_file.name):
            os.unlink(input_file.name)

@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """获取生成的音频文件"""
    # 从临时目录查找文件
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/wav')
    raise HTTPException(status_code=404, detail="文件不存在")

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 HTML 前端"""
    html_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>HTML 文件未找到</h1>"

if __name__ == '__main__':
    # 启动时自动加载模型
    print("[INFO] 启动 FastAPI 服务器...")
    load_model()
    
    uvicorn.run(app, host="0.0.0.0", port=5001)

