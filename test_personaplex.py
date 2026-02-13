#!/usr/bin/env python3
"""
PersonaPlex 模型测试脚本
适用于 RunPod GPU 环境的最低配置测试
"""

import os
import torch
import yaml
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, MoshiForConditionalGeneration, MoshiProcessor
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")


class PersonaPlexTester:
    """PersonaPlex 模型测试类"""
    
    def __init__(self, config_path="config.yaml"):
        """初始化测试器"""
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config["model"]["device"])
        self.model = None
        self.processor = None
        
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def check_huggingface_auth(self):
        """检查 Hugging Face 认证"""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️  警告: 未找到 HF_TOKEN 环境变量")
            print("请设置: export HF_TOKEN=your_token_here")
            print("或在 RunPod 的环境变量中设置")
            return False
        
        try:
            login(token=hf_token)
            print("✅ Hugging Face 认证成功")
            return True
        except Exception as e:
            print(f"❌ Hugging Face 认证失败: {e}")
            return False
    
    def load_model(self):
        """加载 PersonaPlex 模型"""
        print("\n📥 开始加载模型...")
        
        model_config = self.config["model"]
        model_id = model_config["model_id"]
        
        try:
            # 检查模型是否已下载
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_path = cache_dir / f"models--{model_id.replace('/', '--')}"
            
            print(f"模型 ID: {model_id}")
            print(f"使用数据类型: {model_config['torch_dtype']}")
            
            # 加载处理器（PersonaPlex 基于 Moshi，优先使用 MoshiProcessor）
            print("加载处理器...")
            try:
                # 首先尝试 MoshiProcessor
                self.processor = MoshiProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
                print("✅ 使用 MoshiProcessor 加载成功")
            except Exception as e1:
                print(f"⚠️  MoshiProcessor 失败: {e1}")
                print("   尝试使用 AutoProcessor...")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        model_id,
                        trust_remote_code=True
                    )
                    print("✅ 使用 AutoProcessor 加载成功")
                except Exception as e2:
                    print(f"❌ AutoProcessor 也失败: {e2}")
                    self.processor = None
            
            # 加载模型（使用 float16 降低显存）
            print("加载模型权重...")
            torch_dtype = getattr(torch, model_config["torch_dtype"])
            
            # 尝试使用 MoshiForConditionalGeneration（基于 Moshi 架构）
            try:
                self.model = MoshiForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=model_config["low_cpu_mem_usage"],
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"⚠️  使用 MoshiForConditionalGeneration 失败，尝试 AutoModel: {e}")
                # 回退到 AutoModel
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=model_config["low_cpu_mem_usage"],
                    trust_remote_code=True
                )
            
            self.model.eval()
            print("✅ 模型加载完成")
            
            # 显示显存使用情况
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"显存使用: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("\n可能的解决方案:")
            print("1. 检查是否已设置 HF_TOKEN 环境变量")
            print("2. 确认已接受模型许可协议: https://huggingface.co/nvidia/personaplex-7b-v1")
            print("3. 检查网络连接和 Hugging Face 访问")
            return False
    
    def prepare_audio(self, audio_path):
        """准备音频输入（转换为 24kHz）"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 读取音频
        audio, sr = sf.read(audio_path)
        
        # 转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 重采样到 24kHz（如果需要）
        if sr != self.config["audio"]["sample_rate"]:
            import librosa
            audio = librosa.resample(
                audio, 
                orig_sr=sr, 
                target_sr=self.config["audio"]["sample_rate"]
            )
        
        return audio, self.config["audio"]["sample_rate"]
    
    def test_inference(self, audio_path=None, text_prompt=None, voice_prompt_path=None):
        """测试推理"""
        print("\n🧪 开始推理测试...")
        
        if self.model is None or self.processor is None:
            print("❌ 模型未加载，请先调用 load_model()")
            return None
        
        # 准备输入
        if audio_path:
            print(f"处理音频输入: {audio_path}")
            audio, sr = self.prepare_audio(audio_path)
        else:
            # 生成测试音频（静音或简单波形）
            print("使用默认测试音频（1秒静音）")
            duration = 1.0
            sr = self.config["audio"]["sample_rate"]
            audio = np.zeros(int(sr * duration))
        
        # 准备文本提示
        if text_prompt is None:
            text_prompt = "You are a helpful AI assistant. Respond naturally and conversationally."
        
        print(f"文本提示: {text_prompt}")
        
        try:
            # 处理输入
            if self.processor:
                inputs = self.processor(
                    audio=audio,
                    sampling_rate=sr,
                    text=text_prompt,
                    return_tensors="pt"
                )
                # 移动到设备
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                # 如果没有 processor，手动准备输入
                print("⚠️  使用手动输入准备...")
                # 将音频转换为 tensor
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                else:
                    audio_tensor = audio.to(self.device) if hasattr(audio, 'to') else torch.tensor(audio).to(self.device)
                
                inputs = {
                    'audio': audio_tensor,
                    'text': text_prompt
                }
            
            # 推理
            print("执行推理...")
            with torch.no_grad():
                # 根据模型类型选择不同的生成方法
                if hasattr(self.model, 'generate'):
                    if 'input_ids' in inputs:
                        outputs = self.model.generate(
                            input_ids=inputs.get('input_ids'),
                            audio_codes=inputs.get('audio_codes'),
                            max_new_tokens=self.config["inference"]["max_new_tokens"],
                            temperature=self.config["inference"]["temperature"],
                            top_p=self.config["inference"]["top_p"],
                            do_sample=self.config["inference"]["do_sample"]
                        )
                    else:
                        # 尝试使用模型的 forward 方法
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            
            # 解码输出
            if self.processor and hasattr(self.processor, 'decode'):
                # 获取文本输出
                if isinstance(outputs, torch.Tensor):
                    text_output = self.processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    text_output = str(outputs)
                print(f"\n✅ 文本输出: {text_output}")
                
                # 获取音频输出（如果模型支持）
                if hasattr(outputs, 'audio_values'):
                    audio_output = outputs.audio_values.cpu().numpy()
                    return text_output, audio_output
                
                return text_output, None
            else:
                print(f"\n✅ 输出: {outputs}")
                return outputs, None
                
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_basic_functionality(self):
        """基础功能测试"""
        print("\n" + "="*50)
        print("基础功能测试")
        print("="*50)
        
        # 测试 1: 模型加载
        print("\n[测试 1] 模型加载")
        if not self.load_model():
            return False
        
        # 测试 2: 简单推理
        print("\n[测试 2] 简单推理测试")
        result = self.test_inference(
            text_prompt="Say hello in a friendly way."
        )
        
        if result:
            print("✅ 基础功能测试通过")
            return True
        else:
            print("❌ 基础功能测试失败")
            return False


def main():
    """主函数"""
    print("="*50)
    print("PersonaPlex 模型测试")
    print("="*50)
    
    # 初始化测试器
    tester = PersonaPlexTester()
    
    # 检查认证
    if not tester.check_huggingface_auth():
        print("\n⚠️  继续尝试加载模型（如果已认证）...")
    
    # 运行基础测试
    success = tester.test_basic_functionality()
    
    if success:
        print("\n" + "="*50)
        print("✅ 所有测试完成")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ 测试失败，请检查错误信息")
        print("="*50)


if __name__ == "__main__":
    main()

