#!/usr/bin/env python3
"""
PersonaPlex 快速测试脚本
适用于 RunPod 最低配置快速验证
"""

import os
import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, MoshiForConditionalGeneration
from huggingface_hub import login

# 配置
MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")

def check_environment():
    """检查运行环境"""
    print("="*60)
    print("环境检查")
    print("="*60)
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = "cuda"
    else:
        print("❌ CUDA 不可用，将使用 CPU（不推荐）")
        device = "cpu"
    
    # 检查 Hugging Face Token
    if HF_TOKEN:
        print(f"✅ HF_TOKEN 已设置")
        try:
            login(token=HF_TOKEN)
            print("✅ Hugging Face 认证成功")
        except Exception as e:
            print(f"⚠️  Hugging Face 认证失败: {e}")
    else:
        print("⚠️  HF_TOKEN 未设置")
        print("   请在 RunPod 环境变量中设置 HF_TOKEN")
    
    print()
    return device

def load_model(device):
    """加载模型"""
    print("="*60)
    print("加载模型")
    print("="*60)
    
    print(f"模型 ID: {MODEL_ID}")
    print("使用 float16 以降低显存占用...")
    
    try:
        # 尝试加载处理器
        print("\n1. 加载处理器...")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        print("✅ 处理器加载成功")
        
        # 尝试加载模型
        print("\n2. 加载模型...")
        try:
            # 首先尝试 MoshiForConditionalGeneration
            model = MoshiForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✅ 使用 MoshiForConditionalGeneration 加载成功")
        except Exception as e1:
            print(f"⚠️  MoshiForConditionalGeneration 失败: {e1}")
            print("   尝试使用 AutoModelForSpeechSeq2Seq...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✅ 使用 AutoModelForSpeechSeq2Seq 加载成功")
        
        model.eval()
        
        # 显示显存使用
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n显存使用: {memory_used:.2f} GB / {memory_reserved:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 确认已设置 HF_TOKEN 环境变量")
        print("2. 确认已接受模型许可协议:")
        print("   https://huggingface.co/nvidia/personaplex-7b-v1")
        print("3. 检查网络连接")
        raise

def test_inference(model, processor, device):
    """测试推理"""
    print("\n" + "="*60)
    print("推理测试")
    print("="*60)
    
    # 创建测试音频（1秒静音，24kHz）
    sample_rate = 24000
    duration = 1.0
    test_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # 文本提示
    text_prompt = "You are a helpful AI assistant. Say hello in a friendly way."
    
    print(f"文本提示: {text_prompt}")
    print(f"测试音频: {len(test_audio)} 采样点 ({duration}秒, {sample_rate}Hz)")
    
    try:
        # 处理输入
        print("\n处理输入...")
        inputs = processor(
            audio=test_audio,
            sampling_rate=sample_rate,
            text=text_prompt,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 推理
        print("执行推理...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True
            )
        
        # 解码输出
        if hasattr(processor, 'decode'):
            text_output = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"\n✅ 文本输出: {text_output}")
        else:
            print(f"\n✅ 输出 tokens: {outputs.shape}")
        
        print("\n✅ 推理测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print("PersonaPlex 快速测试")
    print("="*60 + "\n")
    
    try:
        # 1. 检查环境
        device = check_environment()
        
        # 2. 加载模型
        model, processor = load_model(device)
        
        # 3. 测试推理
        success = test_inference(model, processor, device)
        
        if success:
            print("\n" + "="*60)
            print("✅ 所有测试通过！")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 测试失败")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

