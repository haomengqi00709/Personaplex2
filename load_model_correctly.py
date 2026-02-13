#!/usr/bin/env python3
"""
正确加载 PersonaPlex 模型的脚本
尝试多种方法找到正确的加载方式
"""

import os
import torch
from transformers import AutoConfig, AutoModel
from huggingface_hub import login
import json

MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")

def check_model_config():
    """检查模型配置"""
    print("="*60)
    print("检查模型配置")
    print("="*60)
    
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    try:
        # 加载配置
        config = AutoConfig.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        
        print(f"\n模型类型: {config.model_type}")
        print(f"架构类: {config.architectures if hasattr(config, 'architectures') else 'N/A'}")
        print(f"自定义代码: {config.auto_map if hasattr(config, 'auto_map') else 'N/A'}")
        
        # 查看配置文件的详细信息
        print("\n配置详情:")
        print(f"- Model type: {getattr(config, 'model_type', 'N/A')}")
        print(f"- Architectures: {getattr(config, 'architectures', 'N/A')}")
        
        # 检查是否有自定义类
        if hasattr(config, 'auto_map'):
            print(f"\n自定义映射: {config.auto_map}")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def try_load_model():
    """尝试加载模型"""
    print("\n" + "="*60)
    print("尝试加载模型")
    print("="*60)
    
    config = check_model_config()
    if not config:
        return None
    
    try:
        # 方法1: 使用 AutoModel 并信任远程代码
        print("\n方法1: 使用 AutoModel + trust_remote_code")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功")
        print(f"模型类型: {type(model).__name__}")
        
        # 检查模型的方法
        print("\n可用方法:")
        methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
        important_methods = [m for m in methods if m in ['generate', 'forward', 'encode', 'decode']]
        print(f"关键方法: {important_methods}")
        
        return model
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_model_structure(model):
    """检查模型结构"""
    if model is None:
        return
    
    print("\n" + "="*60)
    print("模型结构检查")
    print("="*60)
    
    print("\n模型属性:")
    attrs = [a for a in dir(model) if not a.startswith('_')]
    print(f"主要属性: {[a for a in attrs if not callable(getattr(model, a, None))][:20]}")
    
    # 检查输入格式
    if hasattr(model, 'config'):
        print(f"\n配置信息:")
        print(f"- Model type: {getattr(model.config, 'model_type', 'N/A')}")

if __name__ == "__main__":
    model = try_load_model()
    if model:
        inspect_model_structure(model)
        print("\n✅ 模型检查完成")
    else:
        print("\n❌ 模型加载失败")

