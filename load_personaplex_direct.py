#!/usr/bin/env python3
"""
直接加载 PersonaPlex 模型（绕过配置检查）
"""

import os
import torch
import json
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel
from huggingface_hub import login

MODEL_ID = "nvidia/personaplex-7b-v1"
HF_TOKEN = os.getenv("HF_TOKEN")

def load_personaplex_direct():
    """直接加载 PersonaPlex，绕过配置检查"""
    
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    print("="*60)
    print("直接加载 PersonaPlex 模型")
    print("="*60)
    
    # 方法1: 尝试使用 snapshot_download 下载整个仓库
    print("\n方法1: 下载整个模型仓库...")
    try:
        local_dir = snapshot_download(
            repo_id=MODEL_ID,
            token=HF_TOKEN,
            local_dir="/tmp/personaplex_model"
        )
        print(f"✅ 模型下载到: {local_dir}")
        
        # 检查是否有自定义代码
        import os
        files = os.listdir(local_dir)
        print(f"文件列表: {files}")
        
        # 查找 Python 文件
        py_files = [f for f in files if f.endswith('.py')]
        if py_files:
            print(f"找到 Python 文件: {py_files}")
        
    except Exception as e:
        print(f"❌ 方法1失败: {e}")
    
    # 方法2: 直接使用 AutoModel，但先修改 CONFIG_MAPPING
    print("\n方法2: 修改 Transformers 配置映射...")
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        
        # 临时添加 personaplex 映射（指向 moshi）
        if 'personaplex' not in CONFIG_MAPPING:
            print("添加 personaplex -> moshi 映射...")
            # 尝试使用 moshi 的配置
            try:
                from transformers import MoshiConfig
                CONFIG_MAPPING['personaplex'] = MoshiConfig
                print("✅ 添加映射成功")
            except:
                print("⚠️  无法添加映射（MoshiConfig 不可用）")
        
        # 现在尝试加载
        print("尝试加载模型...")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✅ 方法2成功")
        return model
        
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法3: 手动构造配置
    print("\n方法3: 手动构造配置...")
    try:
        # 下载 config.json
        config_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="config.json",
            token=HF_TOKEN
        )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        print(f"配置内容: {json.dumps(config_dict, indent=2)[:500]}")
        
        # 检查 auto_map
        auto_map = config_dict.get('auto_map', {})
        if auto_map:
            print(f"发现 auto_map: {auto_map}")
            # 如果有自定义代码，尝试手动加载
            # 这需要更复杂的处理
        
        # 尝试直接加载权重
        print("尝试直接加载权重...")
        # 这需要知道模型的确切结构
        
    except Exception as e:
        print(f"❌ 方法3失败: {e}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    model = load_personaplex_direct()
    if model:
        print("\n✅ 模型加载成功！")
    else:
        print("\n❌ 所有方法都失败了")

