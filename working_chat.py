#!/usr/bin/env python3
"""
使用官方 PersonaPlex 代码库的简单语音对话界面
如果官方代码库可用，会使用它；否则显示说明
"""

import os
import sys
import gradio as gr

# 尝试导入官方代码库
OFFICIAL_AVAILABLE = False
try:
    sys.path.insert(0, '/workspace/personaplex')
    # 尝试查找官方模块
    import importlib.util
    
    # 查找可能的入口文件
    possible_paths = [
        '/workspace/personaplex/personaplex/__init__.py',
        '/workspace/personaplex/src/personaplex/__init__.py',
        '/workspace/personaplex/personaplex.py',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("personaplex", path)
            if spec:
                OFFICIAL_AVAILABLE = True
                break
    
    # 或者尝试直接导入
    try:
        import personaplex
        OFFICIAL_AVAILABLE = True
    except:
        pass
        
except Exception as e:
    print(f"官方代码库检查: {e}")

def get_setup_instructions():
    """获取设置说明"""
    return """
## ⚠️ 需要设置官方代码库

当前 transformers 版本不支持 PersonaPlex 的 processor。

### 快速设置步骤：

```bash
# 1. 运行设置脚本
cd /workspace/Personaplex2
chmod +x setup_and_run.sh
./setup_and_run.sh

# 2. 查看官方文档
cat /workspace/personaplex/README.md

# 3. 按照官方文档运行示例
```

### 或者手动设置：

```bash
cd /workspace
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex
pip install -r requirements.txt
cat README.md  # 查看使用方法
```

设置完成后，请按照官方文档运行测试。
"""

def simple_chat(audio, text_prompt):
    """简单的聊天函数（占位）"""
    if not OFFICIAL_AVAILABLE:
        return None, get_setup_instructions()
    
    # 如果官方代码库可用，这里应该调用官方 API
    return None, "✅ 官方代码库已检测到，请按照官方文档使用"

# 创建界面
with gr.Blocks(title="PersonaPlex 语音对话", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ PersonaPlex 实时语音对话
    
    简单测试界面
    """)
    
    if not OFFICIAL_AVAILABLE:
        gr.Markdown(get_setup_instructions())
        
        setup_btn = gr.Button("📥 运行设置脚本", variant="primary")
        setup_output = gr.Markdown()
        
        def run_setup():
            import subprocess
            result = subprocess.run(
                ['bash', '/workspace/Personaplex2/setup_and_run.sh'],
                capture_output=True,
                text=True,
                cwd='/workspace/Personaplex2'
            )
            return f"```\n{result.stdout}\n{result.stderr}\n```"
        
        setup_btn.click(fn=run_setup, outputs=setup_output)
    else:
        gr.Markdown("✅ 官方代码库已检测到！")
        
        with gr.Row():
            audio_input = gr.Audio(
                label="🎤 说话",
                type="filepath",
                sources=["microphone"],
                format="wav"
            )
        
        text_prompt = gr.Textbox(
            label="角色设定",
            value="You are a helpful AI assistant.",
            lines=2
        )
        
        chat_btn = gr.Button("🚀 发送", variant="primary")
        
        audio_output = gr.Audio(label="🔊 AI 回复", type="filepath")
        text_output = gr.Markdown(label="📝 回复")
        
        chat_btn.click(fn=simple_chat, inputs=[audio_input, text_prompt], outputs=[audio_output, text_output])

if __name__ == "__main__":
    print("="*60)
    print("PersonaPlex 语音对话界面")
    print("端口: 5001")
    print("="*60)
    
    if not OFFICIAL_AVAILABLE:
        print("⚠️  官方代码库未找到")
        print("   界面将显示设置说明")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=5001,
        share=False
    )

