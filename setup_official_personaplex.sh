#!/bin/bash
# 设置官方 PersonaPlex 代码库用于实际测试

echo "🚀 设置官方 PersonaPlex 代码库..."

cd /workspace

# 1. 克隆官方仓库
if [ ! -d "personaplex" ]; then
    echo "📥 克隆官方 PersonaPlex 仓库..."
    git clone https://github.com/NVIDIA/personaplex.git
else
    echo "✅ 官方仓库已存在，更新中..."
    cd personaplex
    git pull
    cd ..
fi

cd personaplex

# 2. 检查 requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 安装依赖..."
    pip install -r requirements.txt
else
    echo "⚠️  未找到 requirements.txt，安装基础依赖..."
    pip install torch transformers accelerate huggingface-hub soundfile librosa numpy
fi

# 3. 检查是否有示例代码
echo ""
echo "📁 检查示例代码..."
if [ -f "README.md" ]; then
    echo "✅ 找到 README.md"
    echo "   查看使用方法: cat README.md"
fi

if [ -f "examples" ] || [ -d "examples" ]; then
    echo "✅ 找到示例目录"
    ls -la examples/ 2>/dev/null || echo "   查看示例: ls examples/"
fi

# 4. 创建测试脚本
echo ""
echo "📝 创建测试脚本..."
cat > /workspace/Personaplex2/test_official.py << 'EOFTEST'
#!/usr/bin/env python3
"""
使用官方 PersonaPlex 代码库进行测试
"""
import sys
import os

# 添加官方代码库路径
sys.path.insert(0, '/workspace/personaplex')

try:
    # 尝试导入官方代码
    print("📥 尝试导入官方 PersonaPlex 代码...")
    
    # 检查是否有可用的导入
    import importlib.util
    
    # 查找主要的模块文件
    possible_files = [
        '/workspace/personaplex/personaplex/__init__.py',
        '/workspace/personaplex/src/personaplex/__init__.py',
        '/workspace/personaplex/personaplex.py',
    ]
    
    module_found = False
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"✅ 找到模块文件: {file_path}")
            spec = importlib.util.spec_from_file_location("personaplex", file_path)
            if spec:
                module_found = True
                break
    
    if not module_found:
        print("⚠️  未找到标准模块文件，检查目录结构...")
        print("\n当前目录结构:")
        import subprocess
        result = subprocess.run(['find', '/workspace/personaplex', '-maxdepth', '2', '-type', 'f', '-name', '*.py'], 
                              capture_output=True, text=True)
        print(result.stdout[:500])  # 显示前500字符
        
        print("\n请查看官方 README 了解正确的使用方法:")
        print("  cat /workspace/personaplex/README.md")
    else:
        print("✅ 可以导入官方代码")
        print("\n请参考官方示例代码进行测试")
        
except Exception as e:
    print(f"❌ 导入失败: {e}")
    print("\n请查看官方 README:")
    print("  cat /workspace/personaplex/README.md")

print("\n" + "="*60)
print("下一步:")
print("1. 查看官方 README: cat /workspace/personaplex/README.md")
print("2. 查看示例代码: ls /workspace/personaplex/examples/")
print("3. 按照官方文档运行测试")
print("="*60)
EOFTEST

chmod +x /workspace/Personaplex2/test_official.py

echo ""
echo "✅ 设置完成！"
echo ""
echo "下一步:"
echo "1. 查看官方 README: cat /workspace/personaplex/README.md"
echo "2. 运行测试脚本: python /workspace/Personaplex2/test_official.py"
echo "3. 按照官方文档使用 PersonaPlex"

