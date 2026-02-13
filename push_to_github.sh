#!/bin/bash
# 推送到 GitHub 的便捷脚本

echo "🚀 准备推送到 GitHub..."
echo ""

# 检查是否已设置远程仓库
if git remote get-url origin &> /dev/null; then
    echo "✅ 远程仓库已配置:"
    git remote -v
    echo ""
    echo "正在推送..."
    git push -u origin main
else
    echo "❌ 远程仓库未配置"
    echo ""
    echo "请先执行以下命令添加远程仓库:"
    echo ""
    echo "  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo ""
    echo "或者使用 SSH:"
    echo ""
    echo "  git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git"
    echo ""
    echo "然后再次运行此脚本:"
    echo "  ./push_to_github.sh"
    echo ""
    echo "详细说明请查看 GITHUB_SETUP.md"
fi

