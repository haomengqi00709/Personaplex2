# GitHub 仓库设置指南

## 📋 步骤 1: 在 GitHub 上创建新仓库

1. 访问 [GitHub](https://github.com)
2. 点击右上角的 **"+"** 按钮，选择 **"New repository"**
3. 填写仓库信息：
   - **Repository name**: `personaplex-test` (或你喜欢的名字)
   - **Description**: "PersonaPlex-7b-v1 模型测试项目，包含 Web 前端和测试脚本"
   - **Visibility**: 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为我们已经有了）
4. 点击 **"Create repository"**

## 📤 步骤 2: 推送代码到 GitHub

### 方法 A: 使用便捷脚本（推荐）

```bash
# 1. 添加远程仓库（替换 YOUR_USERNAME 和 REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 2. 运行推送脚本
./push_to_github.sh
```

### 方法 B: 手动推送

```bash
# 1. 添加远程仓库（替换 YOUR_USERNAME 和 REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 2. 推送代码（分支已重命名为 main）
git push -u origin main
```

### 如果使用 SSH（推荐）

如果你配置了 SSH 密钥，可以使用：

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

## 🔐 步骤 3: 认证

如果推送时要求认证：

### 方法 A: 使用 Personal Access Token (PAT)

1. 访问 https://github.com/settings/tokens
2. 点击 **"Generate new token"** → **"Generate new token (classic)"**
3. 设置权限：
   - ✅ `repo` (完整仓库访问权限)
4. 生成并复制 token
5. 推送时使用 token 作为密码：
   ```bash
   Username: your_github_username
   Password: your_personal_access_token
   ```

### 方法 B: 使用 GitHub CLI

```bash
# 安装 GitHub CLI（如果还没安装）
# macOS: brew install gh
# Linux: 查看 https://cli.github.com/

# 登录
gh auth login

# 推送（会自动处理认证）
git push -u origin main
```

## ✅ 验证

推送成功后，访问你的 GitHub 仓库页面，应该能看到所有文件。

## 📝 后续更新

以后更新代码时：

```bash
# 1. 查看更改
git status

# 2. 添加更改的文件
git add .

# 3. 提交更改
git commit -m "描述你的更改"

# 4. 推送到 GitHub
git push
```

## 🔗 有用的 Git 命令

```bash
# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 查看当前分支
git branch

# 创建新分支
git checkout -b feature-branch

# 切换分支
git checkout main
```

## ⚠️ 注意事项

1. **不要提交敏感信息**:
   - `.env` 文件（已在 .gitignore 中）
   - `HF_TOKEN` 等密钥
   - 大型模型文件

2. **.gitignore 已配置**:
   - Python 缓存文件
   - 模型权重文件
   - 音频文件
   - 环境变量文件

3. **如果推送失败**:
   - 检查网络连接
   - 确认仓库 URL 正确
   - 确认有推送权限

## 🎯 快速命令参考

```bash
# 完整流程（首次推送）
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main

# 后续更新
git add .
git commit -m "Update description"
git push
```

