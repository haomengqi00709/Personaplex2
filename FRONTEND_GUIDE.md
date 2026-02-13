# PersonaPlex Web 前端使用指南

## 🚀 快速启动

### 方法 1: 使用启动脚本（推荐）

```bash
chmod +x start_web.sh
./start_web.sh
```

### 方法 2: 直接运行

```bash
python app.py
```

## 📋 使用步骤

### 1. 启动 Web 界面

运行 `app.py` 后，你会看到类似输出：

```
Running on local URL:  http://127.0.0.0:7860
```

### 2. 访问界面

- **本地访问**: http://localhost:7860
- **RunPod 访问**: 使用 RunPod 提供的公共 URL（通常在 Pod 详情页）

### 3. 使用界面

#### 步骤 1: 加载模型
- 点击 **"🔄 加载模型"** 按钮
- 等待模型加载完成（首次加载需要 2-5 分钟）
- 查看模型状态确认加载成功

#### 步骤 2: 准备输入
- **音频输入**（可选）:
  - 点击上传按钮选择音频文件
  - 或使用麦克风录制
  - 推荐格式: 24kHz WAV 文件
  
- **文本提示**（必需）:
  - 输入英文描述 AI 的角色和风格
  - 例如: "You are a friendly customer service agent. Be helpful and professional."
  
- **参考语音**（可选）:
  - 上传参考音频文件来控制输出音色

#### 步骤 3: 生成响应
- 点击 **"🚀 生成响应"** 按钮
- 等待处理完成（通常 1-3 秒）
- 查看文本输出和播放音频输出

## 🎨 界面功能

### 主要区域

1. **模型控制区**
   - 加载模型按钮
   - 模型状态显示
   - 系统信息（GPU、显存等）

2. **输入设置区**
   - 音频上传/录制
   - 文本提示输入
   - 参考语音上传

3. **输出结果区**
   - AI 语音输出（可播放）
   - AI 文本输出
   - 状态信息

## 💡 使用技巧

### 文本提示示例

**友好助手:**
```
You are a helpful AI assistant. Be friendly, concise, and conversational.
```

**客服代表:**
```
You are a professional customer service agent. Be polite, patient, and solution-oriented.
```

**技术支持:**
```
You are a technical support specialist. Explain things clearly and provide step-by-step guidance.
```

**创意角色:**
```
You are a creative storyteller. Use vivid language and engaging narratives.
```

### 音频处理建议

1. **音频格式**: 
   - 推荐: 24kHz, 16-bit, 单声道 WAV
   - 其他格式会自动转换，但可能影响质量

2. **音频长度**:
   - 建议: 1-10 秒
   - 过长会增加处理时间

3. **音频质量**:
   - 清晰、无背景噪音
   - 正常音量

## 🔧 在 RunPod 上部署

### 步骤 1: 启动 Pod

1. 创建 RunPod Pod（GPU 至少 16GB VRAM）
2. 设置环境变量 `HF_TOKEN`
3. 上传项目文件

### 步骤 2: 安装依赖

```bash
cd /workspace/Personaplex2
pip install -r requirements.txt
```

### 步骤 3: 启动 Web 服务

```bash
python app.py
```

### 步骤 4: 访问界面

1. 在 RunPod Pod 详情页找到 **"Connect"** 按钮
2. 选择 **"HTTP Service"** 或 **"Public URL"**
3. 点击生成的链接访问界面

### 步骤 5: 配置端口（如果需要）

如果 RunPod 需要特定端口，修改 `app.py`:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=8080,  # 改为 RunPod 指定的端口
    share=False
)
```

## 🐛 故障排除

### 问题 1: 无法访问界面

**解决方案:**
- 确认服务器已启动
- 检查防火墙设置
- 在 RunPod 上确认公共 URL 已启用

### 问题 2: 模型加载失败

**解决方案:**
- 检查 `HF_TOKEN` 环境变量
- 确认已接受模型许可协议
- 检查网络连接
- 查看控制台错误信息

### 问题 3: 音频上传失败

**解决方案:**
- 确认音频格式支持（WAV, MP3, FLAC）
- 检查文件大小（建议 < 10MB）
- 尝试转换音频格式

### 问题 4: 推理速度慢

**解决方案:**
- 使用更小的音频文件
- 减少 `max_new_tokens` 参数
- 确保使用 GPU（检查系统信息）

### 问题 5: 显存不足

**解决方案:**
- 使用更大的 GPU
- 确保使用 float16（已在代码中设置）
- 关闭其他占用显存的程序

## 📊 性能参考

在标准配置下的预期性能：

| 操作 | 时间 |
|------|------|
| 模型加载 | 2-5 分钟 |
| 首次推理 | 3-5 秒 |
| 后续推理 | 1-3 秒 |
| 显存占用 | 12-15 GB |

## 🔒 安全提示

1. **不要在生产环境使用**: 这是测试界面，不适合生产部署
2. **保护 Token**: 不要在代码中硬编码 `HF_TOKEN`
3. **限制访问**: 在 RunPod 上考虑使用密码保护
4. **监控资源**: 注意 GPU 使用情况，避免超时

## 🎯 下一步

测试成功后，你可以：

1. **自定义界面**: 修改 `app.py` 添加更多功能
2. **API 集成**: 将后端改为 REST API
3. **批量处理**: 添加批量音频处理功能
4. **流式处理**: 实现实时音频流处理

## 📝 示例工作流

```
1. 启动 Web 界面
   → python app.py

2. 加载模型
   → 点击"加载模型"按钮
   → 等待 2-5 分钟

3. 准备测试
   → 上传测试音频（或使用麦克风）
   → 输入文本提示

4. 生成响应
   → 点击"生成响应"
   → 查看文本和音频输出

5. 调整参数
   → 修改文本提示
   → 尝试不同的音频输入
   → 测试不同角色
```

## 🔗 相关链接

- [Gradio 文档](https://www.gradio.app/docs/)
- [PersonaPlex 模型](https://huggingface.co/nvidia/personaplex-7b-v1)
- [RunPod 文档](https://docs.runpod.io/)

