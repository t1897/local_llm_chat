# 本地 MLX AI 对话应用（最小版）

一个轻量本地 Web App，用于本地化 AI 对话：

- 左侧：输入框
- 右侧：模型回复
- 多轮会话：每轮问答会自动带上之前的上下文
- 参数：`Streaming`、`System Prompt`、`Temperature`、`Top P`、`Top K`、`Repetition Penalty`、`Repeat Context`、`Max Tokens`
- 可选强制输出语言：`Auto / 中文 / English / 日本語 / 한국어`
- 模型下拉选择：默认接入  
  `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

## 1) 环境要求

- macOS (Apple Silicon)
- Python 3.10+

## 2) 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) 启动

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 4) 首次下载模型

首次请求会自动从 Hugging Face 下载模型，下载完后会缓存到本地，再次启动会更快。

## 5) 新增模型（预留扩展）

通过环境变量 `EXTRA_MODELS` 追加模型到下拉框：

```bash
EXTRA_MODELS="mlx-community/Qwen2.5-7B-Instruct-4bit,mlx-community/Llama-3.2-3B-Instruct-4bit" uvicorn app:app --host 127.0.0.1 --port 8000
```

## 6) 文件说明

- `app.py`: FastAPI 后端 + MLX 模型加载/推理 + streaming SSE
- `static/index.html`: 单页前端（左右布局 + 参数面板）
- `requirements.txt`: Python 依赖

## 7) 会话行为

- 点击 `Send` 会把当前会话历史一起发送给模型（多轮上下文）。
- 点击 `Stop` 可中断当前生成（用于终止思考并开始新问题）。
- 点击 `New Chat` 会开启一个新的会话线程，旧会话会保存在左侧“会话历史”里，可随时切换继续。
- 左侧“会话历史”支持删除单条会话（删除当前会话会自动切换到其他会话）。
- 每条用户消息支持“编辑重试”：可修改该轮提问并从该轮开始重新生成，后续旧结果会被替换。
- 会话线程保存在浏览器 `sessionStorage`：刷新页面可恢复；关闭浏览器后自动清空。
