# Local MLX Chat（中文说明）

[English README](README.md)

Local MLX Chat 是一个面向 Apple Silicon 的轻量本地对话应用，核心目标是：

- **只用本地 MLX 模型推理**
- 用最简单的部署方式快速启动
- 具备实用的对话能力（多轮、流式、历史管理、参数调节）

默认模型：

- `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

## 项目是什么

这是一个本地聊天前后端：

- 后端：FastAPI
- 推理：`mlx-lm`（`load`、`stream_generate`、采样器与重复惩罚）
- 前端：单页 HTML/CSS/JS（无重型前端框架）

重点说明：

- 推理在本机执行（Apple Silicon）
- 首次使用会从 Hugging Face 下载模型
- 下载后会走本地缓存，不会每次重复下载
- 本项目不是云端 API 转发器

## 功能列表

- 多轮会话：每轮都会携带历史上下文
- `Send` / `Stop` / `New Chat`
- 会话线程：新建、切换、删除
- 用户消息支持“编辑重试”
- 参数预设：保存、选择、覆盖、删除
- 自定义模型管理：在 UI 添加/删除模型 ID
- 流式输出（SSE）与非流式输出
- 流式生成时实时显示 token 速度（`tok/s`）
- 思考区与最终答案分区显示（模型支持时）
- 可强制输出语言（`Auto` / `中文` / `English` / `日本語` / `한국어`）
- 主要参数可持久化保存

## 环境要求

- macOS（Apple Silicon）
- Python 3.10+

## 快速启动

```bash
cd <your-project-directory>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 模型配置

默认模型：

- `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

如果你想在下拉框加入其他模型，可用环境变量：

```bash
EXTRA_MODELS="mlx-community/Qwen2.5-7B-Instruct-4bit,mlx-community/Llama-3.2-3B-Instruct-4bit" \
uvicorn app:app --host 127.0.0.1 --port 8000
```

你也可以直接在界面里添加模型：

- 在 `模型管理（自定义模型 ID）` 输入模型 ID
- 点击 `添加模型`
- 模型会保存到浏览器 `localStorage`，下次打开仍可选择
- `删除当前模型` 只会删除你手动添加的模型，不会删默认模型

## 参数说明

- `Temperature`：越高越发散，越低越稳定
- `Top P`：核采样概率质量阈值
- `Top K`：仅在概率最高的 K 个 token 中采样
- `Repetition Penalty`：重复惩罚，抑制长段复读
- `Repeat Context`：重复惩罚参考的 token 窗口
- `Max Tokens`：单次回复最大新生成 token 数（当前后端上限 `4096`）
- `Streaming`：是否流式返回增量 token
- `Thinking`：是否启用思考模式（依赖模型/模板支持）
- `System Prompt`：全局角色和约束
- `Response Language`：强制输出语言

注意：`Max Tokens` 是单次回复长度上限，不是模型总上下文上限。

## 会话与存储

- 会话线程和消息存储在 `sessionStorage`
- 刷新页面后可恢复当前浏览器会话中的聊天记录
- 关闭浏览器后会话记录会清空
- 参数设置存储在 `localStorage`，重开浏览器仍可保留

## 目录结构

- `app.py`：API、模型加载、Prompt 构建、生成流程
- `static/index.html`：界面、聊天渲染、历史管理、参数保存
- `requirements.txt`：Python 依赖
- `.model_overrides/`：模型配置兼容覆盖目录（运行时自动生成）

## 常见问题

### 为什么第一次回复很慢？

首次请求通常会下载并初始化模型，后续会快很多。

### 如何删除本地模型缓存？

模型缓存由 `huggingface_hub` 管理。先查看缓存根目录：

```bash
python -c "from huggingface_hub import scan_cache_dir; print(scan_cache_dir().cache_dir)"
```

再删除目标模型对应缓存目录即可。

### 为什么这里 `Max Tokens` 只有 4096？

这是项目当前为了稳定性和响应时延做的上限控制。
