# Local MLX Chat（本地 AI 对话应用）

这是一个只依赖本地 MLX 模型推理的 Web 对话应用，目标是：

- 用最轻量的方式在本机跑大语言模型对话
- 不依赖 OpenAI/Claude 等云端 API
- 支持多轮会话、流式输出、思考区/最终答案分离
- 预留模型扩展能力，后续可添加更多 MLX 模型

默认模型：

- `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

## 项目定位

这个项目是一个本地部署的聊天前后端：

- 后端：FastAPI + `mlx-lm`
- 前端：单页 HTML/CSS/JS（无重前端框架）
- 模型调用：直接走 `mlx_lm.load()` 与 `mlx_lm.stream_generate()`

关键点：

- 生成在本机进行（CPU/GPU 为本机 Apple Silicon）
- 首次需要联网从 Hugging Face 拉取模型权重
- 模型下载后会使用本地缓存，后续无需重复下载

## 功能总览

- 多轮会话上下文：会把历史问答一起传给模型
- 可切换流式输出（Streaming）/ 非流式输出
- 可中断生成（Stop）
- 思考过程与最终答案分区显示（可折叠思考）
- 编辑上一轮用户问题并从该轮重生成
- 会话线程管理：新建、切换、删除
- 设置持久化：系统提示词和主要采样参数保存在浏览器本地
- 输出语言控制：`Auto / 中文 / English / 日本語 / 한국어`

## 架构与数据流

1. 前端将当前输入 + 会话历史 + 参数发送到 `/api/chat` 或 `/api/chat/stream`
2. 后端按聊天模板拼接 prompt（含 system prompt、历史消息、当前消息）
3. 后端调用 `mlx-lm` 在本地生成
4. 前端实时渲染 token（流式模式）或一次性渲染（非流式模式）

## 环境要求

- macOS（Apple Silicon）
- Python 3.10+

## 安装与启动

```bash
cd <your-project-directory>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

浏览器访问：

```text
http://127.0.0.1:8000
```

## 模型说明（本地 MLX）

默认通过 `mlx-lm` 加载 Hugging Face 上的 MLX 权重模型：

- `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

本项目不是把请求转发到云 API，而是本地加载模型并生成。  
首次下载慢是正常现象，之后主要是本地推理速度。

### 模型兼容补丁说明

当模型配置中的 `model_type` 与当前 `mlx-lm` 版本存在命名差异时，后端会自动生成一个本地兼容覆盖目录（`.model_overrides`）来修正配置映射，然后继续加载。

## 新增模型（扩展入口）

可通过环境变量把更多模型注入下拉框：

```bash
EXTRA_MODELS="mlx-community/Qwen2.5-7B-Instruct-4bit,mlx-community/Llama-3.2-3B-Instruct-4bit" \
uvicorn app:app --host 127.0.0.1 --port 8000
```

## 参数说明（生成质量核心）

- `Temperature`：随机性，越高越发散，越低越稳定
- `Top P`：核采样阈值，控制候选 token 累积概率
- `Top K`：每步只在概率最高的 K 个 token 中采样
- `Repetition Penalty`：重复惩罚，值越高越抑制重复
- `Repeat Context`：重复惩罚回看窗口长度
- `Max Tokens`：单次最多新生成 token 数（当前后端上限 4096）
- `Streaming`：是否流式返回 token
- `Thinking`：是否启用模型思考模式（依赖模型/模板支持）
- `System Prompt`：全局角色和规则
- `Response Language`：强制输出语言（Auto 则不强制）

## 会话与存储行为

- `Send`：发送并生成
- `Stop`：中断当前生成
- `New Chat`：创建新会话线程
- 左侧历史：可切换、删除会话
- 用户消息支持“编辑重试”

存储策略：

- 会话记录存储在 `sessionStorage`
- 刷新页面后可恢复当前浏览器会话中的记录
- 关闭浏览器后会话记录自动清空
- 参数和系统提示词存储在 `localStorage`（会跨浏览器重开保留）

## 目录结构

- `app.py`：后端 API、模型加载、采样与流式生成
- `static/index.html`：前端 UI、会话管理、参数持久化与渲染逻辑
- `requirements.txt`：依赖列表
- `.model_overrides/`：模型兼容覆盖目录（运行时生成）

## 常见问题

### 1) 为什么第一次发送很慢？

首次请求会触发模型下载与加载，后续会明显变快。

### 2) 如何删除本地模型缓存？

模型缓存由 `huggingface_hub` 管理。可删除对应缓存目录后重新下载。  
也可以先用以下命令查看缓存位置：

```bash
python -c "from huggingface_hub import scan_cache_dir; print(scan_cache_dir().cache_dir)"
```

### 3) 为什么 `Max Tokens` 不是模型宣传的超长上下文？

这里的 `Max Tokens` 是“单次新生成上限”，不是模型总上下文长度上限。  
当前项目后端为了稳定性把该值限制在 `<= 4096`。

## 开发提示

- 启动开发模式：`uvicorn app:app --reload`
- 如果修改了前端但页面未更新，尝试强制刷新浏览器缓存
- 如需接入更多模型，优先确保该模型与当前 `mlx-lm` 版本兼容
