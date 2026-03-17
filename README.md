# 静态 workflow RAG CLI（含 Conversational / Rerank / Trace / Minimal Agent）

这是一个本地可运行的 **静态 workflow RAG CLI** 项目，并已补齐下一阶段面试导向能力：

- Conversational RAG（会话状态、Query Rewrite、会话压缩、真正多轮）
- True Reranker（Recall + Rerank 两阶段）
- Observability / Trace（可追踪、可诊断）
- Minimal Document QA Agent（状态、工具、路由、循环、停止条件）

## 技术栈
- Python 3.11
- Typer + Rich
- ChromaDB (PersistentClient)
- rank-bm25
- Ollama 本地模型（默认 `qwen3:8b` / `nomic-embed-text`）
- python-docx

## 安装
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
```

## 启动 Ollama（示意）
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama serve
```

## 数据目录
- 向量库：`data/chroma/`
- chunk 清单：`data/chunks.jsonl`
- trace：`data/traces.jsonl`
- 评测集：`data/eval_set.jsonl`

## CLI 命令
### 1) ingest：导入文档
```bash
python -m app.cli ingest ./data/docs/novel.md
```

### 2) ask：单轮问答（可选 rewrite / rerank / trace）
```bash
python -m app.cli ask "主角为什么离开故乡？" --language zh-CN --rewrite --show-rewrite --reranker --trace --debug
```

### 3) chat：真正多轮 Conversational RAG
```bash
python -m app.cli chat --language zh-CN --reranker --trace --show-rewrite
```

### 4) agent-chat：Minimal Document QA Agent
```bash
python -m app.cli agent-chat --language zh-CN --show-steps
```

### 5) trace-show：查看最近 trace
```bash
python -m app.cli trace-show --limit 10
```

### 6) evaluate：最小离线评测（A类必补）
```bash
python -m app.cli evaluate --language zh-CN --reranker
```

## 当前版本边界
- 仍以本地 CLI 为主，不提供 Web/FastAPI
- Agent 为最小可用版本，重点是动态控制流演示
