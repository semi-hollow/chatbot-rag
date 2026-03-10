# 静态 workflow RAG CLI

这是一个“最小可用”的 **静态 workflow RAG CLI** 项目，用于本地导入文档并在终端中进行问答验证。

## 特性
- 文档导入：`.txt` / `.md` / `.docx`
- 文档切块
- BM25 关键词检索（`rank-bm25`）
- 向量检索（Chroma `PersistentClient`）
- Hybrid 融合（RRF）
- 可选 reranker（当前占位实现）
- Ollama 本地模型问答（默认 `qwen3:8b`）
- 中英文 prompt 切换（`zh-CN` / `en-US`）
- CLI 多轮交互 + 引用输出

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

## CLI 用法
### 1) 导入文档
```bash
python -m app.cli ingest ./data/docs/novel.md
```

### 2) 单轮问答
```bash
python -m app.cli ask "主角为什么离开故乡？" --language zh-CN
python -m app.cli ask "Why did the hero leave home?" --language en-US
```

### 3) 多轮聊天
```bash
python -m app.cli chat --language zh-CN
```

## 数据目录
- 向量库：`data/chroma/`
- chunk 清单：`data/chunks.jsonl`

> 本阶段仅实现“静态 workflow RAG CLI”，不包含 Agent、Web、FastAPI、长期记忆与工程化评测。
