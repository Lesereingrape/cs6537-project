# 部署说明

## 1. 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 可用的 OpenAI-Compatible LLM 接口（可对接 ChatGLM API 网关）

## 2. 安装步骤

```bash
cd d:\cs6537project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=你的接口地址
OPENAI_MODEL=THUDM/chatglm3-6b
```

> 说明：`ChatOpenAI` 使用 OpenAI 兼容协议，可连接支持该协议的开源模型服务。

## 4. 构建向量索引

```bash
python src/main.py --build-index
```

成功后将在 `data/vector_store/` 生成索引。

## 5. 运行查询

```bash
python src/main.py --query "我想买适合地铁通勤、预算400内的耳机" --top-k 4
```

输出包括：
- 检索片段预览
- AI 导购建议（含推荐理由与清单）

## 6. 可扩展项（后续阶段）

- 具体协作接口请查看：`docs/interfaces/member_collaboration_interfaces.md`

## 7. 成员协作接口提示

- 成员A接口：将标准化内容样本放入 `data/products/`（文件格式 `.md`）
- 成员C接口：读取本系统输出文本，按其评估指标表进行打分与记录
- 成员D接口：将提示词模板中的变量注入为查询字符串后调用 `src/main.py --query`

说明：当前仓库已删除其他成员的具体交付内容，仅保留可对接接口与提示。
