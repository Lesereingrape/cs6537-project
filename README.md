# CS6537 Member-B Workspace

RAG 模拟购物代理 实现与运行

## 保留内容

- `src/`：RAG 系统代码
- `data/products/`：知识库输入样例
- `data/vector_store/`：索引输出目录（运行后生成）
- `docs/architecture/system_architecture.md`：系统架构图
- `docs/deployment/deployment_guide.md`：部署说明
- `docs/interfaces/member_collaboration_interfaces.md`：与其他成员交互接口说明

## 快速运行

1. 安装依赖：`pip install -r requirements.txt`
2. 构建索引：`python src/main.py --build-index`
3. 查询测试：`python src/main.py --query "我想买通勤降噪耳机" --top-k 4`
