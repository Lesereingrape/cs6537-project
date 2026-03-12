import argparse

from langchain_openai import ChatOpenAI

from config import get_config
from rag_pipeline import (
    build_shopping_advice,
    build_vector_store,
    get_embeddings,
    load_product_documents,
    load_vector_store,
    split_documents,
)


def run_build_index() -> None:
    cfg = get_config()
    docs = load_product_documents(cfg.product_data_dir)
    if not docs:
        raise ValueError("未找到商品知识库文件，请检查 data/products 目录。")

    chunks = split_documents(docs)
    embeddings = get_embeddings(cfg.embedding_model)
    build_vector_store(chunks, embeddings, cfg.vector_store_dir)
    print(f"索引构建完成，共写入 {len(chunks)} 个文本块。")


def run_query(query: str, top_k: int = 4) -> None:
    cfg = get_config()
    embeddings = get_embeddings(cfg.embedding_model)
    store = load_vector_store(embeddings, cfg.vector_store_dir)
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    contexts = retriever.invoke(query)

    llm = ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=0.2,
    )
    answer = build_shopping_advice(query, contexts, llm)

    print("\n=== 检索片段 ===")
    for i, item in enumerate(contexts, start=1):
        source = item.metadata.get("source", "unknown")
        preview = item.page_content[:130].replace("\n", " ")
        print(f"{i}. ({source}) {preview}...")

    print("\n=== Agent 输出 ===")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="CS6537 模拟 AI 购物代理（RAG）")
    parser.add_argument("--build-index", action="store_true", help="构建向量索引")
    parser.add_argument("--query", type=str, default=None, help="用户查询")
    parser.add_argument("--top-k", type=int, default=4, help="检索召回数量")
    args = parser.parse_args()

    if args.build_index:
        run_build_index()
        return

    if args.query:
        run_query(args.query, top_k=args.top_k)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
