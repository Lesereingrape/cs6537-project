from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def load_product_documents(product_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for file_path in sorted(product_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": file_path.name}))
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=80)
    return splitter.split_documents(documents)


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(chunks: List[Document], embeddings: HuggingFaceEmbeddings, persist_dir: Path) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(persist_dir))
    return store


def load_vector_store(embeddings: HuggingFaceEmbeddings, persist_dir: Path) -> Chroma:
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


def build_shopping_advice(query: str, contexts: List[Document], llm: ChatOpenAI) -> str:
    context_text = "\n\n".join([f"[{item.metadata.get('source', 'unknown')}]\n{item.page_content}" for item in contexts])

    prompt = ChatPromptTemplate.from_template(
        """
你是一个电商导购助手，请基于给定资料给出购物建议。

要求：
1) 先给出推荐商品类别与理由。
2) 明确引用资料中的关键参数（如价格区间、容量、功效、材质）。
3) 最后输出一个“建议购买清单（最多3项）”。
4) 如果资料不足，明确说明“不足信息”。

用户问题：
{query}

参考资料：
{context}
"""
    )

    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context_text})
    return result.content
