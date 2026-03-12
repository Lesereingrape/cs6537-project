from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os


load_dotenv()


class AppConfig(BaseModel):
    base_dir: Path = Path(__file__).resolve().parent.parent
    product_data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "products"
    vector_store_dir: Path = Path(__file__).resolve().parent.parent / "data" / "vector_store"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = os.getenv("OPENAI_MODEL", "THUDM/chatglm3-6b")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str | None = os.getenv("OPENAI_BASE_URL") or None


def get_config() -> AppConfig:
    return AppConfig()
