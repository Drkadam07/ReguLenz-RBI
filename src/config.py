from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b-instruct"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    EVAL_DATA_PATH: Path = BASE_DIR / "data" / "evaluation" / "eval_benchmark.json"
    
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)