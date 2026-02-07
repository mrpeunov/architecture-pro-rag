import logging
import time
from dataclasses import asdict
from pathlib import Path
import chromadb
from redis import Redis
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb.api.models import Collection


from app.chunker import MarkdownChunker, Chunk

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()


class EmbeddingsSearcher:
    def __init__(self, ):
        self.client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        self.embeddings_manager = EmbeddingsManager()
        self.redis = Redis(host="localhost", port=6379)

    def search(self, query: str, n_results: int = 5) -> list[dict[str, str]]:
        collection_name = self.redis.get("current_collection").decode("utf-8")

        query_embedding = self.embeddings_manager.create_embeddings([query])[0]

        collection = self.client.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        final_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        )):
            logger.info(f"\n--- Результат {i + 1} (сходство: {1 - distance:.3f}) ---")
            logger.info(f"Файл: {metadata['file_path']}")
            logger.info(f"Заголовок: {metadata['title']}")
            logger.info(f"Чанк {metadata['chunk_index'] + 1}/{metadata['total_chunks']}")
            logger.info(f"Текст: {doc[:300]}...")
            final_results.append({
                "file_path": metadata['file_path'],
                "title": metadata['title'],
                "text": doc,
            })

        return final_results


class GenerateEmbeddingsPipeline:
    BATCH_SIZE = 100

    def __init__(
        self,
        directory: str = "knowledge_base"
    ):
        chromadb.PersistentClient(path="./chroma_db")
        self.client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        self.redis = Redis(host="localhost", port=6379)
        self.embeddings_manager = EmbeddingsManager()
        self.directory = Path(directory)
        self.chunker = MarkdownChunker()

    def run(self):
        """Обработка всех markdown файлов в директории"""
        md_files = self._get_files()
        all_chunks = self._get_chunks(md_files)

        collection_name = self._create_collection_name()
        collection = self._create_collection(collection_name)

        self._load_chunks(collection, all_chunks)
        logger.info(f"Загружено {len(all_chunks)} чанков в коллекцию '{collection_name}'")
        self.redis.set("current_collection", collection_name)

    def _create_collection(self, collection_name: str) -> Collection:
        return self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Косинусная схожесть
        )

    def _create_collection_name(self) -> str:
        return f"knowledge_base_{int(time.time())}"

    def _get_files(self) -> list[Path]:
        md_files = []
        for item in self.directory.iterdir():
            if item.is_file() and item.suffix == '.md':
                md_files.append(item)
            elif item.is_dir():
                md_files.extend(item.glob("*.md"))
        logger.info(f"Найдено {len(md_files)} markdown файлов")
        return md_files

    def _get_chunks(self, md_files: list[Path]) -> list[Chunk]:
        all_chunks = []
        for file_path in md_files:
            try:
                chunks = self.chunker.chunk_file(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Обработан {file_path}: {len(chunks)} чанков")
            except Exception as e:
                logger.info(f"Ошибка обработки {file_path}: {e}")

        if not all_chunks:
            logger.info("Нет чанков для загрузки")

        return all_chunks

    def _load_chunks(self, collection: Collection, all_chunks: list[Chunk]) -> None:
        # Подготавливаем данные для загрузки
        ids = [chunk.metadata.chunk_id for chunk in all_chunks]
        texts = [chunk.text for chunk in all_chunks]
        metadatas = [asdict(chunk.metadata) for chunk in all_chunks]

        # Создаем эмбеддинги
        logger.info("Создание эмбеддингов...")
        embeddings = self.embeddings_manager.create_embeddings(texts)

        for i in range(0, len(ids), self.BATCH_SIZE):
            batch_ids = ids[i:i + self.BATCH_SIZE]
            batch_embeddings = embeddings[i:i + self.BATCH_SIZE]
            batch_metadatas = metadatas[i:i + self.BATCH_SIZE]
            batch_texts = texts[i:i + self.BATCH_SIZE]

            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Загружено {i + len(batch_ids)}/{len(ids)} чанков")
