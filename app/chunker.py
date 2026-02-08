import hashlib
from pathlib import Path
import markdown
from dataclasses import dataclass
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkMetadata:
    file_path: str
    file_name: str
    directory: str
    title: str
    chunk_id: str
    chunk_index: int
    total_chunks: int
    start_position: int
    end_position: int



@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata



class MarkdownChunker:
    """Класс для обработки markdown файлов с сохранением структуры"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Инициализация сплиттера.
        chunk_size: приблизительное количество токенов в чанке
        chunk_overlap: перекрытие между чанками для сохранения контекста
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Приоритет разбивки по структуре MD
            keep_separator=True
        )


    def extract_metadata_from_md(self, file_path: Path) -> dict[str, str]:
        """Извлечение базовых метаданных из markdown файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'directory': str(file_path.parent),
            'title': file_path.stem
        }

        # Попытка извлечь заголовок первого уровня из содержимого
        for line in lines:
            if line.startswith('# '):
                metadata['title'] = line[2:].strip()
                break
            elif line.startswith('## '):
                metadata['title'] = line[3:].strip()
                break

        return metadata

    def markdown_to_text(self, md_content: str) -> str:
        """Конвертация markdown в чистый текст с сохранением структуры"""
        # Преобразуем markdown в HTML
        html = markdown.markdown(md_content)
        # Извлекаем текст, сохраняя разделение на абзацы
        soup = BeautifulSoup(html, 'html.parser')

        # Сохраняем структуру заголовков как часть текста
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            header.insert_before(f"\n{'#' * int(header.name[1])} ")

        text = soup.get_text()
        # Удаляем лишние пустые строки, но сохраняем структуру
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Разбивка файла на чанки с метаданными"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Конвертируем markdown в текст
        clean_text = self.markdown_to_text(content)

        # Разбиваем на чанки
        chunks = self.text_splitter.split_text(clean_text)

        # Собираем метаданные для каждого чанка
        base_metadata = self.extract_metadata_from_md(file_path)
        chunked_documents = []

        for i, chunk in enumerate(chunks):
            # Генерируем уникальный ID для чанка на основе пути и позиции
            chunk_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()

            # Определяем позицию в исходном документе (приблизительно)
            start_pos = 0 if i == 0 else len(chunk) * i - self.text_splitter._chunk_overlap

            chunked_documents.append(
                Chunk(
                    text=chunk,
                    metadata=ChunkMetadata(
                        **base_metadata,
                        chunk_id=chunk_id,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        start_position=start_pos,
                        end_position=start_pos + len(chunk),
                    )
                )
            )

        return chunked_documents