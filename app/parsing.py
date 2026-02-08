import json
import logging
import time
import re
import os
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from typing import Any
import requests


logger = logging.getLogger(__name__)


class ParsingPipeline:
    TIMEOUT = 1

    def __init__(self, main_directory, pages_config_path: str, replacing_map_path: str) -> None:
        self._main_directory = main_directory
        self._pages: dict[str, list[str]] = self._load_file(pages_config_path)
        self._replacing_map: dict[str, str] = self._load_file(replacing_map_path)

    def run(self):
        for directory, pages in self._pages.items():
            logger.info(f"Обрабатываем категорию: {directory}")
            for page in pages:
                try:
                    logger.info(f"Загружаем страницу: {page}")
                    md_content = self._parse_page(page=page)
                    replaced_md_content = self._replace_with_case(md_content)
                    self._save_to_file(
                        filepath=f"{self._main_directory}/{directory}/{self._create_filename(page)}",
                        content=replaced_md_content
                    )
                    time.sleep(self.TIMEOUT)

                except Exception as e:
                    logger.exception(e)
                    logger.info(f"Page: {page}: Не удалось загрузить")

        logger.info(f"Парсинг завершен. Результаты сохранены в {self._main_directory}")

    def _load_file(self, pages_config_path: str) -> Any:
        try:
            with open(pages_config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Ошибка при загрузке из файла: {pages_config_path}: {e}")

    def _parse_page(self, page: str) -> str:
        page_dict = self._fetch_page_from_api(page)

        html_content = page_dict['parse']['text']['*']
        title = page_dict['parse']['title']

        soup = BeautifulSoup(html_content, 'html.parser')
        self._clean_html(soup)

        content_element = self._extract_main_content(soup)
        if not content_element:
            raise ValueError("Не удалось найти основной контент")

        markdown_content = self._html_to_markdown(content_element)

        cleaned_markdown = self._clean_markdown(markdown_content)

        formatted_markdown = self._format_markdown(title, cleaned_markdown, page)

        replaced_markdown = self._replace_with_case(formatted_markdown)

        return replaced_markdown

    def _fetch_page_from_api(self, page: str) -> dict | None:
        """Загружает содержимое страницы через Fandom API"""
        api_url = "https://attackontitan.fandom.com/api.php"
        params = {
            'action': 'parse',  # Действие: разобрать страницу
            'format': 'json',  # Формат ответа
            'page': f'{page}_(Anime)',  # Название страницы (например, "War_Hammer_Titan_(Anime)")
            'prop': 'text',  # Свойство: получить текст (можно также 'wikitext', 'images' и др.)
            'utf8': 1,  # Кодировка UTF-8
            'origin': '*',  # Разрешить кросс-доменные запросы (CORS)
        }
        headers = {'User-Agent': 'YourBotName/1.0 (your_email@example.com)'}
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup | None:
        """Извлекает основной контент статьи"""
        # Для Fandom вики
        content_selectors = [
            'div.mw-parser-output',
            'div#content',
            'article',
            'div.page-content',
            'main'
        ]

        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                # Создаем копию для безопасного редактирования
                content_copy = BeautifulSoup(str(content), 'html.parser')
                return content_copy

        return None

    def _clean_html(self, soup: BeautifulSoup) -> None:
        """Удаляет ненужные элементы из HTML"""
        # Удаляем скрипты и стили
        for tag in soup(['script', 'style', 'noscript', 'iframe', 'svg']):
            tag.decompose()

        # Удаляем все изображения
        for img in soup.find_all('img'):
            img.decompose()

        # Удаляем навигационные элементы и боковые панели
        for element in soup.find_all(class_=re.compile(r'nav|sidebar|toc|infobox|mw-editsection|reference')):
            element.decompose()

        # Удаляем таблицы (опционально, можно закомментировать если нужны)
        for table in soup.find_all('table'):
            table.decompose()

        # Удаляем пустые элементы
        for element in soup.find_all():
            if len(element.get_text(strip=True)) == 0 and not element.find_all():
                element.decompose()

    def _html_to_markdown(self, element: BeautifulSoup) -> str:
        """Конвертирует HTML в Markdown"""
        html_str = str(element)

        md_converter = md(
            html_str,
            heading_style="ATX",
            convert=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'strong', 'em', 'code', 'pre', 'blockquote']
        )

        return md_converter

    def _clean_markdown(self, markdown_text: str) -> str:
        """Очищает Markdown от японского текста и лишних элементов"""
        lines = markdown_text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Удаляем строки, содержащие японские символы
            if self._contains_japanese(line):
                continue

            # Удаляем строки с нежелательным содержанием
            if self._should_remove_line(line):
                continue

            # Очищаем строку от лишних символов
            cleaned_line = self._clean_line(line)

            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)

        return '\n'.join(cleaned_lines)

    def _contains_japanese(self, text: str) -> bool:
        """Проверяет, содержит ли текст японские символы"""
        # Диапазоны Unicode для японских символов
        japanese_ranges = [
            (0x3040, 0x309F),  # Хирагана
            (0x30A0, 0x30FF),  # Катакана
            (0x4E00, 0x9FFF),  # Кандзи (CJK Unified Ideographs)
            (0x3400, 0x4DBF),  # Кандзи Extension A
        ]

        for char in text:
            code = ord(char)
            for start, end in japanese_ranges:
                if start <= code <= end:
                    return True
        return False

    def _should_remove_line(self, line: str) -> bool:
        """Определяет, нужно ли удалить строку"""
        line_lower = line.lower().strip()

        # Ключевые слова для удаления
        remove_keywords = [
            'edit',
            'japanese',
            'nihongo',
            'kanji',
            'hiragana',
            'katakana',
            'この記事',
            'ウィキ',
            'ファイル:',
            'category:',
            'references',
            'external links',
            'see also',
            'navigation menu'
        ]

        for keyword in remove_keywords:
            if keyword in line_lower:
                return True

        # Удаляем слишком короткие строки (менее 3 символов)
        if len(line.strip()) < 3:
            return True

        return False

    def _clean_line(self, line: str) -> str:
        """Очищает строку от лишних символов"""
        # Удаляем множественные пробелы
        line = re.sub(r'\s+', ' ', line)

        # Удаляем маркеры редактирования [edit]
        line = re.sub(r'\[edit\]', '', line, flags=re.IGNORECASE)

        # Удаляем ссылки в квадратных скобках
        line = re.sub(r'\[.*?\]\((.*?)\)', r'\1', line)  # Markdown ссылки
        line = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', line)  # Wiki ссылки
        line = re.sub(r'\[\[(.*?)\]\]', r'\1', line)  # Простые wiki ссылки

        # Удаляем HTML entities
        line = re.sub(r'&[a-z]+;', '', line)

        return line.strip()

    def _format_markdown(self, title: str, content: str, link: str) -> str:
        """Форматирует финальный Markdown документ"""
        lines = [
            f"# {title}\n",
            "---\n",
            content
        ]
        return '\n'.join(lines)

    def _replace_with_case(self, content: str) -> str:
        # Создаем регулярное выражение для поиска всех слов из словаря
        # re.escape экранирует спецсимволы, re.IGNORECASE делает поиск регистронезависимым
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, REPLACING_MAP.keys())) + r')\b', re.IGNORECASE)

        def replace_match(match):
            original_word = match.group(0)
            replacement = self._replacing_map[original_word.lower()]

            # Сохраняем регистр
            if original_word.isupper():
                return replacement.upper()
            elif original_word.istitle():  # Первая буква заглавная
                return replacement.title()
            elif original_word.islower():
                return replacement.lower()
            else:
                # Для смешанных регистров (например, TiTaN) - приводим к нижнему
                return replacement.lower()

        return pattern.sub(replace_match, content)

    def _save_to_file(self, filepath: str, content: str) -> None:
        """Сохраняет контент в файл"""
        # Создаем директорию если она не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def _create_filename(self, title: str) -> str:
        title = self._replace_with_case(' '.join(title.split('_')))
        """Создает безопасное имя файла из заголовка"""
        # Удаляем недопустимые символы для имени файла
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
        safe_title = safe_title.replace(' ', '_')
        safe_title = safe_title[:100]  # Ограничиваем длину
        return f"{safe_title}.md"
