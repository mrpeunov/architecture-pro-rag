import logging

from typer import Typer

from app.bot import run_bot_app
from app.cron import run_cron_app
from app.embeddings import GenerateEmbeddingsPipeline
from dotenv import load_dotenv

from app.llm import LLMAnswerer
from app.parsing import ParsingPipeline

manager = Typer()

load_dotenv(".env")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@manager.command("run_parsing")
def run_parsing():
    parsing_pipeline = ParsingPipeline(
        main_directory="./knowledge_base",
        pages_config_path="./configs/pages-config.json",
        replacing_map_path='./configs/replacing-map.json',
    )
    parsing_pipeline.run()


@manager.command("run_cron")
def run_cron():
    run_cron_app()


@manager.command("generate_embeddings")
def generate_embeddings():
    pipeline = GenerateEmbeddingsPipeline()
    pipeline.run()


@manager.command("run_local_search")
def run_local_search():
    llm_answerer = LLMAnswerer()

    while True:
        query = input("Enter query or exit: ")

        if query == "exit":
            break

        try:
            answer = llm_answerer.get_answer(query)
            logging.warning(f"Answer: {answer}")

        except Exception as e:
            logging.error(e)


@manager.command("run_bot")
def run_bot():
    run_bot_app()


@manager.command("run_tests")
def run_tests():
    from app.test import tests

    tests()

if __name__ == "__main__":
    manager()