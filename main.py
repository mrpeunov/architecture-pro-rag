import logging
from typer import Typer
from dotenv import load_dotenv
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


if __name__ == "__main__":
    manager()
