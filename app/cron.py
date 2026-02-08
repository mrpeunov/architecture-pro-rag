import logging

import aiomisc
from aiomisc.service.cron import CronService

from app.embeddings import GenerateEmbeddingsPipeline
from app.parsing import ParsingPipeline

logger = logging.getLogger(__name__)


class CronEmbeddingGenerator(CronService):
    def __init__(self, spec: str = "* * * * *"):
        super().__init__()
        self.spec = spec
        self.embedding_pipeline = GenerateEmbeddingsPipeline()
        self.parsing_pipeline = ParsingPipeline(
            main_directory="./knowledge_base",
            pages_config_path="./configs/pages-config.json",
            replacing_map_path='./configs/replacing-map.json',
        )

    async def callback(self) -> None:
        logger.info('Running cron callback')
        self.parsing_pipeline.run()
        self.embedding_pipeline.run()

    async def start(self) -> None:
        self.register(self.callback, spec=self.spec)
        await super().start()


def run_cron_app():
    service = CronEmbeddingGenerator()
    with aiomisc.entrypoint(service, log_config=False) as loop:
        loop.run_forever()
