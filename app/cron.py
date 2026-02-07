import logging

import aiomisc
from aiomisc.service.cron import CronService

from app.embeddings import GenerateEmbeddingsPipeline

logger = logging.getLogger(__name__)


class CronEmbeddingGenerator(CronService):
    def __init__(self, spec: str = "* * * * *"):
        super().__init__()
        self.spec = spec
        self.pipeline = GenerateEmbeddingsPipeline()

    async def callback(self) -> None:
        logger.info('Running cron callback')
        self.pipeline.run()

    async def start(self) -> None:
        self.register(self.callback, spec=self.spec)
        await super().start()


def run_cron_app():
    service = CronEmbeddingGenerator()
    with aiomisc.entrypoint(service, log_config=False) as loop:
        loop.run_forever()
