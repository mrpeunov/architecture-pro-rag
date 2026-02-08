import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from app.llm import LLMAnswerer
from dotenv import load_dotenv

load_dotenv(".env")
token = os.getenv("BOT_TOKEN")

bot = Bot(token=token)
llm_answerer = LLMAnswerer()
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hi! Just ask me something.")


@dp.message()
async def echo_all(message: types.Message):
    answer = llm_answerer.get_answer(message.text)
    await message.answer(answer)


def run_bot_app():
    asyncio.run(dp.start_polling(bot))
