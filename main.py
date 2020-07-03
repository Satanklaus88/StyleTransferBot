import asyncio

from aiogram import Bot, Dispatcher, executor
from config import BOT_TOKEN
from aiogram.contrib.fsm_storage.memory import MemoryStorage

bot = Bot(BOT_TOKEN, parse_mode="HTML")
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


if __name__ == '__main__':
    from handlers import dp
    executor.start_polling(dp)
