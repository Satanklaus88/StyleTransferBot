import asyncio
import os
import random
import shutil

import torch
import torchvision.models as models
from PIL import Image
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import Message

from config import admin_id, styles
from main import bot, dp
from nstmodels import Styletransfer
from utils import Load_image, Save_image

device = torch.device("cpu")


class NSTProceed(StatesGroup):
    Q1 = State()
    Q2 = State()


async def send_to_admin(*args):
    await bot.send_message(chat_id=admin_id, text="Бот запущен")


@dp.message_handler(Command("start"), state={None, NSTProceed.Q1, NSTProceed.Q2})
async def start(message: Message, state: FSMContext):
    await message.answer("Привет, этот бот переносит стиль с одной картинки на другую \n"
                         "Для более подробной информации наберите /help")
    if state is not None:
        await state.finish()


@dp.message_handler(Command("stop"), state={None, NSTProceed.Q1, NSTProceed.Q2})
async def start(message: Message, state: FSMContext):
    await message.answer("Перенос стиля прерван, начните сначала")
    if state is not None:
        await state.finish()


@dp.message_handler(Command("help"))
async def help(message: Message):
    await message.answer("Для работы бота необходимы два изображения: одно с контентом "
                         "и второе со стилем для переноса. Бот из всех картинок вырезает центральную квадратную часть,"
                         "так что старайтесь подавать изображения более-менее 'квадратные' :) \n"
                         "Наиболее интересные и красивые результаты получаются, когда на изображении контента нет "
                         "множества мелких деталей, а картинка стиля... например, похожа на эту :)")
    with open(styles[0][0], 'rb') as f:
        await message.answer_photo(f, caption=styles[0][1])
    await message.answer("В боте есть возможность выбрать случайное изображение стиля(помимо этого) из небольшого "
                         "набора, если вы не хотите искать свое :) \n"
                         "Для этого есть команда /getrandomstyle \n"
                         "Сохраните изображение стиля себе в галерею и затем отправьте его боту. \n"
                         "Для запуска переноса стиля - команда /stylize")


@dp.message_handler(Command("getrandomstyle"))
async def getrandomstyle(message: Message):
    n = random.randint(1, 4)
    with open(styles[n][0], 'rb') as f:
        await message.answer_photo(f, caption=styles[n][1])


@dp.message_handler(Command("stylize"), state=None)
async def start_nst(message: Message):
    await message.answer("Отправьте базовое изображение.\n"
                         "Для отмены /stop")
    usid = str(message.from_user.id)
    shutil.rmtree(f'user{usid}', ignore_errors=True)
    await NSTProceed.Q1.set()


@dp.message_handler(state=NSTProceed.Q1, content_types=['photo'])
async def send_context(message: Message, state: FSMContext):
    usid = str(message.from_user.id)
    os.mkdir('user' + usid)
    await message.photo[-1].download('user' + usid + '/context.jpg')
    await message.answer("Отправьте изображение стиля")
    await NSTProceed.next()


@dp.message_handler(state=NSTProceed.Q2, content_types=['photo'])
async def send_style(message: Message, state: FSMContext):
    usid = str(message.from_user.id)
    await message.photo[-1].download(f'user{usid}/style.jpg')
    await message.answer("Теперь подождите, пока переносится стиль (примерно 7 минут:)")
    style = Load_image(f'user{usid}/style.jpg', imsize=256)
    content, size = Load_image(f'user{usid}/context.jpg', imsize=256, return_size=True)
    if size > 512:
        size = 512
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    input_img = content.clone()
    st = Styletransfer(cnn, style, content)
    output = st.convert()
    Save_image(output, f'user{usid}/res.jpg')
    res = Image.open(f'user{usid}/res.jpg')
    resized = res.resize((size, size), resample=Image.LANCZOS)
    resized.save(f'user{usid}/output.jpg')
    with open(f'user{usid}/output.jpg', 'rb') as f:
        await message.answer_photo(f, caption='Готово!')
    shutil.rmtree(f'user{usid}', ignore_errors=True)
    await state.finish()


@dp.message_handler(content_types=['document', 'audio', 'photo', 'voice'])
async def echo(message: Message):
    await message.reply("Для справки наберите /help")


@dp.message_handler()
async def echo(message: Message):
    await message.reply("Для справки наберите /help")
