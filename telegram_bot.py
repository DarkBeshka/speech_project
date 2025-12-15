import asyncio
import io
import logging
import os
from pathlib import Path

import soundfile as sf
import torch
from TTS.utils.synthesizer import Synthesizer
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

BASE_DIR = Path(__file__).parent
CFG_PATH = BASE_DIR / "ruslan_glowtts_exp" / "run-December-15-2025_10+31AM-0000000" / "config.json"
MODEL_PATH = BASE_DIR / "ruslan_glowtts_exp" / "run-December-15-2025_10+31AM-0000000" / "best_model_84384.pth"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

synth = Synthesizer(
    tts_checkpoint=str(MODEL_PATH),
    tts_config_path=str(CFG_PATH),
    use_cuda=torch.cuda.is_available(),
)


def text_to_wav_bytes(text: str) -> io.BytesIO:
    """Синтезирует речь и возвращает WAV в буфере памяти."""
    wav = synth.tts(text)
    buf = io.BytesIO()
    sf.write(buf, wav, synth.output_sample_rate, format="WAV")
    buf.seek(0)
    buf.name = "tts.wav"
    return buf

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправь мне текст, и я озвучу его"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    if not text:
        await update.message.reply_text("Текст пустой. Пришли что-нибудь осмысленное.")
        return

    await update.message.reply_chat_action(action=ChatAction.RECORD_VOICE)

    loop = asyncio.get_running_loop()
    try:
        wav_bytes = await loop.run_in_executor(None, text_to_wav_bytes, text)
    except Exception as exc:
        logger.exception("Ошибка синтеза: %s", exc)
        await update.message.reply_text("Не удалось синтезировать аудио, попробуй снова.")
        return

    await update.message.reply_voice(voice=wav_bytes, caption="Готово!")


def main() -> None:
    token = "TOKEN"

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("========Бот запущен=======")
    application.run_polling()

if __name__ == "__main__":

    main()
