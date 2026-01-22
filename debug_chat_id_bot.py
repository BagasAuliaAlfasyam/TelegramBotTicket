import os
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load .env.local kalau ada, otherwise .env
env_local = Path(__file__).parent / ".env.local"
if env_local.exists():
    load_dotenv(env_local)
    print("✅ Loaded from .env.local")
else:
    load_dotenv()
    print("⚠️ Loaded from .env (production)")

# Pilih mau pakai bot yang mana:
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_COLLECTING")
# atau untuk reporting:
# TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_REPORTING")


async def chatid_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    text = (
        f"chat_id: {chat.id}\n"
        f"type: {chat.type}\n"
        f"title: {chat.title}"
    )
    await update.message.reply_text(text)


def main():
    if not TOKEN:
        raise RuntimeError("TOKEN belum di-set. Cek .env kamu bro.")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("chatid", chatid_handler))
    app.run_polling()


if __name__ == "__main__":
    main()
