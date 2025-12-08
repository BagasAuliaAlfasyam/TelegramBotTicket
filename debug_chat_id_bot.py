import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()  # baca .env

# Pilih mau pakai bot yang mana:
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_REPORTING")
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
