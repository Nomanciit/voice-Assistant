import os
import asyncio
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, silero, elevenlabs
from livekit.rag import Index, ChatMode

load_dotenv()

# Load the index for retrieval-augmented generation (RAG)
index = Index.load("/path/to/your/index")  # Update with the correct path to your index

dental_chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoid using unpronounceable punctuation."
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        chat_ctx=initial_ctx,
    )
    assistant.start(ctx.room)

    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        # Retrieve context using the dental_chat_engine
        context = dental_chat_engine.query(txt)

        # Append user message and retrieved context to the chat context
        chat_ctx = assistant.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        chat_ctx.append(role="system", text=f"Relevant context: {context}")

        # Generate a response using the LLM
        stream = llm.openai.LLM().chat(chat_ctx=chat_ctx)
        await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
