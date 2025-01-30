import asyncio
from typing import Annotated
import re
import os
from dotenv import load_dotenv
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatImage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import  openai, silero, llama_index
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import ChatMode
import requests
from livekit.plugins.elevenlabs import tts
from livekit.plugins.deepgram import stt


load_dotenv()

# Initialize RAG components
PERSIST_DIR = "./dental-knowledge-storage"
if not os.path.exists(PERSIST_DIR):
    # Load dental knowledge documents and create index
    documents = SimpleDirectoryReader("dental_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load existing dental knowledge index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Create chat engine for dental knowledge
dental_chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)

class DentalAssistantFunction(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate dental issues using vision capabilities,"
            "for example, an image of teeth, gums, or the webcam feed showing the same."
        )
    )
    async def analyze_dental_image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            )
        ],
    ):
        print(f"Analyzing dental image: {user_msg}")
        return None

    @agents.llm.ai_callable(
        description="Called when a user wants to book an appointment. This function sends a booking link to the provided email address and name."
    )
    async def book_appointment(
        self,
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The email address to send the booking link to"
            ),
        ],
        name: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The name of the person booking the appointment"
            ),
        ],
    ):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return "The email address seems incorrect. Please provide a valid one."

        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            headers = {'Content-Type': 'application/json'}
            data = {'email': email, 'name': name}
            response = requests.post(webhook_url, json=data, headers=headers)
            response.raise_for_status()
            return f"Dental appointment booking link sent to {email}. Please check your email."
        except requests.RequestException as e:
            print(f"Error booking appointment: {e}")
            return "There was an error booking your dental appointment. Please try again later."

    async def check_appointment_status(
        self,
        email: str,
    ):
        api_token = os.getenv('API_TOKEN')
        print("Checking dental appointment status")

        try:
            api_url = f"{os.getenv('CRM_CONTACT_LOOKUP_ENDPOINT')}?email={email}"
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            for contact in data.get('contacts', []):
                if 'livekit_appointment_booked' in contact.get('tags', []):
                    return "You have successfully booked a dental appointment."
            return "You haven't booked a dental appointment yet. Would you like assistance in scheduling one?"
        except requests.RequestException as e:
            print(f"Error during API request: {e}")
            return "Error checking the dental appointment status."

    @agents.llm.ai_callable(
        description="Assess the urgency of a dental issue and determine if a human agent should be called."
    )
    async def assess_dental_urgency(
        self,
        symptoms: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Description of the dental symptoms or issues"
            ),
        ],
    ):
        urgent_keywords = ["severe pain", "swelling", "bleeding", "trauma", "knocked out", "broken"]
        if any(keyword in symptoms.lower() for keyword in urgent_keywords):
            return "call_human_agent"
        else:
            return "Your dental issue doesn't appear to be immediately urgent, but it's still important to schedule an appointment soon for a proper evaluation."

async def get_video_track(room: rtc.Room):
    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    """Your name is Ghazal, a dental assistant for caimplant clinic Dental .
                     - You are soft, caring, and have a bit of humor in your responses to keep the conversation engaging and friendly. Your role is crucial in creating a positive, supportive, and professional environment that encourages patients to feel confident and comfortable with their dental care journey.
                    
                    * Core Responsibilities:
                     - Provide appointment booking for dental care services, including urgent attention, routine check-ups, and long-term treatments. Emphasize that prices depend on individual needs, which can only be assessed during an onsite consultation.
                     - Access a comprehensive dental knowledge base to provide accurate information about dental procedures, conditions, and care. This includes dental implants and full-arch procedures.
                     - Navigate insurance and appointments efficiently, ensuring patients have a seamless experience.
                   
                    *Patient Interaction Guidelines:
                       **Show Empathy and Care
                    
                         - Actively listen to patient concerns and validate their feelings with phrases like, "I understand how you feel" or "We’re here to help you."
                         - Avoid being pushy—hear them out first, then guide them gently toward a free consultation.
                         - Keep your tone energetic and eager to help! Never sound uninterested or tired.
                    *Effective Communication Techniques
                    
                     - Speak clearly and use simple language to ensure patients understand the information.
                     - Be patient and friendly in all interactions, ensuring that patients feel heard and valued.
                     - Keep your tone lively and professional—never eat, smoke, or chew gum during calls or chats.
                    
                    *Goals in Every Interaction:
                     - Encourage appointment bookings by highlighting the importance of regular dental care and the benefits of an in-person visit.
                     - Build trust and rapport by engaging patients in short, multiple interactions, even for lengthy information.
                     - Subtly gather patient details (name and email) during the conversation:
                     - Politely ask for their name early in the interaction.
                     - Encourage them to type their email address to avoid mistakes and reconfirm it afterward.
                     - Reject non-dental queries politely, stating your purpose.
                     - By embodying these principles, you’ll help maintain the high standards of our clinic, making patients feel supported and confident in their dental care. Welcome aboard, and we look forward to your success!"""
                ),
            )
        ]
    )

    #gpt = openai.LLM(model="gpt-4o-mini")
    gpt = openai.LLM.with_groq(model="llama-3.2-3b-preview")
    eleven_tts=tts.TTS(
        api_key="sk_b179d6c09757ec2cb015f23121ac91778193c75006938954",
        model="eleven_turbo_v2_5",
        voice=tts.Voice(
            id="5MRtDX7fpi72xPHrBDUS",
            name="Ghazal 1",
            category="premade",
            settings=tts.VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            ),
        ),
        language="en",
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
    )
    deepgram_stt = stt.STT(
        model="nova-2-general",
        interim_results=True,
        smart_format=True,
        punctuate=True,
        filler_words=True,
        profanity_filter=False,
        keywords=[("LiveKit", 1.5)],
        language="en-US",
    )

 
    latest_image: rtc.VideoFrame | None = None
    human_agent_present = False

    # Create a combined LLM that uses both GPT and the dental knowledge base
    combined_llm = llama_index.LLM(
        chat_engine=dental_chat_engine
    )

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        #stt=openai.STT.with_groq(model="whisper-large-v3-turbo"),
        stt = deepgram_stt,
        llm=combined_llm,
        #tts=openai.TTS(),
        tts = eleven_tts,
        fnc_ctx=DentalAssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False, human_agent_present: bool = False, ai_assistance_required: bool = False):
        if human_agent_present and not ai_assistance_required:
            print("Human agent is present. AI assistant is silent.")
            return

        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            print(f"Calling with latest image")
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))
        
        # First try to get response from dental knowledge base
        try:
            response = await dental_chat_engine.chat(text)
            if response and response.response:
                stream = response.response
            else:
                # Fallback to GPT if no relevant information found
                stream = gpt.chat(chat_ctx=chat_context)
        except Exception as e:
            print(f"Error with dental knowledge base: {e}")
            stream = gpt.chat(chat_ctx=chat_context)

        await assistant.say(stream, allow_interruptions=True)

    async def follow_up_appointment(email: str):
        fnc = assistant.fnc_ctx
        await asyncio.sleep(20)
        print(f"Finished waiting, checking dental appointment status for {email}")
        status = await fnc.check_appointment_status(email)
        await asyncio.create_task(_answer(status))

    async def create_sip_participant(phone_number, room_name):
        print("trying to call an agent")
        LIVEKIT_URL = os.getenv('LIVEKIT_URL')
        LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
        LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
        SIP_TRUNK_ID = os.getenv('SIP_TRUNK_ID')

        livekit_api = api.LiveKitAPI(
            LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
        )

        sip_trunk_id = SIP_TRUNK_ID
        await livekit_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=sip_trunk_id,
                sip_call_to=phone_number,
                room_name=f"{room_name}",
                participant_identity=f"sip_{phone_number}",
                participant_name="Human Agent",
                play_ringtone=1
            )
        )
        await livekit_api.aclose()

    @ctx.room.on("participant_disconnected")
    def on_customer_agent_finished(RemoteParticipant: rtc.RemoteParticipant):
        print("Human Agent disconnected. AI Agent taking over.")
        asyncio.create_task(_answer("Human Agent interaction completed. Politely ask if it was helpful and if user is happy to proceed with the in-person appointment."))

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message and not human_agent_present:
            asyncio.create_task(_answer(msg.message))
        elif msg.message and human_agent_present and "help me" in msg.message.lower():
            asyncio.create_task(_answer(msg.message, human_agent_present=True, ai_assistance_required=True))
        else:
            print("No Assistance is needed as Human agent is tackling the communication")

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        nonlocal human_agent_present
        if len(called_functions) == 0:
            return
        
        function = called_functions[0]
        function_name = function.call_info.function_info.name
        print(function_name)
        
        if function_name == "assess_dental_urgency":
            result = function.result
            if result == "call_human_agent":
                print("calling an agent")
                human_agent_phone = os.getenv('HUMAN_AGENT_PHONE')
                asyncio.sleep(10)
                asyncio.create_task(create_sip_participant(human_agent_phone, ctx.room.name))
                human_agent_present = True
            else:
                asyncio.create_task(_answer(result, human_agent_present=False))

        elif function_name == "book_appointment":
            email = called_functions[0].call_info.arguments.get("email")
            if email:
                asyncio.create_task(follow_up_appointment(email))
        elif function_name == "analyze_dental_image":
            user_instruction = called_functions[0].call_info.arguments.get("user_msg")
            asyncio.create_task(_answer(user_instruction, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hello! I'm Ghazal, a dental assistant for caimplant clinic Dental . Can I know if you are the patient or you're representing the patient?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame
            await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))