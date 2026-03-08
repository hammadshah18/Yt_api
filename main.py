from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import whisper
import tempfile
import yt_dlp
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from langdetect import detect

import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import os


load_dotenv()

# --------------------------
# CONFIG
# --------------------------
whisper_model = whisper.load_model("tiny")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.5-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(
    title="YouTube RAG Chat API",
    version="2.0"
)

vector_stores = {}
documents_store = {}

# --------------------------
# REQUEST MODELS
# --------------------------

class VideoRequest(BaseModel):
    video_url: str


class ChatRequest(BaseModel):
    video_url: str
    question: str


# --------------------------
# UTILS
# --------------------------

def extract_video_id(url: str):

    parsed = urlparse(url)

    # youtu.be short links
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    # youtube shorts
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]

        qs = parse_qs(parsed.query)

        if "v" in qs:
            return qs["v"][0]

    return None

# if not os.path.exists("yt_audio.mp3"):# this feature will rempve in future is code wont work
#     raise Exception("Audio download failed")

def download_audio(url):  # this feature will rempve in future is code wont work
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'yt_audio.%(ext)s',
        'quiet': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android']
            }
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])   

        
# if not os.path.exists("yt_audio.mp3"):
#     raise Exception("Audio download failed") 

def whisper_transcribe_video(video_url):
    try:
        import os
        import tempfile
        import yt_dlp

        temp_dir = tempfile.gettempdir()
        audio_template = os.path.join(temp_dir, "yt_audio")
        audio_path = audio_template + ".mp3"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_template,
            "noplaylist": True,
            "quiet": True,
            "continuedl": False,
            "nopart": True,
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 30,
            "js_runtimes": {"deno": {}},
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if not os.path.exists(audio_path):
            raise Exception("Audio file was not downloaded")

        result = whisper_model.transcribe(audio_path)

        transcript_text = ""
        for segment in result["segments"]:
            start = segment["start"]
            text = segment["text"]
            transcript_text += f"[{start:.2f}s] {text} "

        os.remove(audio_path)

        return transcript_text

    except Exception as e:
        print("Whisper transcription error:", e)
        return None
    
    

def get_transcript(video_id, video_url=None):

    transcript = None

    # First try YouTube transcript API
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)

        text = ""

        for chunk in transcript_list:
            start = getattr(chunk, "start", 0)
            content = getattr(chunk, "text", "")

            text += f"[{start:.2f}s] {content} "

        transcript = text
        print("Transcript source: YouTube captions")

    except (TranscriptsDisabled, NoTranscriptFound):
        transcript = None
        print("No captions found, using Whisper")

    # If transcript not found → Whisper fallback
    if transcript is None and video_url is not None:

        transcript = whisper_transcribe_video(video_url)

    return transcript


def chunk_text(text, chunk_size=500):

    chunks = []

    words = text.split()

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i+chunk_size])

        chunks.append(chunk)

    return chunks


def build_vector_store(video_id, chunks):

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    vector_stores[video_id] = index
    documents_store[video_id] = chunks


def retrieve_chunks(video_id, question, k=5):

    index = vector_stores[video_id]

    question_embedding = embedding_model.encode([question])

    D, I = index.search(np.array(question_embedding), k)

    results = []

    for idx in I[0]:

        results.append(documents_store[video_id][idx])

    return results


# --------------------------
# ROUTES
# --------------------------

@app.get("/")
def home():

    return {
        "status": "running"
    }


@app.post("/process_video")
def process_video(req: VideoRequest):

    video_id = extract_video_id(req.video_url)

    transcript = get_transcript(video_id, req.video_url)
    thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

    if not transcript:
        return {"error": "Transcript not available"}

    chunks = chunk_text(transcript, chunk_size=200)

    build_vector_store(video_id, chunks)

    return {
        "video_id": video_id,
        "thumbnail": thumbnail,
        "total_chunks": len(chunks),
        "total_chunks_indexed": len(chunks),
        "status": "indexed"
    }


@app.post("/chat")
def chat(req: ChatRequest):

    video_id = extract_video_id(req.video_url)

    if video_id not in vector_stores:

        return {
            "error": "Video not processed. Call /process_video first."
        }

    relevant_chunks = retrieve_chunks(video_id, req.question)

    context = "\n".join(relevant_chunks)

    try:
        lang = detect(req.question)
    except:
        lang = "unknown"

    prompt = f"""
You are an AI assistant that answers questions about a YouTube video.

Context from transcript:
{context}

User question:
{req.question}

Important rules:
- Answer in the SAME language as the user question.
- If the user asks in Urdu, answer in Urdu.
- If the user asks in Sindhi or Roman Sindhi, respond the same way.
- If the user asks in Roman Urdu, respond in Roman Urdu.
- Use timestamps if present.
"""

    response = llm.generate_content(prompt)

    return {

        "video_id": video_id,
        "language_detected": lang,
        "answer": response.text,
        "context_chunks_used": relevant_chunks

    }