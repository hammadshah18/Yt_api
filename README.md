# YouTube RAG Chat API

A powerful FastAPI-based Retrieval Augmented Generation (RAG) system that enables intelligent conversational interactions with YouTube video content. Extract transcripts, create vector embeddings, and ask questions about video content with AI-powered responses.

## 🎯 Features

- **YouTube Transcript Extraction**: Automatically fetches captions from YouTube videos
- **Automatic Fallback**: Uses Whisper for automatic speech recognition when captions aren't available
- **Vector Embeddings**: Creates semantic embeddings using Sentence Transformers
- **FAISS Vector Store**: Fast similarity search for relevant content retrieval
- **Multi-language Support**: Detects user language and responds accordingly
- **Gemini AI Integration**: Powered by Google's Gemini 2.5 Flash model
- **RESTful API**: Easy-to-use FastAPI endpoints

## 📋 Prerequisites

- Python 3.11+
- FFmpeg (required for audio processing)
- Google Gemini API Key
- Internet connection for YouTube video access

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd yt_plugin
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## 📦 Dependencies

### Core Dependencies
- **fastapi** (0.116.1): Web framework for building APIs
- **uvicorn** (0.35.0): ASGI server for FastAPI
- **pydantic** (2.12.3): Data validation and settings

### YouTube Processing
- **youtube-transcript-api** (1.2.4): Extract YouTube captions
- **yt-dlp** (2026.3.3): Download video audio
- **openai-whisper** (20250625): Automatic speech recognition
- **ffmpeg-python**: Audio processing

### AI & ML
- **google-generativeai** (0.8.5): Gemini AI integration
- **langchain**: LLM framework and utilities
- **sentence-transformers** (5.2.0): Create semantic embeddings
- **torch** (2.10.0): PyTorch for ML models

### Vector Database
- **faiss-cpu** (1.13.1): Fast similarity search
- **numpy** (2.4.1): Numerical computing

### Utilities
- **langdetect** (1.0.9): Language detection
- **python-dotenv** (1.0.1): Environment variable management

## 🔧 Configuration

### Available Settings
```python
# In main.py
whisper_model = whisper.load_model("tiny")  # Change to "base", "small", "medium", "large"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Change embedding model
chunk_size = 200  # Words per chunk (adjust for granularity)
```

## 🎮 Usage

### 1. Start the Server
```bash
uvicorn main:app --reload
```

Server will be available at: `http://localhost:8000`

### 2. API Documentation
Access interactive docs: `http://localhost:8000/docs`

### 3. API Endpoints

#### Health Check
```bash
GET /
```
Response:
```json
{
  "status": "running"
}
```

#### Process Video
Extract transcript and create vector store from a YouTube video.

```bash
POST /process_video
```

Request body:
```json
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

Response:
```json
{
  "video_id": "dQw4w9WgXcQ",
  "thumbnail": "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
  "total_chunks": 45,
  "total_chunks_indexed": 45,
  "status": "indexed"
}
```

#### Chat with Video
Ask questions about a processed YouTube video.

```bash
POST /chat
```

Request body:
```json
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "question": "What is the main topic of this video?"
}
```

Response:
```json
{
  "answer": "The main topic is...",
  "relevance_score": 0.95,
  "sources": [...]
}
```

## 📝 How It Works

### Workflow
1. **Video Processing**
   - Extract video ID from YouTube URL
   - Fetch transcript using YouTube API or Whisper fallback
   - Split transcript into chunks (default: 200 words)
   - Generate embeddings for each chunk

2. **Vector Store Creation**
   - Use Sentence Transformers to create semantic embeddings
   - Store embeddings in FAISS index
   - Maintain mapping between embeddings and original text

3. **Question Answering**
   - Detect user's language
   - Convert question to embedding
   - Search FAISS index for top-k relevant chunks
   - Provide context to Gemini AI
   - Generate answer in user's language

### Supported Video URL Formats
- Standard: `https://www.youtube.com/watch?v=VIDEO_ID`
- Short: `https://youtu.be/VIDEO_ID`
- Shorts: `https://www.youtube.com/shorts/VIDEO_ID`

## 🌍 Supported Languages

The system detects these languages and responds accordingly:
- English
- Spanish
- French
- German
- Chinese
- Japanese
- Arabic
- Hindi
- Urdu
- And 50+ more languages

## 🔍 Examples

### Example 1: Processing a Tutorial Video
```bash
curl -X POST "http://localhost:8000/process_video" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Example 2: Asking a Question
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "question": "What are the key points mentioned?"
  }'
```

### Example 3: Multi-language Support
```bash
# Question in Urdu
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "question": "یہ ویڈیو کس بارے میں ہے؟"
  }'
```

## ⚙️ Performance Optimization

### Tips for Better Results
1. **Adjust Chunk Size**: Smaller chunks = more granular results, larger chunks = broader context
2. **Model Selection**: Use "base" or "small" Whisper for faster transcription
3. **Embedding Model**: All-MiniLM-L6-v2 is lightweight; use larger models for better quality
4. **FAISS Parameters**: Adjust `k` value in `retrieve_chunks()` for more/fewer results

### System Requirements
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: CUDA-compatible GPU optional (faster inference)
- **Storage**: ~1GB for models

## 🛠️ Development

### Project Structure
```
yt_plugin/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
├── .venv/                  # Virtual environment
└── README.md              # This file
```

### Adding New Features
1. Create new routes in `main.py`
2. Define request/response models using Pydantic
3. Add utility functions as needed
4. Test with FastAPI docs (`/docs`)

## 🐛 Troubleshooting

### Issue: "Audio file was not downloaded"
**Solution**: Ensure FFmpeg is installed and in PATH
```bash
# Windows
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

### Issue: "Transcript not available"
**Solution**: The video doesn't have captions. The system will try Whisper, which requires:
- Internet connectivity
- Sufficient processing time (varies by video length)

### Issue: "GEMINI_API_KEY not found"
**Solution**: Ensure `.env` file exists with your API key
```env
GEMINI_API_KEY=your-key-here
```

### Issue: Out of Memory Errors
**Solution**: 
- Use smaller Whisper model (`whisper.load_model("tiny")`)
- Reduce chunk size
- Process shorter videos

## 📚 API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid URL format) |
| 422 | Validation Error |
| 500 | Server Error (transcript not available) |

## 🔐 Security Considerations

1. **API Keys**: Never commit `.env` file to version control
2. **Rate Limiting**: Implement rate limiting for production
3. **Input Validation**: All inputs validated via Pydantic
4. **HTTPS**: Use HTTPS in production

## 📈 Future Enhancements

- [ ] Database persistence for processed videos
- [ ] Advanced caching mechanisms
- [ ] Batch video processing
- [ ] Custom embedding models
- [ ] User authentication
- [ ] Rate limiting and quotas
- [ ] Video quality selection
- [ ] Response streaming

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For issues, questions, or suggestions:
- Create an Issue on GitHub
- Check existing documentation
- Review FastAPI docs: https://fastapi.tiangolo.com/

## 🙏 Acknowledgments

- FastAPI - Modern web framework
- LangChain - LLM framework
- Google Gemini - AI model
- Sentence Transformers - Embedding generation
- FAISS - Vector similarity search
- OpenAI Whisper - Speech recognition

---

**Last Updated**: March 2026  
**Version**: 2.0
