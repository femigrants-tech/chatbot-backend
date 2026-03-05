# Femigrants RAG Backend

A powerful RAG (Retrieval-Augmented Generation) backend built with FastAPI, Pinecone Assistant, LangChain, and Google Gemini.

## Features

- 🤖 **RAG-based Chat**: Intelligent question answering using your knowledge base
- 💬 **Context-Aware Conversations**: Maintains chat history for coherent multi-turn conversations
- 📁 **File Management**: Upload, view, and delete files in your knowledge base
- 🎨 **Beautiful UI**: Modern web interface for file management
- 🔐 **Secure**: API key-based authentication for Pinecone and Google Gemini

## Tech Stack

- **FastAPI**: Modern web framework for building APIs
- **Pinecone Assistant**: Vector database for semantic search
- **LangChain**: Framework for building LLM applications
- **Google Gemini**: Large language model for generating responses
- **CORS Enabled**: Ready for frontend integration

## Prerequisites

- Python 3.8 or higher
- Pinecone account and API key ([Sign up here](https://www.pinecone.io/))
- Google Cloud account with Gemini API access ([Get API key](https://makersuite.google.com/app/apikey))

## Installation

1. **Clone the repository**
   ```bash
   cd "Backend"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your API keys:
   ```
   PINECONE_API_KEY=your_actual_pinecone_api_key
   PINECONE_INDEX_NAME=femigrants-rag-chatbot
   PINECONE_ASSISTANT_NAME=femigrants-assistant
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

## Running the Application

1. **Start the FastAPI server**
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the file manager interface**
   - Open `file_manager.html` in your web browser
   - Or navigate to: `file:///path/to/Backend/file_manager.html`

3. **API Documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Chat Endpoints

**POST** `/chat`
- Chat with RAG system using your knowledge base
- Maintains conversation context with `chat_context` parameter

### File Management Endpoints

**POST** `/files/upload` - Upload files with optional metadata  
**GET** `/files` - List all files (with optional filtering)  
**GET** `/files/{file_id}` - Get file details and download URL  
**DELETE** `/files/{file_id}` - Delete a single file  
**POST** `/files/bulk-delete` - Delete multiple files at once  
**PUT** `/files/{file_id}/metadata` - Update file metadata  
**GET** `/files/by-status/{status}` - Get files by status (Available, Processing, Failed)  
**GET** `/files/statistics` - Get storage and file statistics  

### Document Retrieval Endpoints

**POST** `/documents/search` - Search documents with ranking and scores  
**POST** `/documents/retrieve` - Retrieve context without chat response  
**GET** `/documents/preview/{file_id}` - Preview document content  

### Monitoring

**GET** `/health` - Health check and connectivity status

📖 **Full API Documentation**: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed endpoint documentation with examples.

### Quick Example

**Important:** The parameter name for conversation history is `chat_context`. Use this exact name when sending requests from your frontend.

```javascript
// Chat with context
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "What is machine learning?",
    chat_context: [
      {"role": "user", "content": "Tell me about AI"},
      {"role": "assistant", "content": "AI is..."}
    ]
  })
});

const data = await response.json();
// data.response - AI response
// data.context_used - Retrieved context snippets
```

## Frontend Integration

The API is configured to accept requests from `http://localhost:3000` (configurable in `main.py`).

### Example: Chat Request from Frontend

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: userMessage,
    chat_context: conversationHistory  // Array of {role, content} objects
  })
});

const data = await response.json();
console.log(data.response);  // AI response
console.log(data.context_used);  // Retrieved context
```

### Chat Context Format

Send conversation history using this format:

```javascript
chat_context: [
  { role: "user", content: "Previous user message" },
  { role: "assistant", content: "Previous AI response" },
  { role: "user", content: "Another user message" },
  { role: "assistant", content: "Another AI response" }
]
```

## Supported File Types

Pinecone Assistant supports various file formats including:
- Text files (.txt, .md)
- PDF documents (.pdf)
- Word documents (.doc, .docx)
- And more...

For multimodal support (images in PDFs), the system can also extract and process visual content.

## File Metadata

You can add metadata to files during upload to help organize and filter your knowledge base:

```json
{
  "category": "documentation",
  "author": "admin",
  "date": "2024-01-15",
  "project": "femigrants"
}
```

Metadata can be used to:
- Filter files in the file list
- Filter context during chat queries
- Organize and categorize content

## Configuration

### Changing CORS Origins

Edit `main.py` to modify allowed origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Add your frontend URL
    ...
)
```

### Changing Assistant Name

Edit `.env` file:
```
PINECONE_ASSISTANT_NAME=your-custom-assistant-name
PINECONE_INDEX_NAME=your-custom-index-name
```

### Adjusting Gemini Temperature

Edit `main.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,  # Adjust between 0 (deterministic) and 1 (creative)
)
```

## Troubleshooting

### "PINECONE_API_KEY environment variable is not set" or "GEMINI_API_KEY environment variable is not set"
- Make sure you've created a `.env` file from `.env.example`
- Verify your API keys are correctly set in `.env`

### "Failed to get or create assistant"
- Verify your Pinecone API key is valid
- Check if you have the correct permissions in your Pinecone account
- The assistant will be created automatically on first run

### "Error uploading file"
- Check file size limits (depends on your Pinecone plan)
- Ensure the file format is supported
- Verify you have sufficient storage in your Pinecone account

### CORS Errors
- Ensure your frontend is running on `http://localhost:3000`
- Or update the `allow_origins` in `main.py` to match your frontend URL

## Testing

### Test All Endpoints

Run the included test script to verify all endpoints are working:

```bash
python test_endpoints.py
```

This will test:
- ✅ Health check
- ✅ File statistics
- ✅ File listing and filtering
- ✅ Document search and retrieval
- ✅ Chat with and without context
- ✅ Document preview

### Manual Testing

Use the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Or use the file manager interface:
- Open `file_manager.html` in your browser

## Development

### Project Structure

```
Backend/
├── main.py                   # FastAPI application with all endpoints
├── file_manager.html         # Web UI for file management
├── requirements.txt          # Python dependencies
├── env.example              # Environment variables template
├── .env                     # Your actual environment variables (git-ignored)
├── .gitignore              # Git ignore file
├── README.md               # This file
├── API_DOCUMENTATION.md    # Complete API reference
├── ENDPOINTS_SUMMARY.md    # Quick reference for new endpoints
└── test_endpoints.py       # Test script for all endpoints
```

### Adding New Features

The codebase is modular and easy to extend:

1. **Add new endpoints**: Define them in `main.py` using FastAPI decorators
2. **Modify RAG logic**: Update the `chat()` endpoint
3. **Customize file handling**: Modify file management endpoints
4. **Change LLM**: Replace `ChatGoogleGenerativeAI` with another LangChain-compatible LLM

## References

- [Pinecone Assistant Documentation](https://docs.pinecone.io/guides/assistant/upload-files)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)

## License

MIT

## Support

For issues or questions, please open an issue in the repository or contact the development team.

---

**Built with ❤️ for Femigrants**

