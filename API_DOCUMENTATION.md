# API Documentation

Complete API reference for the Femigrants RAG Backend.

Base URL: `http://localhost:8000`

## Table of Contents
- [Chat Endpoints](#chat-endpoints)
- [File Management](#file-management)
- [Document Retrieval](#document-retrieval)
- [Statistics & Monitoring](#statistics--monitoring)

---

## Chat Endpoints

### POST `/chat`
Chat with the RAG system using your knowledge base.

**Request Body:**
```json
{
  "message": "What is machine learning?",
  "chat_context": [
    {"role": "user", "content": "Tell me about AI"},
    {"role": "assistant", "content": "AI is..."}
  ]
}
```

**Response:**
```json
{
  "response": "Machine learning is a subset of AI...",
  "context_used": [
    {
      "text": "Relevant context from documents...",
      "score": 0.87,
      "metadata": {"source": "file1.pdf"}
    }
  ]
}
```

---

## File Management

### POST `/files/upload`
Upload a file to the knowledge base.

**Request:**
- Content-Type: `multipart/form-data`
- `file`: File to upload
- `metadata`: Optional JSON string with metadata

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/files/upload" \
  -F "file=@document.pdf" \
  -F 'metadata={"category": "documentation", "author": "admin"}'
```

**Response:**
```json
{
  "message": "File uploaded successfully",
  "file_id": "abc123...",
  "filename": "document.pdf",
  "status": "Processing"
}
```

---

### GET `/files`
List all files in the knowledge base.

**Query Parameters:**
- `filter_metadata` (optional): JSON string to filter by metadata

**Example:**
```bash
GET /files?filter_metadata={"category":"documentation"}
```

**Response:**
```json
{
  "files": [
    {
      "id": "abc123",
      "name": "document.pdf",
      "status": "Available",
      "size": 1024000,
      "metadata": {"category": "documentation"},
      "created_on": "2025-01-15T10:30:00Z",
      "percent_done": 1.0
    }
  ],
  "total": 1
}
```

---

### GET `/files/{file_id}`
Get details about a specific file.

**Query Parameters:**
- `include_url` (optional): Boolean, set to `true` to get download URL

**Response:**
```json
{
  "id": "abc123",
  "name": "document.pdf",
  "status": "Available",
  "size": 1024000,
  "metadata": {},
  "created_on": "2025-01-15T10:30:00Z",
  "updated_on": "2025-01-15T10:35:00Z",
  "percent_done": 1.0,
  "signed_url": "https://storage.googleapis.com/...",
  "error_message": null
}
```

---

### DELETE `/files/{file_id}`
Delete a file from the knowledge base.

⚠️ **Warning:** This operation cannot be undone.

**Response:**
```json
{
  "message": "File deleted successfully",
  "file_id": "abc123"
}
```

---

### PUT `/files/{file_id}/metadata`
Update metadata for a file (informational endpoint).

**Request Body:**
```json
{
  "metadata": {
    "category": "updated_category",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "message": "Metadata update requested",
  "file_id": "abc123",
  "note": "To update metadata, please delete and re-upload the file with new metadata",
  "current_metadata": {"category": "old_category"},
  "requested_metadata": {"category": "updated_category"}
}
```

---

### POST `/files/bulk-delete`
Delete multiple files at once.

⚠️ **Warning:** This operation cannot be undone.

**Request Body:**
```json
{
  "file_ids": ["abc123", "def456", "ghi789"]
}
```

**Response:**
```json
{
  "message": "Deleted 2 out of 3 files",
  "results": {
    "success": ["abc123", "def456"],
    "failed": [
      {
        "file_id": "ghi789",
        "error": "File not found"
      }
    ]
  }
}
```

---

### GET `/files/by-status/{status}`
Get all files with a specific status.

**Common statuses:** `Available`, `Processing`, `Failed`

**Example:**
```bash
GET /files/by-status/Available
```

**Response:**
```json
{
  "status": "Available",
  "files": [
    {
      "id": "abc123",
      "name": "document.pdf",
      "status": "Available",
      "size": 1024000,
      "metadata": {},
      "created_on": "2025-01-15T10:30:00Z",
      "percent_done": 1.0
    }
  ],
  "count": 1
}
```

---

## Document Retrieval

### POST `/documents/search`
Search for relevant documents based on a query.

**Request Body:**
```json
{
  "query": "What are the benefits of machine learning?",
  "filter_metadata": {"category": "documentation"},
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "What are the benefits of machine learning?",
  "results": [
    {
      "rank": 1,
      "text": "Machine learning provides automated insights...",
      "score": 0.92,
      "metadata": {"category": "documentation", "source": "ml_guide.pdf"},
      "file_id": "abc123"
    }
  ],
  "total_results": 5
}
```

---

### POST `/documents/retrieve`
Retrieve context snippets without generating a chat response.

**Request Body:**
```json
{
  "query": "authentication process",
  "top_k": 3,
  "filter_metadata": {"category": "security"}
}
```

**Response:**
```json
{
  "query": "authentication process",
  "context_snippets": [
    {
      "text": "The authentication process involves...",
      "score": 0.88,
      "metadata": {"category": "security"}
    }
  ],
  "count": 3
}
```

---

### GET `/documents/preview/{file_id}`
Get a preview of document content.

**Query Parameters:**
- `max_length` (optional): Maximum length of preview text (default: 500)

**Example:**
```bash
GET /documents/preview/abc123?max_length=200
```

**Response:**
```json
{
  "file_id": "abc123",
  "name": "document.pdf",
  "status": "Available",
  "size": 1024000,
  "metadata": {},
  "created_on": "2025-01-15T10:30:00Z",
  "preview": "This document contains information about...",
  "signed_url": "https://storage.googleapis.com/..."
}
```

---

## Statistics & Monitoring

### GET `/files/statistics`
Get statistics about all uploaded files.

**Response:**
```json
{
  "total_files": 15,
  "total_size_bytes": 52428800,
  "total_size_mb": 50.0,
  "status_breakdown": {
    "Available": 12,
    "Processing": 2,
    "Failed": 1
  },
  "metadata_keys_used": ["category", "author", "date", "priority"]
}
```

---

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "assistant_name": "femigrants-assistant",
  "index_name": "femigrants-rag-chatbot",
  "pinecone": "connected",
  "gemini": "configured"
}
```

---

## Error Responses

All endpoints return standard HTTP error codes:

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Rate Limits

Rate limits depend on your Pinecone and Google Gemini plan. Check your respective dashboards for current limits.

---

## Interactive Documentation

When the server is running, access interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Example Usage Scenarios

### Scenario 1: Upload and Search
```python
import requests

# 1. Upload a file
files = {'file': open('document.pdf', 'rb')}
metadata = '{"category": "research", "author": "John Doe"}'
response = requests.post(
    'http://localhost:8000/files/upload',
    files=files,
    data={'metadata': metadata}
)
file_id = response.json()['file_id']

# 2. Wait for processing (check status)
status = requests.get(f'http://localhost:8000/files/{file_id}')
print(status.json()['status'])

# 3. Search documents
search_response = requests.post(
    'http://localhost:8000/documents/search',
    json={
        'query': 'What is the main topic?',
        'top_k': 3
    }
)
print(search_response.json()['results'])
```

### Scenario 2: Chat with Context
```javascript
// Chat with conversation history
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Explain the methodology",
    chat_context: [
      { role: "user", content: "What is this research about?" },
      { role: "assistant", content: "This research focuses on..." }
    ]
  })
});

const data = await response.json();
console.log(data.response);
console.log(data.context_used);
```

### Scenario 3: Bulk Operations
```python
import requests

# Get statistics
stats = requests.get('http://localhost:8000/files/statistics')
print(f"Total files: {stats.json()['total_files']}")

# Get failed files
failed = requests.get('http://localhost:8000/files/by-status/Failed')
failed_ids = [f['id'] for f in failed.json()['files']]

# Delete all failed files
if failed_ids:
    delete_response = requests.post(
        'http://localhost:8000/files/bulk-delete',
        json={'file_ids': failed_ids}
    )
    print(delete_response.json()['message'])
```

---

## Best Practices

1. **File Upload**
   - Always include meaningful metadata to make searching easier
   - Check file status before trying to search its content
   - Use metadata filtering to organize your knowledge base

2. **Search & Retrieval**
   - Start with broad queries, then narrow down with filters
   - Use `top_k` to limit results and improve performance
   - Check relevance scores to gauge result quality

3. **Chat Context**
   - Limit chat_context to recent messages (last 5-10 exchanges)
   - Always use the parameter name `chat_context`
   - Include both user and assistant messages for best results

4. **Error Handling**
   - Always check response status codes
   - Implement retry logic for 503 errors
   - Validate metadata JSON before uploading

5. **Performance**
   - Use bulk operations when dealing with multiple files
   - Cache file statistics if checking frequently
   - Use metadata filters to reduce search scope

---

## Support

For issues or questions:
- Check the main README.md for setup instructions
- Review error messages in API responses
- Check Pinecone and Gemini service status
- Verify environment variables are correctly set

