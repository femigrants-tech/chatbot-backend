# New Endpoints Summary

This document provides a quick reference for all the new document retrieval and management endpoints added to the Femigrants RAG Backend.

## 📊 New Endpoints Overview

### Document Retrieval Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/documents/search` | POST | Search documents with semantic ranking |
| `/documents/retrieve` | POST | Retrieve context snippets without chat |
| `/documents/preview/{file_id}` | GET | Preview document content |

### Enhanced File Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/files/bulk-delete` | POST | Delete multiple files at once |
| `/files/{file_id}/metadata` | PUT | Update file metadata (info) |
| `/files/by-status/{status}` | GET | Filter files by status |
| `/files/statistics` | GET | Get storage statistics |

---

## 🔍 Document Retrieval Endpoints

### 1. POST `/documents/search`

**Purpose:** Search for relevant documents with semantic ranking and scores.

**Use Case:** When you want to find the most relevant content from your knowledge base without generating a chat response.

**Request:**
```json
{
  "query": "What are the benefits of machine learning?",
  "filter_metadata": {"category": "research"},
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
      "metadata": {"category": "research"},
      "file_id": "abc123"
    }
  ],
  "total_results": 5
}
```

**When to Use:**
- Building a search interface
- Finding relevant documents
- Getting ranked results with scores
- Filtering by metadata

---

### 2. POST `/documents/retrieve`

**Purpose:** Retrieve context snippets without generating a response.

**Use Case:** Preview what context will be used for a chat query, or retrieve raw context for custom processing.

**Request:**
```json
{
  "query": "authentication process",
  "top_k": 3,
  "filter_metadata": {"type": "security"}
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
      "metadata": {"type": "security"}
    }
  ],
  "count": 3
}
```

**When to Use:**
- Debugging RAG queries
- Building custom UIs
- Testing context retrieval
- Preview before chat

---

### 3. GET `/documents/preview/{file_id}`

**Purpose:** Get a preview of document content along with metadata.

**Use Case:** Show users a snippet of a document before they download or reference it.

**Request:**
```
GET /documents/preview/abc123?max_length=300
```

**Response:**
```json
{
  "file_id": "abc123",
  "name": "document.pdf",
  "status": "Available",
  "size": 1024000,
  "metadata": {"category": "research"},
  "created_on": "2025-01-15T10:30:00Z",
  "preview": "This document discusses...",
  "signed_url": "https://..."
}
```

**When to Use:**
- File preview in UI
- Quick content inspection
- Verify file processing
- Show snippets to users

---

## 📁 Enhanced File Management Endpoints

### 4. POST `/files/bulk-delete`

**Purpose:** Delete multiple files in a single request.

**Use Case:** Clean up old files, remove failed uploads, or batch delete by criteria.

**Request:**
```json
{
  "file_ids": ["abc123", "def456", "ghi789"]
}
```

**Response:**
```json
{
  "message": "Deleted 3 out of 3 files",
  "results": {
    "success": ["abc123", "def456", "ghi789"],
    "failed": []
  }
}
```

**When to Use:**
- Bulk cleanup operations
- Delete failed uploads
- Remove outdated content
- Administrative tasks

---

### 5. PUT `/files/{file_id}/metadata`

**Purpose:** Update metadata for a file (informational endpoint).

**Use Case:** Check what metadata changes would be needed (Pinecone requires re-upload for actual changes).

**Request:**
```json
{
  "metadata": {
    "category": "updated",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "message": "Metadata update requested",
  "file_id": "abc123",
  "note": "To update metadata, please delete and re-upload the file",
  "current_metadata": {"category": "old"},
  "requested_metadata": {"category": "updated"}
}
```

**When to Use:**
- Check current metadata
- Plan metadata updates
- Document what needs to change
- UI metadata editor

---

### 6. GET `/files/by-status/{status}`

**Purpose:** Filter files by their processing status.

**Use Case:** Monitor processing, find failed uploads, or get only available files.

**Common Statuses:**
- `Available` - File is ready to use
- `Processing` - File is being processed
- `Failed` - Processing failed

**Request:**
```
GET /files/by-status/Processing
```

**Response:**
```json
{
  "status": "Processing",
  "files": [
    {
      "id": "abc123",
      "name": "large_doc.pdf",
      "status": "Processing",
      "percent_done": 0.75
    }
  ],
  "count": 1
}
```

**When to Use:**
- Monitor processing status
- Find failed uploads
- Wait for processing to complete
- Status dashboard

---

### 7. GET `/files/statistics`

**Purpose:** Get comprehensive statistics about your knowledge base.

**Use Case:** Dashboard, monitoring, storage planning, analytics.

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
  "metadata_keys_used": ["category", "author", "date"]
}
```

**When to Use:**
- Admin dashboard
- Storage monitoring
- Usage analytics
- Health checks

---

## 🎯 Quick Use Cases

### Use Case 1: Smart Search Interface
```javascript
// 1. Search for documents
const searchResults = await fetch('/documents/search', {
  method: 'POST',
  body: JSON.stringify({ query: userQuery, top_k: 10 })
});

// 2. Show previews
for (const result of results) {
  const preview = await fetch(`/documents/preview/${result.file_id}`);
  displayPreview(preview);
}
```

### Use Case 2: Monitoring Dashboard
```javascript
// Get statistics
const stats = await fetch('/files/statistics').then(r => r.json());

// Check processing status
const processing = await fetch('/files/by-status/Processing').then(r => r.json());

// Show health
displayDashboard(stats, processing);
```

### Use Case 3: Bulk Cleanup
```javascript
// Get failed files
const failed = await fetch('/files/by-status/Failed').then(r => r.json());
const failedIds = failed.files.map(f => f.id);

// Delete all failed files
await fetch('/files/bulk-delete', {
  method: 'POST',
  body: JSON.stringify({ file_ids: failedIds })
});
```

### Use Case 4: RAG Pipeline Debugging
```javascript
// 1. Retrieve context to see what's being found
const context = await fetch('/documents/retrieve', {
  method: 'POST',
  body: JSON.stringify({ query: "my question", top_k: 5 })
}).then(r => r.json());

console.log("Context found:", context);

// 2. If context looks good, proceed with chat
const response = await fetch('/chat', {
  method: 'POST',
  body: JSON.stringify({ message: "my question" })
});
```

---

## 🧪 Testing

Run the included test script to verify all endpoints:

```bash
python test_endpoints.py
```

This will test all endpoints and provide a comprehensive report.

---

## 📖 Full Documentation

For detailed API documentation with all parameters and examples, see:
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference
- **[README.md](README.md)** - Setup and getting started

---

## 🔗 Integration Examples

### Python
```python
import requests

# Search documents
response = requests.post('http://localhost:8000/documents/search', json={
    'query': 'machine learning',
    'top_k': 5
})
results = response.json()
```

### JavaScript/TypeScript
```typescript
// Get statistics
const stats = await fetch('http://localhost:8000/files/statistics')
  .then(res => res.json());

console.log(`Total files: ${stats.total_files}`);
```

### cURL
```bash
# Preview a document
curl "http://localhost:8000/documents/preview/abc123?max_length=200"

# Bulk delete
curl -X POST http://localhost:8000/files/bulk-delete \
  -H "Content-Type: application/json" \
  -d '{"file_ids": ["abc123", "def456"]}'
```

---

## ✨ Key Features

- ✅ **Semantic Search** - Find documents by meaning, not just keywords
- ✅ **Relevance Scoring** - Know which results are most relevant
- ✅ **Metadata Filtering** - Filter by category, author, date, etc.
- ✅ **Bulk Operations** - Manage multiple files efficiently
- ✅ **Status Monitoring** - Track processing and identify issues
- ✅ **Statistics & Analytics** - Understand your knowledge base
- ✅ **Document Previews** - See content before downloading
- ✅ **Context Retrieval** - Debug and optimize RAG queries

---

**Need Help?** Check the main [README.md](README.md) or [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for more details!

