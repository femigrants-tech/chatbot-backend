# RAG Pipeline Fix Summary

## Issues Fixed

### 1. **CRITICAL: Using Wrong Pinecone Method**
**Problem**: We were using `assistant.chat()` which generates a complete AI response, but we needed `assistant.context()` to retrieve ONLY the context chunks for RAG.

**Before** (WRONG):
```python
# This generates a chat response with its own AI answer
# We can't extract context chunks from this!
response = assistant.chat(
    messages=[{"role": "user", "content": query}],
    filter=filter_metadata
)
# response has: message, citations - but NO context attribute!
```

**After** (CORRECT):
```python
# This retrieves ONLY context snippets without generating a response
# Perfect for RAG - we get the chunks and pass them to Gemini
context_response = assistant.context(
    query=query,
    filter=filter_metadata
)
# context_response has: snippets[] with content, score, reference!
```

**This was the root cause!** The chat() method doesn't return retrievable context - it generates its own AI response. The context() method is specifically designed for RAG use cases.

### 2. **Correct Context Extraction from Snippets**
**Problem**: The context() method returns snippets in a specific format that we weren't parsing correctly.

**Fixed**:
- Parse `snippets` array from context response
- Extract `content`, `score`, and `reference` from each snippet
- Get file information (name, id, pages) from reference
- Properly format for Gemini consumption

**Context Response Structure**:
```python
{
  'id': '...',
  'snippets': [
    {
      'type': 'text',
      'content': 'The actual text content...',  # This is what we need!
      'score': 0.93,  # Relevance score
      'reference': {
        'type': 'pdf',
        'pages': [1],
        'file': {
          'name': 'filename.pdf',
          'id': '...',
          'status': 'Available'
        }
      }
    }
  ]
}
```

### 3. **Poor LLM Prompting**
**Problem**: The system prompt wasn't emphasizing the importance of using the provided context.

**Fixed**:
- Improved system prompt with explicit instructions to use context
- Added source numbering and relevance scores to context
- Better formatting of context chunks
- Clear warning when no context is available

### 4. **Limited Context Retrieval**
**Problem**: Only retrieving 5 context items by default, which might miss important information.

**Fixed**:
- Increased default `top_k` to 10 for chat endpoint
- Made `top_k` configurable as a parameter
- Added context length tracking in logs

### 5. **Gemini Model Version**
**Problem**: Using outdated `gemini-pro` model name.

**Fixed**: Updated to `gemini-2.5-flash`

### 6. **Files List Response Format**
**Problem**: Assumed `list_files()` returned an object with a `files` attribute, but it actually returns a list directly.

**Fixed**: 
- Updated to handle list response correctly
- Simplified code by removing unnecessary checks
- All file-related endpoints now work properly

## Changes Made

### Modified Files:

1. **`main.py`**:
   - `get_context_from_pinecone()`: Complete rewrite with proper message format and extensive debugging
   - `/chat` endpoint: Improved prompting and increased context retrieval
   - `/documents/search`: Uses corrected context retrieval
   - `/documents/preview`: Uses proper message format
   - `/files` endpoint: Added debug logging and multiple response format handling
   - `/files/upload`: Added debug logging and automatic filename preservation

2. **New Test Files**:
   - `test_rag_pipeline.py`: Comprehensive RAG diagnostic tool
   - `debug_test.py`: Simple upload/list test

## How to Test

### Step 1: Restart Your Server

Make sure to restart your FastAPI server to load the changes:

```bash
# Stop the current server (Ctrl+C)
# Then restart it:
python3 main.py
# or
uvicorn main:app --reload
```

### Step 2: Run the Diagnostic Test

```bash
python3 test_rag_pipeline.py
```

This will:
1. Check server health
2. List files in your knowledge base
3. Test document search
4. Test context retrieval
5. Test full chat with RAG
6. Provide a diagnostic summary

### Step 3: Watch Server Logs

The server will now print detailed debugging information:

```
============================================================
Querying Pinecone with: What is in the documents?
Filter: None
============================================================
Response type: <class 'ChatResponse'>
Response attributes: [...]
Found 10 context items
...
```

This will show you:
- What Pinecone is returning
- How many context items were found
- What text is being extracted
- What's being sent to Gemini
- What response Gemini generates

## Expected Behavior

### ✅ Working RAG Pipeline:

1. **Query arrives** → Server logs the question
2. **Pinecone query** → Shows query and filter
3. **Context found** → Shows N context items retrieved
4. **Context extracted** → Shows text previews from each chunk
5. **Gemini called** → Shows model name
6. **Response generated** → Shows response preview

### ❌ If Context Not Retrieved:

Possible causes:
1. **Files not indexed**: Check Pinecone console to verify files are "Available" status
2. **Embeddings mismatch**: Pinecone Assistant uses its own embeddings
3. **Query format wrong**: Fixed in this update
4. **No matching content**: Your query might not match any document content

## Debugging Tips

### 1. Check File Status

```bash
curl http://localhost:8000/files/statistics
```

Look for:
- `total_files > 0`
- `status_breakdown` showing "Available" files

### 2. Test Direct Search

```bash
curl -X POST http://localhost:8000/documents/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "top_k": 5}'
```

This shows raw context retrieval without LLM generation.

### 3. Check Context Retrieval

```bash
curl -X POST http://localhost:8000/documents/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "top_k": 5}'
```

### 4. Monitor Server Terminal

Watch for:
- "Found X context items" - should be > 0
- "Text preview:" - should show actual document content
- "Returning X context items" - should match what was found

## Common Issues

### Issue: "No context found in response!"

**Possible causes**:
1. Files are still processing (status != "Available")
2. Pinecone Assistant API changed
3. Query doesn't match any content

**Solution**: 
- Wait for files to finish processing
- Check file status with `/files` endpoint
- Try a more general query

### Issue: "Context retrieved but LLM gives wrong answer"

**Possible causes**:
1. Retrieved context isn't relevant
2. Context is too fragmented
3. LLM isn't following instructions

**Solution**:
- Check what context is actually retrieved (use `/documents/search`)
- Increase `top_k` to get more context
- Improve document chunking when uploading

### Issue: Files uploaded but not appearing

**Possible causes**:
1. Pinecone API response format mismatch
2. Files still processing

**Solution**:
- Check server logs for `list_files` response
- Wait a few seconds and try again
- Files may be indexed but take time to appear in list

## Next Steps

1. **Upload test documents** if you haven't already
2. **Run the diagnostic test** to verify the pipeline
3. **Check server logs** for detailed debugging info
4. **Test with real queries** related to your documents
5. **Monitor context retrieval** to ensure relevant chunks are being found

## Additional Notes

- Context retrieval is now set to `top_k=10` by default for better coverage
- All endpoints now have extensive debug logging
- The improved prompt explicitly tells Gemini to use the provided context
- Original filename is automatically preserved in metadata

