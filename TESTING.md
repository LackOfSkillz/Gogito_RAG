# Testing Guide: Streaming & Cache Features

## Quick Start Testing

### 1. Start the API Server

```powershell
cd C:\Users\gmix\Documents\Cogito_RAG
python run_api.py
```

Expected: API starts on `http://127.0.0.1:8000`

### 2. Start the React UI (Optional)

```powershell
cd cogito-ui
npm install  # First time only
npm run dev
```

Expected: UI starts on `http://localhost:5173`

---

## Test 1: Streaming Answers (SSE)

### Using React UI

1. Open `http://localhost:5173`
2. Type a question: "Give me an executive summary of Survalent"
3. Press Send (or Enter)

**Success Criteria:**
- ✅ First token appears in < 1 second (warm API)
- ✅ Tokens stream in real-time (not all at once)
- ✅ Answer builds progressively word-by-word
- ✅ Sources appear at the bottom after completion
- ✅ Timings show in small gray text

### Using curl

```powershell
curl -X POST http://127.0.0.1:8000/api/ask_stream `
  -H "Content-Type: application/json" `
  -d '{"question":"Explain SCADA architecture","depth":10,"mode":"balanced"}' `
  -N
```

**Success Criteria:**
- ✅ See `data: {"type":"metadata"...}` first
- ✅ Then stream of `data: {"type":"token","content":"..."}`
- ✅ Finally `data: {"type":"done","timings":{...}}`
- ✅ Ends with `data: [DONE]`

---

## Test 2: Cancel Functionality

### Using React UI

1. Start typing a question: "Tell me everything about..."
2. Click "Send"
3. **Immediately click the "⏹ Cancel" button**

**Success Criteria:**
- ✅ Streaming stops within 200ms
- ✅ Message shows "[Cancelled by user]" or partial answer
- ✅ No errors in console
- ✅ Can immediately ask another question

---

## Test 3: Answer Cache

### Test Cache Hit

1. Ask: "Give me an executive summary of Survalent" (mode: balanced, depth: 10)
2. Wait for answer (should take ~3-5 seconds)
3. **Ask the exact same question again**

**Success Criteria:**
- ✅ Second answer returns in < 100ms
- ✅ Green badge shows "⚡ From cache"
- ✅ Timings show ~0s for retrieval and LLM
- ✅ Answer is identical to first

### Test Cache Miss (Different Parameters)

1. Ask same question but change:
   - Depth: 15 (instead of 10)
   - OR Mode: "thorough" (instead of "balanced")
2. Should be a **cache miss** (takes full time)

### View Cache Stats

```powershell
Invoke-RestMethod -Method GET http://127.0.0.1:8000/api/cache/stats
```

**Expected Output:**
```json
{
  "total_entries": 1,
  "total_hits": 1,
  "hit_rate": 1.0,
  "oldest_entry_age_s": 120,
  "newest_entry_age_s": 10,
  "size_mb": 0.01
}
```

**Or check in UI:**
- Cache Stats panel on left sidebar
- Shows entries, hits, hit rate, size

### Clear Cache

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/cache/clear
```

**Success Criteria:**
- ✅ Returns `{"ok": true, "message": "Cache cleared"}`
- ✅ Stats show 0 entries
- ✅ Previous "cached" questions now take full time

---

## Test 4: Performance Benchmarks

### First Token Latency (Warm)

1. Start API and wait 30 seconds for warmup
2. Ask any question
3. Measure time from click to first word appearing

**Target:** < 1 second

### Cancel Response Time

1. Start a long query
2. Click Cancel
3. Note how fast the streaming stops

**Target:** < 200ms

### Cache Hit Performance

1. Ask a question (full query)
2. Ask same question again (cache hit)
3. Measure response time

**Target:** < 100ms

---

## Test 5: Cache Correctness

### Test Invalidation (Future - after ingest changes)

1. Ask a question about a document
2. Note the answer
3. Update/change that document
4. Run `python cogito_loader.py --progress`
5. Ask the same question again

**Success Criteria:**
- ✅ Cache should be invalidated
- ✅ New answer reflects updated content
- ✅ NOT cached (takes full time)

### Test Normalization

These should all hit the same cache entry:

- "Give me an executive summary of Survalent"
- "give me an executive summary of survalent"
- "Give me an executive summary of Survalent."
- "Give me an executive summary of Survalent?"

**Success Criteria:**
- ✅ First query: full time
- ✅ Next 3: < 100ms cache hits

---

## Test 6: Error Handling

### Test Network Interruption

1. Start a streaming query
2. Stop the API server mid-stream

**Success Criteria:**
- ✅ UI shows error message
- ✅ No UI crash
- ✅ Can recover and retry after API restart

### Test Invalid Input

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8000/api/ask_stream `
  -ContentType "application/json" `
  -Body '{"question":"","depth":10}'
```

**Success Criteria:**
- ✅ Returns error or empty result gracefully
- ✅ No server crash

---

## Test 7: UI Features

### Settings Panel

- ✅ Model selector shows available models
- ✅ Mode selector: fast/balanced/thorough
- ✅ Sliders work: depth (4-20), temperature (0-1), max tokens (300-2000)
- ✅ Cache toggle enables/disables caching

### Cache Stats Panel

- ✅ Shows real-time stats
- ✅ Updates after each query
- ✅ "Clear Cache" button works
- ✅ Stats refresh after clearing

### Messages Display

- ✅ User messages on right (blue)
- ✅ Assistant messages on left (dark)
- ✅ Cache badge appears when cached
- ✅ Sources list appears when available
- ✅ Timings appear in gray monospace
- ✅ Auto-scrolls to newest message

---

## Troubleshooting

### Streaming not working

**Symptom:** Entire answer appears at once
**Fix:** Check that you're using `/api/ask_stream` not `/api/ask`

### Cache not hitting

**Symptom:** Same query takes full time
**Fix:** 
- Check cache stats show > 0 entries
- Verify "Enable Cache" is checked
- Try exact same question (case-insensitive, but punctuation may vary)

### Cancel not responsive

**Symptom:** Cancel button doesn't stop streaming
**Fix:**
- Check browser console for errors
- Verify AbortController support (modern browsers)
- Try refreshing page

### First token slow

**Symptom:** > 2 seconds for first token
**Fix:**
- Wait 30s for API warmup
- Check LM Studio is running and model loaded
- Try smaller model or lower max_tokens

---

## Performance Targets Summary

| Metric | Target | Status |
|--------|--------|--------|
| First token (warm) | < 1s | ✅ |
| Cancel response | < 200ms | ✅ |
| Cache hit | < 100ms | ✅ |
| Cache hit rate | > 30% | 📊 Monitor |
| Streaming smooth | No lag | ✅ |

---

## Next Steps

After successful testing:

1. Monitor cache hit rate over time
2. Adjust cache cleanup policy if needed (default: 30 days)
3. Consider adding cache prewarming for common queries
4. Prepare for Spaces feature (next implementation phase)

---

## Reporting Issues

If you encounter issues:

1. Check browser console (F12)
2. Check API server logs
3. Verify LM Studio is running
4. Test with non-streaming `/api/ask` first
5. Check cache stats for anomalies
