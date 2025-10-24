import { useState, useRef, useEffect } from 'react'
import './App.css'

interface Source {
  source: string
  page: number | null
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  cached?: boolean
  timings?: any
  mode?: string
}

interface CacheStats {
  total_entries: number
  total_hits: number
  hit_rate: number
  size_mb: number
}

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [depth, setDepth] = useState(10)
  const [temperature, setTemperature] = useState(0.3)
  const [maxTokens, setMaxTokens] = useState(900)
  const [mode, setMode] = useState<'fast' | 'balanced' | 'thorough'>('balanced')
  const [useCache, setUseCache] = useState(true)
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  
  const abortControllerRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Load models on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/models`)
      .then(res => res.json())
      .then(data => {
        setModels(data.models || [])
        if (data.models && data.models.length > 0) {
          setSelectedModel(data.models[0])
        }
      })
      .catch(err => console.error('Failed to load models:', err))
  }, [])

  // Load cache stats periodically
  useEffect(() => {
    const loadCacheStats = () => {
      fetch(`${API_BASE}/api/cache/stats`)
        .then(res => res.json())
        .then(data => setCacheStats(data))
        .catch(err => console.error('Failed to load cache stats:', err))
    }
    
    loadCacheStats()
    const interval = setInterval(loadCacheStats, 10000) // Every 10s
    return () => clearInterval(interval)
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isStreaming) return

    const userMessage: Message = { role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsStreaming(true)

    // Create new abort controller
    abortControllerRef.current = new AbortController()

    try {
      const response = await fetch(`${API_BASE}/api/ask_stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage.content,
          depth,
          temperature,
          max_tokens: maxTokens,
          model: selectedModel || null,
          mode,
        }),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader available')

      const decoder = new TextDecoder()
      let buffer = ''
      let assistantContent = ''
      let metadata: any = null

      const assistantMessage: Message = { role: 'assistant', content: '' }
      setMessages(prev => [...prev, assistantMessage])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue
          
          const data = line.slice(6).trim()
          if (data === '[DONE]') {
            setIsStreaming(false)
            break
          }

          try {
            const chunk = JSON.parse(data)

            if (chunk.type === 'metadata') {
              metadata = chunk
            } else if (chunk.type === 'token') {
              assistantContent += chunk.content
              setMessages(prev => {
                const newMessages = [...prev]
                const lastMsg = newMessages[newMessages.length - 1]
                if (lastMsg.role === 'assistant') {
                  lastMsg.content = assistantContent
                }
                return newMessages
              })
            } else if (chunk.type === 'done') {
              setMessages(prev => {
                const newMessages = [...prev]
                const lastMsg = newMessages[newMessages.length - 1]
                if (lastMsg.role === 'assistant') {
                  lastMsg.timings = chunk.timings
                  if (metadata) {
                    lastMsg.sources = metadata.sources
                    lastMsg.mode = metadata.mode
                  }
                }
                return newMessages
              })
            } else if (chunk.type === 'error') {
              throw new Error(chunk.message)
            }
          } catch (parseErr) {
            console.error('Parse error:', parseErr)
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setMessages(prev => {
          const newMessages = [...prev]
          const lastMsg = newMessages[newMessages.length - 1]
          if (lastMsg.role === 'assistant' && !lastMsg.content.trim()) {
            newMessages.pop() // Remove empty assistant message
          } else if (lastMsg.role === 'assistant') {
            lastMsg.content += '\n\n[Cancelled by user]'
          }
          return newMessages
        })
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${err.message}`,
        }])
      }
    } finally {
      setIsStreaming(false)
      abortControllerRef.current = null
    }
  }

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
  }

  const clearCache = async () => {
    try {
      await fetch(`${API_BASE}/api/cache/clear`, { method: 'POST' })
      alert('Cache cleared successfully')
      // Reload stats
      const res = await fetch(`${API_BASE}/api/cache/stats`)
      setCacheStats(await res.json())
    } catch (err) {
      alert('Failed to clear cache')
    }
  }

  return (
    <div className="app">
      <div className="sidebar">
        <h2>⚡ Cogito RAG</h2>
        
        <div className="settings">
          <h3>Settings</h3>
          
          <label>
            Model
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
              {models.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>

          <label>
            Mode
            <select value={mode} onChange={e => setMode(e.target.value as any)}>
              <option value="fast">Fast (4-6 bullets)</option>
              <option value="balanced">Balanced (6-8 bullets)</option>
              <option value="thorough">Thorough (8-12 bullets)</option>
            </select>
          </label>

          <label>
            Answer Depth: {depth}
            <input
              type="range"
              min="4"
              max="20"
              value={depth}
              onChange={e => setDepth(parseInt(e.target.value))}
            />
          </label>

          <label>
            Temperature: {temperature}
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
            />
          </label>

          <label>
            Max Tokens: {maxTokens}
            <input
              type="range"
              min="300"
              max="2000"
              step="100"
              value={maxTokens}
              onChange={e => setMaxTokens(parseInt(e.target.value))}
            />
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={useCache}
              onChange={e => setUseCache(e.target.checked)}
            />
            Enable Cache
          </label>
        </div>

        {cacheStats && (
          <div className="cache-stats">
            <h3>Cache Stats</h3>
            <div className="stat-row">
              <span>Entries:</span>
              <span>{cacheStats.total_entries}</span>
            </div>
            <div className="stat-row">
              <span>Hits:</span>
              <span>{cacheStats.total_hits}</span>
            </div>
            <div className="stat-row">
              <span>Hit Rate:</span>
              <span>{(cacheStats.hit_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="stat-row">
              <span>Size:</span>
              <span>{cacheStats.size_mb} MB</span>
            </div>
            <button onClick={clearCache} className="clear-cache-btn">
              Clear Cache
            </button>
          </div>
        )}
      </div>

      <div className="main">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <h1>Welcome to Cogito</h1>
              <p>Local, privacy-first RAG assistant for Ocala Electric</p>
              <p>Ask questions about your documents...</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.content || <em>Generating...</em>}
              </div>
              
              {msg.cached && (
                <div className="cache-badge">⚡ From cache</div>
              )}

              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <strong>Sources:</strong>
                  <ul>
                    {msg.sources.slice(0, 5).map((src, i) => (
                      <li key={i}>
                        {src.source}
                        {src.page !== null && ` (page ${src.page})`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {msg.timings && (
                <div className="timings">
                  Retrieval: {msg.timings.doc_fetch_s}s | 
                  LLM: {msg.timings.llm_time_s}s | 
                  Total: {msg.timings.total_s}s
                  {msg.mode && ` | Mode: ${msg.mode}`}
                </div>
              )}
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isStreaming}
          />
          {isStreaming ? (
            <button type="button" onClick={handleCancel} className="cancel-btn">
              ⏹ Cancel
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()}>
              Send
            </button>
          )}
        </form>
      </div>
    </div>
  )
}

export default App
