import { useState } from 'react'
import { api, QueryResponse, Source } from '../api/client'
import { Send, Loader2, Sparkles, Clock, Database } from 'lucide-react'

export default function QueryInterface() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [topK, setTopK] = useState(5)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const result = await api.query({
        query: query.trim(),
        top_k: topK,
        use_cache: true,
        return_sources: true,
      })
      setResponse(result)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold text-white mb-2">Ask Your Knowledge Base</h2>
        <p className="text-slate-400">Get intelligent answers from your documents</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question here..."
            className="w-full px-4 py-3 pr-12 bg-slate-800/50 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
            rows={4}
            disabled={loading}
          />
          <div className="absolute bottom-3 right-3 flex items-center space-x-2">
            <label className="text-sm text-slate-400">Top K:</label>
            <input
              type="number"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
              className="w-16 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-white text-sm"
              disabled={loading}
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="w-full bg-gradient-to-r from-primary-600 to-purple-600 hover:from-primary-700 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Send className="h-5 w-5" />
              <span>Ask Question</span>
            </>
          )}
        </button>
      </form>

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {response && (
        <div className="space-y-6 mt-8">
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-start space-x-3">
              <Sparkles className="h-6 w-6 text-primary-400 mt-1 flex-shrink-0" />
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-white mb-2">Answer</h3>
                <p className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                  {response.answer}
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-slate-400 mb-1">
                <Clock className="h-4 w-4" />
                <span className="text-sm">Latency</span>
              </div>
              <p className="text-2xl font-bold text-white">{response.latency_ms.toFixed(0)}ms</p>
            </div>

            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-slate-400 mb-1">
                <Database className="h-4 w-4" />
                <span className="text-sm">Sources</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {response.sources?.length || 0}
              </p>
            </div>

            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-slate-400 mb-1">
                <span className="text-sm">Cache</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {response.cache_hit ? 'HIT' : 'MISS'}
              </p>
            </div>
          </div>

          {response.retrieval_stats && (
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-slate-400 mb-2">Retrieval Stats</h4>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-slate-500">Variants:</span>
                  <span className="ml-2 text-white">{response.retrieval_stats.num_variants}</span>
                </div>
                <div>
                  <span className="text-slate-500">Retrieved:</span>
                  <span className="ml-2 text-white">{response.retrieval_stats.total_retrieved}</span>
                </div>
                <div>
                  <span className="text-slate-500">Final:</span>
                  <span className="ml-2 text-white">{response.retrieval_stats.final_count}</span>
                </div>
              </div>
            </div>
          )}

          {response.sources && response.sources.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Source Documents</h3>
              <div className="space-y-3">
                {response.sources.map((source: Source, index: number) => (
                  <div
                    key={source.chunk_id}
                    className="bg-slate-900/50 border border-slate-600 rounded-lg p-4"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-sm font-semibold text-primary-400">
                        Source {index + 1}
                      </span>
                      <span className="text-xs text-slate-500">
                        Score: {source.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400 space-y-1">
                      <p>Chunk ID: {source.chunk_id}</p>
                      {source.metadata && Object.keys(source.metadata).length > 0 && (
                        <div className="mt-2">
                          <p className="text-slate-500 mb-1">Metadata:</p>
                          <pre className="text-xs bg-slate-800/50 p-2 rounded overflow-x-auto">
                            {JSON.stringify(source.metadata, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

