import { useState, useEffect } from 'react'
import { api } from '../api/client'
import type { SystemStats, MetricsResponse } from '../api/client'
import { BarChart3, Database, FileText, Zap, TrendingUp, Clock, Activity } from 'lucide-react'

export default function SystemStats() {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [timePeriod, setTimePeriod] = useState('1h')

  useEffect(() => {
    loadData()
  }, [timePeriod])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [statsData, metricsData] = await Promise.all([
        api.getStats(),
        api.getMetrics(timePeriod),
      ])
      setStats(statsData)
      setMetrics(metricsData)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load stats')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Activity className="h-8 w-8 text-primary-400 animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Loading statistics...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold text-white mb-2">System Statistics</h2>
        <p className="text-slate-400">Monitor your knowledge base performance</p>
      </div>

      <div className="flex justify-end mb-4">
        <select
          value={timePeriod}
          onChange={(e) => setTimePeriod(e.target.value)}
          className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
        >
          <option value="1h">Last Hour</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
        </select>
      </div>

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Database className="h-8 w-8 text-primary-400" />
              <div>
                <h3 className="text-sm font-medium text-slate-400">Vector Store</h3>
                <p className="text-2xl font-bold text-white">
                  {stats.document_processor.vector_store_size.toLocaleString()}
                </p>
                <p className="text-xs text-slate-500">Total Chunks</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <FileText className="h-8 w-8 text-purple-400" />
              <div>
                <h3 className="text-sm font-medium text-slate-400">Documents</h3>
                <p className="text-2xl font-bold text-white">
                  {stats.document_processor.total_documents_processed.toLocaleString()}
                </p>
                <p className="text-xs text-slate-500">Processed</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Zap className="h-8 w-8 text-yellow-400" />
              <div>
                <h3 className="text-sm font-medium text-slate-400">Embedding Model</h3>
                <p className="text-lg font-bold text-white truncate">
                  {stats.embedding.model || 'N/A'}
                </p>
                <p className="text-xs text-slate-500">Version: {stats.embedding.version}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <BarChart3 className="h-8 w-8 text-green-400" />
              <div>
                <h3 className="text-sm font-medium text-slate-400">Total Queries</h3>
                <p className="text-2xl font-bold text-white">
                  {metrics.total_queries.toLocaleString()}
                </p>
                <p className="text-xs text-slate-500">
                  {metrics.successful_queries} successful, {metrics.failed_queries} failed
                </p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <TrendingUp className="h-8 w-8 text-blue-400" />
              <div>
                <h3 className="text-sm font-medium text-slate-400">Cache Hit Rate</h3>
                <p className="text-2xl font-bold text-white">
                  {(metrics.cache_hit_rate * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-slate-500">Performance metric</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Clock className="h-8 w-8 text-pink-400" />
              <div>
                <h3 className="text-sm font-medium text-slate-400">Avg Latency</h3>
                <p className="text-2xl font-bold text-white">
                  {metrics.avg_latency_ms.toFixed(0)}ms
                </p>
                <p className="text-xs text-slate-500">P99: {metrics.p99_latency_ms.toFixed(0)}ms</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {stats?.cache && (
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Cache Statistics</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-slate-400">Cache Size:</span>
              <span className="ml-2 text-white font-semibold">{stats.cache.size}</span>
            </div>
            <div>
              <span className="text-sm text-slate-400">Hit Rate:</span>
              <span className="ml-2 text-white font-semibold">
                {(stats.cache.hit_rate * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">System Information</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">System:</span>
            <span className="text-white">{stats?.system || 'N/A'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Initialized:</span>
            <span className="text-white">
              {stats?.initialized ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Embedding Version:</span>
            <span className="text-white">
              {stats?.document_processor.embedding_version || 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

