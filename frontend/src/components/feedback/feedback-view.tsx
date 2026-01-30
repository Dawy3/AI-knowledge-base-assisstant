'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BarChart3, ThumbsUp, ThumbsDown, Star, TrendingUp, RefreshCw, Loader2 } from 'lucide-react'
import { getFeedbackStats, getHealthReady, type FeedbackStats, type HealthStatus } from '@/lib/api'
import { clsx } from 'clsx'

export function FeedbackView() {
  const [days, setDays] = useState(7)

  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ['feedback-stats', days],
    queryFn: () => getFeedbackStats(days),
  })

  const { data: health, isLoading: healthLoading, refetch: refetchHealth } = useQuery({
    queryKey: ['health-ready'],
    queryFn: getHealthReady,
    refetchInterval: 30000, // Poll every 30 seconds
  })

  const handleRefresh = () => {
    refetchStats()
    refetchHealth()
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Analytics & Health</h1>
            <p className="text-sm text-muted-foreground">
              Monitor feedback and system status
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="px-3 py-1.5 rounded-lg border border-border bg-background text-sm"
            >
              <option value={7}>Last 7 days</option>
              <option value={14}>Last 14 days</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 90 days</option>
            </select>
            <button
              onClick={handleRefresh}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* System Health */}
        <div className="rounded-xl border border-border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            System Health
          </h2>
          {healthLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : health ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <HealthCard label="API" status={health.status} />
              <HealthCard label="Database" status={health.database} />
              <HealthCard label="Vector Store" status={health.vector_store} />
              <HealthCard label="Cache" status={health.cache} />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              Unable to fetch health status
            </p>
          )}
        </div>

        {/* Feedback Stats */}
        <div className="rounded-xl border border-border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Feedback Statistics
          </h2>
          {statsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : stats ? (
            <div className="space-y-6">
              {/* Key metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  label="Total Feedback"
                  value={stats.total_feedback}
                  icon={BarChart3}
                />
                <MetricCard
                  label="Average Rating"
                  value={stats.average_rating?.toFixed(1) || 'N/A'}
                  icon={Star}
                  suffix="/5"
                />
                <MetricCard
                  label="Thumbs Up"
                  value={stats.thumbs_up_count}
                  icon={ThumbsUp}
                  iconColor="text-green-600"
                />
                <MetricCard
                  label="Thumbs Down"
                  value={stats.thumbs_down_count}
                  icon={ThumbsDown}
                  iconColor="text-red-600"
                />
              </div>

              {/* Rating distribution */}
              {stats.rating_distribution && Object.keys(stats.rating_distribution).length > 0 && (
                <div>
                  <h3 className="text-sm font-medium mb-3">Rating Distribution</h3>
                  <div className="space-y-2">
                    {[5, 4, 3, 2, 1].map((rating) => {
                      const count = stats.rating_distribution?.[rating] || 0
                      const total = stats.total_feedback || 1
                      const percentage = (count / total) * 100
                      return (
                        <div key={rating} className="flex items-center gap-3">
                          <span className="text-sm w-8 text-right">{rating}</span>
                          <Star className="w-4 h-4 text-muted-foreground" />
                          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-foreground rounded-full transition-all"
                              style={{ width: `${percentage}%` }}
                            />
                          </div>
                          <span className="text-sm text-muted-foreground w-12">
                            {count}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {stats.total_feedback === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No feedback collected yet. Start chatting to collect feedback!
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              Unable to fetch feedback statistics
            </p>
          )}
        </div>

        {/* Tips */}
        <div className="rounded-xl border border-border p-6 bg-muted/30">
          <h2 className="text-lg font-semibold mb-3">Tips for Improving Quality</h2>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="text-foreground">1.</span>
              Upload more relevant documents to expand your knowledge base
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground">2.</span>
              Review low-rated responses to identify areas for improvement
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground">3.</span>
              Use feedback data to fine-tune your model or adjust chunking strategies
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground">4.</span>
              Monitor cache hit rates to optimize semantic caching thresholds
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

function HealthCard({ label, status }: { label: string; status?: string }) {
  const isHealthy = status === 'healthy' || status === 'ok'
  return (
    <div className="p-4 rounded-lg bg-muted/50">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <div className="flex items-center gap-2">
        <div
          className={clsx(
            'w-2 h-2 rounded-full',
            isHealthy ? 'bg-green-500' : status ? 'bg-red-500' : 'bg-muted-foreground'
          )}
        />
        <span className="text-sm font-medium capitalize">
          {status || 'Unknown'}
        </span>
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  icon: Icon,
  suffix,
  iconColor,
}: {
  label: string
  value: number | string
  icon: React.ComponentType<{ className?: string }>
  suffix?: string
  iconColor?: string
}) {
  return (
    <div className="p-4 rounded-lg bg-muted/50">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={clsx('w-4 h-4', iconColor || 'text-muted-foreground')} />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className="text-2xl font-semibold">
        {value}
        {suffix && <span className="text-sm text-muted-foreground">{suffix}</span>}
      </p>
    </div>
  )
}
