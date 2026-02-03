'use client'

import { Settings, Zap, Thermometer, Hash, Key, User, Trash2 } from 'lucide-react'
import { useSettingsStore, useChatStore } from '@/lib/store'

export function SettingsView() {
  const {
    apiKey,
    userId,
    modelTier,
    temperature,
    maxTokens,
    setApiKey,
    setUserId,
    setModelTier,
    setTemperature,
    setMaxTokens,
  } = useSettingsStore()

  const { clearConversations, conversations } = useChatStore()

  const handleClearHistory = () => {
    if (confirm('Are you sure you want to delete all conversations? This cannot be undone.')) {
      clearConversations()
    }
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Settings
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure your AI assistant preferences
          </p>
        </div>

        {/* Authentication */}
        <section className="rounded-xl border border-border p-6 space-y-4">
          <h2 className="text-lg font-semibold">Authentication</h2>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Key className="w-4 h-4 text-muted-foreground" />
              API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key (optional)"
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20"
            />
            <p className="text-xs text-muted-foreground">
              Optional. Required in production environments.
            </p>
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <User className="w-4 h-4 text-muted-foreground" />
              User ID
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter your user ID (optional)"
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20"
            />
            <p className="text-xs text-muted-foreground">
              Used to track conversations and personalize responses.
            </p>
          </div>
        </section>

        {/* Model Settings */}
        <section className="rounded-xl border border-border p-6 space-y-4">
          <h2 className="text-lg font-semibold">Model Configuration</h2>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Zap className="w-4 h-4 text-muted-foreground" />
              Model Tier
            </label>
            <select
              value={modelTier}
              onChange={(e) => setModelTier(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20"
            >
              <option value="tier_1">Tier 1 (Fast, Cost-effective)</option>
              <option value="tier_2">Tier 2 (Balanced)</option>
              <option value="tier_3">Tier 3 (High Quality)</option>
              <option value="auto">Auto (Query-based routing)</option>
            </select>
            <p className="text-xs text-muted-foreground">
              Choose the model tier based on your quality vs. cost preferences.
            </p>
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Thermometer className="w-4 h-4 text-muted-foreground" />
              Temperature: {temperature}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Precise (0)</span>
              <span>Creative (1)</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Hash className="w-4 h-4 text-muted-foreground" />
              Max Tokens
            </label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              min={128}
              max={4096}
              className="w-full px-3 py-2 rounded-lg border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20"
            />
            <p className="text-xs text-muted-foreground">
              Maximum number of tokens in the response (128-4096).
            </p>
          </div>
        </section>

        {/* Data Management */}
        <section className="rounded-xl border border-border p-6 space-y-4">
          <h2 className="text-lg font-semibold">Data Management</h2>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Trash2 className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="text-sm font-medium">Clear Conversation History</p>
                <p className="text-xs text-muted-foreground">
                  Delete all {conversations.length} conversation{conversations.length !== 1 ? 's' : ''} from local storage
                </p>
              </div>
            </div>
            <button
              onClick={handleClearHistory}
              disabled={conversations.length === 0}
              className="px-4 py-2 rounded-lg border border-border hover:bg-red-50 hover:border-red-200 hover:text-red-600 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Clear All
            </button>
          </div>
        </section>

        {/* About */}
        <section className="rounded-xl border border-border p-6">
          <h2 className="text-lg font-semibold mb-3">About</h2>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>AI Knowledge Assistant v1.0.0</p>
            <p>A RAG-powered knowledge base with semantic search, intelligent caching, and multi-tier model routing.</p>
          </div>
        </section>
      </div>
    </div>
  )
}
