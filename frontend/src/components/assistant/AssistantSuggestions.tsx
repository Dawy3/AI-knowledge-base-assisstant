'use client'

import { FileQuestion, BookOpen, Search, Lightbulb } from 'lucide-react'
import { useChatStore, useSettingsStore } from '@/lib/store'
import { streamChatMessage, sendChatMessage } from '@/lib/api'

interface AssistantSuggestionsProps {
  conversationId: string
}

const suggestions = [
  {
    icon: FileQuestion,
    text: 'What documents do I have?',
  },
  {
    icon: BookOpen,
    text: 'Summarize my knowledge base',
  },
  {
    icon: Search,
    text: 'Find information about...',
  },
  {
    icon: Lightbulb,
    text: 'What topics are covered?',
  },
]

export function AssistantSuggestions({ conversationId }: AssistantSuggestionsProps) {
  const { addMessage, updateMessage, setStreaming, getMessageHistory } = useChatStore()
  const { modelTier, useCache, temperature, maxTokens } = useSettingsStore()

  const handleSuggestionClick = async (text: string) => {
    setStreaming(true)

    // Add user message
    addMessage(conversationId, {
      role: 'user',
      content: text,
    })

    // Add placeholder for assistant message
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
    })

    try {
      const history = getMessageHistory(conversationId).slice(0, -1)
      let fullResponse = ''
      let responseData: {
        queryId?: string
        sources?: Array<{
          chunk_id: string
          document_id: string
          content_preview: string
          score: number
        }>
        cached?: boolean
        model?: string
        latencyMs?: number
      } = {}

      try {
        for await (const chunk of streamChatMessage({
          message: text,
          conversation_id: conversationId,
          history,
          stream: true,
          use_cache: useCache,
          ...(modelTier && { model_tier: modelTier }),
          temperature,
          max_tokens: maxTokens,
        })) {
          if (chunk.startsWith('{')) {
            try {
              const parsed = JSON.parse(chunk)
              if (parsed.query_id) {
                responseData = {
                  queryId: parsed.query_id,
                  sources: parsed.sources,
                  cached: parsed.cached,
                  model: parsed.model,
                  latencyMs: parsed.latency_ms,
                }
              }
            } catch {
              fullResponse += chunk
            }
          } else {
            fullResponse += chunk
          }
          updateMessage(conversationId, assistantMessageId, fullResponse)
        }
      } catch {
        const response = await sendChatMessage({
          message: text,
          conversation_id: conversationId,
          history,
          stream: false,
          use_cache: useCache,
          ...(modelTier && { model_tier: modelTier }),
          temperature,
          max_tokens: maxTokens,
        })

        fullResponse = response.message
        responseData = {
          queryId: response.query_id,
          sources: response.sources,
          cached: response.cached,
          model: response.model,
          latencyMs: response.latency_ms,
        }
        updateMessage(conversationId, assistantMessageId, fullResponse)
      }

      // Update final message with metadata
      const conversations = useChatStore.getState().conversations
      const conversation = conversations.find((c) => c.id === conversationId)
      if (conversation) {
        const updatedMessages = conversation.messages.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: fullResponse,
                queryId: responseData.queryId,
                sources: responseData.sources,
                cached: responseData.cached,
                model: responseData.model,
                latencyMs: responseData.latencyMs,
              }
            : msg
        )
        useChatStore.setState({
          conversations: conversations.map((c) =>
            c.id === conversationId ? { ...c, messages: updatedMessages } : c
          ),
        })
      }
    } catch (error) {
      console.error('Failed to send message:', error)
      updateMessage(
        conversationId,
        assistantMessageId,
        'Sorry, there was an error processing your request. Please try again.'
      )
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="grid grid-cols-2 gap-2 w-full max-w-[300px]">
      {suggestions.map((suggestion, index) => {
        const Icon = suggestion.icon
        return (
          <button
            key={index}
            onClick={() => handleSuggestionClick(suggestion.text)}
            className="flex items-center gap-2 p-3 text-xs text-left bg-muted hover:bg-accent rounded-lg transition-colors"
          >
            <Icon className="w-4 h-4 text-muted-foreground flex-shrink-0" />
            <span className="line-clamp-2">{suggestion.text}</span>
          </button>
        )
      })}
    </div>
  )
}
