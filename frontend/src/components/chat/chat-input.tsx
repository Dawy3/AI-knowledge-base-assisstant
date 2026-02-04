'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { useChatStore, useSettingsStore } from '@/lib/store'
import { sendChatMessage } from '@/lib/api'

interface ChatInputProps {
  conversationId: string
}

export function ChatInput({ conversationId }: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { addMessage, updateMessage, isStreaming, setStreaming, getMessageHistory } = useChatStore()
  const { modelTier, temperature, maxTokens } = useSettingsStore()

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return

    const userMessage = input.trim()
    setInput('')
    setStreaming(true)

    // Add user message
    addMessage(conversationId, {
      role: 'user',
      content: userMessage,
    })

    // Add placeholder for assistant message
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
    })

    try {
      const history = getMessageHistory(conversationId).slice(0, -1) // Exclude the empty assistant message

      // Use REST endpoint (no streaming)
      const response = await sendChatMessage({
        message: userMessage,
        conversation_id: conversationId,
        history,
        stream: false,
        use_cache: true,
        ...(modelTier && { model_tier: modelTier }),
        temperature,
        max_tokens: maxTokens,
      })

      // Update the message with full response and metadata
      const conversations = useChatStore.getState().conversations
      const conversation = conversations.find((c) => c.id === conversationId)
      if (conversation) {
        const updatedMessages = conversation.messages.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: response.message,
                queryId: response.query_id,
                sources: response.sources,
                cached: response.cached,
                model: response.model,
                latencyMs: response.latency_ms,
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

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-border p-4">
      <div className="flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            disabled={isStreaming}
            rows={1}
            className="w-full resize-none rounded-xl border border-border bg-background px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isStreaming}
            className="absolute right-2 bottom-2 p-2 rounded-lg bg-foreground text-background disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
          >
            {isStreaming ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground text-center mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  )
}
