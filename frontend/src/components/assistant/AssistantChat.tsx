'use client'

import { useRef, useEffect } from 'react'
import { Plus, MessageSquare } from 'lucide-react'
import { useChatStore } from '@/lib/store'
import { ChatMessage } from '@/components/chat/chat-message'
import { ChatInput } from '@/components/chat/chat-input'
import { AssistantSuggestions } from './AssistantSuggestions'

export function AssistantChat() {
  const {
    currentConversationId,
    createConversation,
    getCurrentConversation,
  } = useChatStore()

  const currentConversation = getCurrentConversation()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [currentConversation?.messages])

  const handleNewChat = () => {
    createConversation()
  }

  // Auto-create conversation if none exists
  useEffect(() => {
    if (!currentConversationId) {
      createConversation()
    }
  }, [currentConversationId, createConversation])

  return (
    <div className="flex flex-col h-full">
      {currentConversation ? (
        <>
          {/* Header with New Chat button */}
          <div className="px-4 py-2 border-b border-border flex items-center justify-between">
            <span className="text-sm text-muted-foreground truncate flex-1">
              {currentConversation.title}
            </span>
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-accent hover:bg-accent/80 rounded-lg transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              New
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {currentConversation.messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-4">
                <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-3">
                  <MessageSquare className="w-6 h-6 text-muted-foreground" />
                </div>
                <h3 className="text-base font-semibold mb-1">How can I help?</h3>
                <p className="text-muted-foreground text-sm mb-4 max-w-[280px]">
                  Ask questions about your documents or try one of these:
                </p>
                <AssistantSuggestions conversationId={currentConversation.id} />
              </div>
            ) : (
              currentConversation.messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <ChatInput conversationId={currentConversation.id} />
        </>
      ) : (
        <div className="h-full flex items-center justify-center">
          <div className="animate-pulse text-muted-foreground">Loading...</div>
        </div>
      )}
    </div>
  )
}
