'use client'

import { useState, useRef, useEffect } from 'react'
import { Plus, Trash2, MessageSquare } from 'lucide-react'
import { useChatStore } from '@/lib/store'
import { ChatMessage } from './chat-message'
import { ChatInput } from './chat-input'
import { clsx } from 'clsx'

export function ChatView() {
  const {
    conversations,
    currentConversationId,
    createConversation,
    setCurrentConversation,
    deleteConversation,
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

  return (
    <div className="flex h-full">
      {/* Conversation List */}
      <div className="w-64 border-r border-border flex flex-col bg-muted/20">
        <div className="p-3 border-b border-border">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-foreground text-background rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {conversations.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No conversations yet
            </p>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={clsx(
                  'group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors',
                  currentConversationId === conv.id
                    ? 'bg-accent'
                    : 'hover:bg-accent/50'
                )}
                onClick={() => setCurrentConversation(conv.id)}
              >
                <MessageSquare className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                <span className="text-sm truncate flex-1">{conv.title}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteConversation(conv.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-border rounded transition-all"
                >
                  <Trash2 className="w-3 h-3 text-muted-foreground" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {currentConversation ? (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {currentConversation.messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-8">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <MessageSquare className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
                  <p className="text-muted-foreground text-sm max-w-md">
                    Ask questions about your documents. The AI will search through your
                    knowledge base and provide relevant answers with sources.
                  </p>
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
          <div className="h-full flex flex-col items-center justify-center text-center p-8">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <MessageSquare className="w-8 h-8 text-muted-foreground" />
            </div>
            <h2 className="text-xl font-semibold mb-2">Welcome to Knowledge Assistant</h2>
            <p className="text-muted-foreground text-sm max-w-md mb-4">
              Your AI-powered assistant for querying your document knowledge base.
            </p>
            <button
              onClick={handleNewChat}
              className="flex items-center gap-2 px-4 py-2 bg-foreground text-background rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
            >
              <Plus className="w-4 h-4" />
              Start New Chat
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
