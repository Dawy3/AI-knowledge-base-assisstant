'use client'

import { MessageCircle } from 'lucide-react'
import { useAssistantUIStore } from '@/lib/store'
import { AssistantPanel } from './AssistantPanel'

export function AssistantProvider({ children }: { children: React.ReactNode }) {
  const { isOpen, toggleOpen } = useAssistantUIStore()

  return (
    <>
      {children}

      {/* Floating Button */}
      <button
        onClick={toggleOpen}
        className={`fixed bottom-6 right-6 w-14 h-14 rounded-full bg-foreground text-background shadow-lg hover:scale-105 transition-all duration-200 flex items-center justify-center z-40 ${
          isOpen ? 'scale-0 opacity-0' : 'scale-100 opacity-100'
        }`}
        aria-label="Open AI Assistant"
      >
        <MessageCircle className="w-6 h-6" />
      </button>

      {/* Slide-out Panel */}
      <AssistantPanel />
    </>
  )
}
