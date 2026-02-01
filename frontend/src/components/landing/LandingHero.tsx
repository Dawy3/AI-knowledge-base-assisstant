'use client'

import { Sparkles, ArrowRight, MessageCircle } from 'lucide-react'
import { useAssistantUIStore } from '@/lib/store'

export function LandingHero() {
  const { setOpen } = useAssistantUIStore()

  return (
    <section className="min-h-[80vh] flex flex-col items-center justify-center text-center px-4 py-16">
      {/* Logo */}
      <div className="w-20 h-20 rounded-2xl bg-foreground flex items-center justify-center mb-8 shadow-lg">
        <Sparkles className="w-10 h-10 text-background" />
      </div>

      {/* Title */}
      <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight mb-4">
        AI Knowledge Assistant
      </h1>

      {/* Description */}
      <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mb-8">
        A powerful RAG-powered AI assistant that understands your documents.
        Upload PDFs, DOCX, and more — then ask questions and get intelligent,
        context-aware answers with source citations.
      </p>

      {/* CTA Button */}
      <button
        onClick={() => setOpen(true)}
        className="group flex items-center gap-2 px-6 py-3 bg-foreground text-background rounded-full text-lg font-medium hover:opacity-90 transition-all shadow-lg hover:shadow-xl"
      >
        Try the Assistant
        <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
      </button>

      {/* Hint */}
      <div className="mt-12 flex items-center gap-2 text-sm text-muted-foreground">
        <span>Or click the</span>
        <span className="inline-flex items-center gap-1.5 px-2 py-1 bg-muted rounded-full">
          <MessageCircle className="w-4 h-4" />
          button
        </span>
        <span>in the bottom right corner</span>
      </div>
    </section>
  )
}
