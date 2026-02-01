'use client'

import {
  Brain,
  FileUp,
  Zap,
  FileText,
  Search,
  Code2,
} from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'RAG-Powered Responses',
    description:
      'Retrieval-Augmented Generation ensures answers are grounded in your actual documents, not hallucinated.',
  },
  {
    icon: FileUp,
    title: 'Document Upload',
    description:
      'Easily upload and index your documents. The system automatically chunks and embeds them for retrieval.',
  },
  {
    icon: Zap,
    title: 'Semantic Caching',
    description:
      'Intelligent caching recognizes similar questions and returns instant responses for repeated queries.',
  },
  {
    icon: FileText,
    title: 'Multi-Format Support',
    description:
      'Supports PDF, DOCX, TXT, Markdown, and more. Your knowledge base can include various document types.',
  },
  {
    icon: Search,
    title: 'Fast Retrieval',
    description:
      'Vector-based semantic search finds the most relevant document chunks in milliseconds.',
  },
  {
    icon: Code2,
    title: 'Easy Embedding',
    description:
      'Designed as a sidebar widget — easily embed this assistant into any web application or platform.',
  },
]

export function LandingFeatures() {
  return (
    <section className="py-16 px-4 bg-muted/30">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-2xl sm:text-3xl font-bold text-center mb-4">
          Features
        </h2>
        <p className="text-muted-foreground text-center mb-12 max-w-2xl mx-auto">
          Everything you need to build an intelligent knowledge assistant for your documents
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div
                key={index}
                className="p-6 bg-background rounded-xl border border-border hover:border-foreground/20 transition-colors"
              >
                <div className="w-12 h-12 rounded-lg bg-muted flex items-center justify-center mb-4">
                  <Icon className="w-6 h-6 text-foreground" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">
                  {feature.description}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
