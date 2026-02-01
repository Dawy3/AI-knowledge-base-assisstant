'use client'

import { LandingHero, LandingFeatures } from '@/components/landing'

export default function Home() {
  return (
    <main className="min-h-screen">
      <LandingHero />
      <LandingFeatures />

      {/* Footer */}
      <footer className="py-8 text-center text-sm text-muted-foreground border-t border-border">
        <p>AI Knowledge Assistant — RAG-Powered Document Q&A</p>
      </footer>
    </main>
  )
}
