import type { Metadata } from 'next'
import './globals.css'
import { Providers } from '@/components/providers'
import { AssistantProvider } from '@/components/assistant'

export const metadata: Metadata = {
  title: 'AI Knowledge Assistant',
  description: 'AI-powered knowledge assistant with RAG capabilities',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <AssistantProvider>
            {children}
          </AssistantProvider>
        </Providers>
      </body>
    </html>
  )
}
