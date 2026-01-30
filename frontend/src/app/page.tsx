'use client'

import { useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { ChatView } from '@/components/chat/chat-view'
import { DocumentsView } from '@/components/documents/documents-view'
import { FeedbackView } from '@/components/feedback/feedback-view'
import { SettingsView } from '@/components/settings/settings-view'

type View = 'chat' | 'documents' | 'feedback' | 'settings'

export default function Home() {
  const [currentView, setCurrentView] = useState<View>('chat')

  return (
    <div className="flex h-screen">
      <Sidebar currentView={currentView} onViewChange={setCurrentView} />
      <main className="flex-1 overflow-hidden">
        {currentView === 'chat' && <ChatView />}
        {currentView === 'documents' && <DocumentsView />}
        {currentView === 'feedback' && <FeedbackView />}
        {currentView === 'settings' && <SettingsView />}
      </main>
    </div>
  )
}
