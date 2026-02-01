'use client'

import { useAssistantUIStore } from '@/lib/store'
import { AssistantHeader } from './AssistantHeader'
import { AssistantChat } from './AssistantChat'
import { DocumentsView } from '@/components/documents/documents-view'
import { SettingsView } from '@/components/settings/settings-view'

export function AssistantPanel() {
  const { isOpen, activePanel } = useAssistantUIStore()

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => useAssistantUIStore.getState().setOpen(false)}
      />

      {/* Panel */}
      <div
        className={`fixed top-0 right-0 h-full w-full sm:w-[420px] bg-background border-l border-border shadow-2xl z-50 flex flex-col transition-transform duration-300 ease-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <AssistantHeader />

        <div className="flex-1 overflow-hidden">
          {activePanel === 'chat' && <AssistantChat />}
          {activePanel === 'documents' && <DocumentsView />}
          {activePanel === 'settings' && <SettingsView />}
        </div>
      </div>
    </>
  )
}
