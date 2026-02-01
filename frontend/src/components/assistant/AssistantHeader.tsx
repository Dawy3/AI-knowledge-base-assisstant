'use client'

import { X, ChevronDown, MessageSquare, FileText, Settings, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { useAssistantUIStore } from '@/lib/store'
import { clsx } from 'clsx'

const panels = [
  { id: 'chat' as const, label: 'Chat', icon: MessageSquare },
  { id: 'documents' as const, label: 'Documents', icon: FileText },
  { id: 'settings' as const, label: 'Settings', icon: Settings },
]

export function AssistantHeader() {
  const { activePanel, setActivePanel, setOpen } = useAssistantUIStore()
  const [dropdownOpen, setDropdownOpen] = useState(false)

  const currentPanel = panels.find((p) => p.id === activePanel)
  const CurrentIcon = currentPanel?.icon || MessageSquare

  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-foreground flex items-center justify-center">
          <Sparkles className="w-5 h-5 text-background" />
        </div>

        {/* Panel Selector Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-accent transition-colors text-sm font-medium"
          >
            <CurrentIcon className="w-4 h-4" />
            {currentPanel?.label}
            <ChevronDown className={clsx('w-4 h-4 transition-transform', dropdownOpen && 'rotate-180')} />
          </button>

          {dropdownOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setDropdownOpen(false)} />
              <div className="absolute top-full left-0 mt-1 w-40 bg-background border border-border rounded-lg shadow-lg z-20 py-1">
                {panels.map((panel) => {
                  const Icon = panel.icon
                  return (
                    <button
                      key={panel.id}
                      onClick={() => {
                        setActivePanel(panel.id)
                        setDropdownOpen(false)
                      }}
                      className={clsx(
                        'w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors',
                        activePanel === panel.id
                          ? 'bg-accent text-foreground'
                          : 'hover:bg-accent/50 text-muted-foreground hover:text-foreground'
                      )}
                    >
                      <Icon className="w-4 h-4" />
                      {panel.label}
                    </button>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>

      <button
        onClick={() => setOpen(false)}
        className="p-2 rounded-lg hover:bg-accent transition-colors"
        aria-label="Close assistant"
      >
        <X className="w-5 h-5" />
      </button>
    </div>
  )
}
