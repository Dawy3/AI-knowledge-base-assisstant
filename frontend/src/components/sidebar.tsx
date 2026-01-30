'use client'

import { MessageSquare, FileText, BarChart3, Settings, Sparkles } from 'lucide-react'
import { clsx } from 'clsx'

type View = 'chat' | 'documents' | 'feedback' | 'settings'

interface SidebarProps {
  currentView: View
  onViewChange: (view: View) => void
}

const navItems = [
  { id: 'chat' as const, label: 'Chat', icon: MessageSquare },
  { id: 'documents' as const, label: 'Documents', icon: FileText },
  { id: 'feedback' as const, label: 'Analytics', icon: BarChart3 },
  { id: 'settings' as const, label: 'Settings', icon: Settings },
]

export function Sidebar({ currentView, onViewChange }: SidebarProps) {
  return (
    <aside className="w-64 border-r border-border bg-muted/30 flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-foreground flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-background" />
          </div>
          <div>
            <h1 className="font-semibold text-sm">Knowledge Assistant</h1>
            <p className="text-xs text-muted-foreground">AI-Powered RAG</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = currentView === item.id
            return (
              <li key={item.id}>
                <button
                  onClick={() => onViewChange(item.id)}
                  className={clsx(
                    'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                    isActive
                      ? 'bg-foreground text-background'
                      : 'hover:bg-accent text-foreground'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">
          RAG Knowledge System
        </p>
      </div>
    </aside>
  )
}
