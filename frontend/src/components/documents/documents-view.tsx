'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { FileText, Trash2, RefreshCw, CheckCircle, XCircle, Loader2, Clock, FileUp } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { clsx } from 'clsx'
import { getDocuments, deleteDocument, type Document } from '@/lib/api'
import { DocumentUpload } from './document-upload'

const statusConfig: Record<string, { icon: typeof Clock; color: string; bg: string; animate?: boolean }> = {
  queued: { icon: Clock, color: 'text-muted-foreground', bg: 'bg-muted' },
  processing: { icon: Loader2, color: 'text-blue-600', bg: 'bg-blue-100', animate: true },
  embedding: { icon: Loader2, color: 'text-purple-600', bg: 'bg-purple-100', animate: true },
  completed: { icon: CheckCircle, color: 'text-green-600', bg: 'bg-green-100' },
  failed: { icon: XCircle, color: 'text-red-600', bg: 'bg-red-100' },
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function DocumentsView() {
  const [showUpload, setShowUpload] = useState(false)
  const [page, setPage] = useState(1)
  const queryClient = useQueryClient()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['documents', page],
    queryFn: () => getDocuments(page, 20),
    refetchInterval: 5000, // Poll for status updates
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const handleDelete = async (doc: Document) => {
    if (confirm(`Are you sure you want to delete "${doc.filename}"?`)) {
      deleteMutation.mutate(doc.id)
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Documents</h1>
            <p className="text-sm text-muted-foreground">
              Manage your knowledge base documents
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => refetch()}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowUpload(true)}
              className="flex items-center gap-2 px-4 py-2 bg-foreground text-background rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
            >
              <FileUp className="w-4 h-4" />
              Upload
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : !data?.documents?.length ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <FileText className="w-8 h-8 text-muted-foreground" />
            </div>
            <h2 className="text-lg font-semibold mb-2">No documents yet</h2>
            <p className="text-muted-foreground text-sm mb-4">
              Upload documents to build your knowledge base
            </p>
            <button
              onClick={() => setShowUpload(true)}
              className="flex items-center gap-2 px-4 py-2 bg-foreground text-background rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
            >
              <FileUp className="w-4 h-4" />
              Upload Document
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            {data.documents.map((doc) => {
              const status = statusConfig[doc.status]
              const StatusIcon = status.icon
              return (
                <div
                  key={doc.id}
                  className="flex items-center gap-4 p-4 rounded-lg border border-border hover:bg-muted/50 transition-colors"
                >
                  {/* File icon */}
                  <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                    <FileText className="w-5 h-5 text-muted-foreground" />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium truncate">{doc.title || doc.filename}</h3>
                      <span
                        className={clsx(
                          'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs',
                          status.bg,
                          status.color
                        )}
                      >
                        <StatusIcon
                          className={clsx('w-3 h-3', status.animate && 'animate-spin')}
                        />
                        {doc.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                      <span>{doc.file_type}</span>
                      <span>{formatFileSize(doc.file_size)}</span>
                      {doc.chunks_created && (
                        <span>{doc.chunks_created} chunks</span>
                      )}
                      <span>
                        {formatDistanceToNow(new Date(doc.created_at), { addSuffix: true })}
                      </span>
                    </div>
                    {doc.error_message && (
                      <p className="text-xs text-red-600 mt-1">{doc.error_message}</p>
                    )}
                  </div>

                  {/* Actions */}
                  <button
                    onClick={() => handleDelete(doc)}
                    disabled={deleteMutation.isPending}
                    className="p-2 rounded-lg hover:bg-border text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                    title="Delete document"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              )
            })}
          </div>
        )}

        {/* Pagination */}
        {data && data.total > data.page_size && (
          <div className="flex items-center justify-center gap-2 mt-4">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 rounded-lg border border-border hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              Previous
            </button>
            <span className="text-sm text-muted-foreground">
              Page {page} of {Math.ceil(data.total / data.page_size)}
            </span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={page >= Math.ceil(data.total / data.page_size)}
              className="px-3 py-1 rounded-lg border border-border hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <DocumentUpload
          onClose={() => setShowUpload(false)}
          onSuccess={() => {
            setShowUpload(false)
            queryClient.invalidateQueries({ queryKey: ['documents'] })
          }}
        />
      )}
    </div>
  )
}
