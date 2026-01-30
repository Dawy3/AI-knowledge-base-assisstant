'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { X, Upload, FileText, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { uploadDocument } from '@/lib/api'
import { clsx } from 'clsx'

interface DocumentUploadProps {
  onClose: () => void
  onSuccess: () => void
}

interface UploadFile {
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  error?: string
}

export function DocumentUpload({ onClose, onSuccess }: DocumentUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [chunkStrategy, setChunkStrategy] = useState('recursive')
  const [chunkSize, setChunkSize] = useState(512)
  const [chunkOverlap, setChunkOverlap] = useState(50)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file) => ({
      file,
      status: 'pending' as const,
    }))
    setFiles((prev) => [...prev, ...newFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/html': ['.html'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  })

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleUpload = async () => {
    if (files.length === 0 || uploading) return

    setUploading(true)

    for (let i = 0; i < files.length; i++) {
      if (files[i].status !== 'pending') continue

      setFiles((prev) =>
        prev.map((f, idx) =>
          idx === i ? { ...f, status: 'uploading' as const } : f
        )
      )

      try {
        await uploadDocument(files[i].file, {
          chunk_strategy: chunkStrategy,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
        })

        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === i ? { ...f, status: 'success' as const } : f
          )
        )
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Upload failed'
        setFiles((prev) =>
          prev.map((f, idx) =>
            idx === i ? { ...f, status: 'error' as const, error: message } : f
          )
        )
      }
    }

    setUploading(false)

    // Check if all uploads were successful
    const allSuccess = files.every((f) => f.status === 'success')
    if (allSuccess) {
      setTimeout(onSuccess, 1000)
    }
  }

  const pendingCount = files.filter((f) => f.status === 'pending').length
  const successCount = files.filter((f) => f.status === 'success').length

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-background rounded-xl shadow-lg max-w-lg w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-lg font-semibold">Upload Documents</h2>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={clsx(
              'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors',
              isDragActive ? 'border-foreground bg-muted' : 'border-border hover:border-foreground/50'
            )}
          >
            <input {...getInputProps()} />
            <Upload className="w-10 h-10 mx-auto mb-3 text-muted-foreground" />
            {isDragActive ? (
              <p className="text-sm">Drop files here...</p>
            ) : (
              <>
                <p className="text-sm font-medium">Drag & drop files here</p>
                <p className="text-xs text-muted-foreground mt-1">
                  or click to browse (PDF, DOCX, TXT, CSV, XLSX, HTML)
                </p>
              </>
            )}
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-2">
              {files.map((f, index) => (
                <div
                  key={index}
                  className="flex items-center gap-3 p-3 rounded-lg bg-muted"
                >
                  <FileText className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{f.file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(f.file.size / 1024).toFixed(1)} KB
                    </p>
                    {f.error && (
                      <p className="text-xs text-red-600">{f.error}</p>
                    )}
                  </div>
                  {f.status === 'pending' && (
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 hover:bg-border rounded transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                  {f.status === 'uploading' && (
                    <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
                  )}
                  {f.status === 'success' && (
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  )}
                  {f.status === 'error' && (
                    <AlertCircle className="w-5 h-5 text-red-600" />
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Chunking options */}
          <div className="space-y-3 p-3 rounded-lg bg-muted/50">
            <h3 className="text-sm font-medium">Chunking Options</h3>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Strategy
                </label>
                <select
                  value={chunkStrategy}
                  onChange={(e) => setChunkStrategy(e.target.value)}
                  className="w-full px-2 py-1.5 text-sm rounded-lg border border-border bg-background"
                >
                  <option value="recursive">Recursive</option>
                  <option value="fixed">Fixed</option>
                  <option value="semantic">Semantic</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Chunk Size
                </label>
                <input
                  type="number"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  min={100}
                  max={2000}
                  className="w-full px-2 py-1.5 text-sm rounded-lg border border-border bg-background"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Overlap
                </label>
                <input
                  type="number"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(Number(e.target.value))}
                  min={0}
                  max={200}
                  className="w-full px-2 py-1.5 text-sm rounded-lg border border-border bg-background"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-border">
          <p className="text-sm text-muted-foreground">
            {files.length === 0
              ? 'No files selected'
              : `${successCount}/${files.length} uploaded`}
          </p>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-border hover:bg-muted transition-colors text-sm"
            >
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={pendingCount === 0 || uploading}
              className="px-4 py-2 rounded-lg bg-foreground text-background hover:opacity-90 transition-opacity text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </span>
              ) : (
                `Upload ${pendingCount > 0 ? pendingCount : ''} File${pendingCount !== 1 ? 's' : ''}`
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
