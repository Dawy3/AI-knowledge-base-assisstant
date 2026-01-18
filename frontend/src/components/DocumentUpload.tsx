import { useState } from 'react'
import { api } from '../api/client'
import { Upload, FileText, Loader2, CheckCircle2, XCircle } from 'lucide-react'

export default function DocumentUpload() {
  const [text, setText] = useState('')
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim() || !source.trim()) {
      setError('Please provide both text and source')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await api.addDocument({
        text: text.trim(),
        source: source.trim(),
      })
      setResult(response)
      setText('')
      setSource('')
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleBatchUpload = async () => {
    if (!text.trim()) {
      setError('Please provide text content')
      return
    }

    // Split text by double newlines or paragraphs
    const documents = text
      .split(/\n\n+/)
      .filter((doc) => doc.trim().length >= 10)
      .map((doc, index) => ({
        text: doc.trim(),
        source: source.trim() || `batch_${Date.now()}_${index}`,
      }))

    if (documents.length === 0) {
      setError('No valid documents found. Each document must be at least 10 characters.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await api.addDocumentsBatch({ documents })
      setResult(response)
      setText('')
      setSource('')
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-4xl font-bold text-white mb-2">Add Documents</h2>
        <p className="text-slate-400">Upload documents to your knowledge base</p>
      </div>

      <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Source Identifier
            </label>
            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="e.g., document_name, url, file_path"
              className="w-full px-4 py-2 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
              disabled={loading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Document Text
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your document text here... (minimum 10 characters)"
              className="w-full px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
              rows={12}
              disabled={loading}
            />
            <p className="mt-2 text-xs text-slate-500">
              For batch upload, separate documents with double newlines (blank line)
            </p>
          </div>

          <div className="flex space-x-4">
            <button
              type="submit"
              disabled={loading || !text.trim() || !source.trim()}
              className="flex-1 bg-gradient-to-r from-primary-600 to-purple-600 hover:from-primary-700 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Upload className="h-5 w-5" />
                  <span>Add Single Document</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleBatchUpload}
              disabled={loading || !text.trim()}
              className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <FileText className="h-5 w-5" />
                  <span>Batch Upload</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start space-x-3">
          <XCircle className="h-5 w-5 text-red-400 mt-0.5 flex-shrink-0" />
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {result && (
        <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <CheckCircle2 className="h-5 w-5 text-green-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-green-400 font-semibold mb-2">Success!</h3>
              <div className="text-sm text-slate-300 space-y-1">
                <p>Status: {result.status}</p>
                {result.doc_id && <p>Document ID: {result.doc_id}</p>}
                {result.num_chunks && <p>Chunks created: {result.num_chunks}</p>}
                {result.total !== undefined && (
                  <>
                    <p>Total: {result.total}</p>
                    <p>Success: {result.success}</p>
                    <p>Skipped: {result.skipped}</p>
                    <p>Errors: {result.errors}</p>
                  </>
                )}
                {result.message && <p className="mt-2">{result.message}</p>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

