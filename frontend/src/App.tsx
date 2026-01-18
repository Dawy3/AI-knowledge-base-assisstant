import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import QueryInterface from './components/QueryInterface'
import DocumentUpload from './components/DocumentUpload'
import SystemStats from './components/SystemStats'
import { Brain, FileText, BarChart3 } from 'lucide-react'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <nav className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-2">
                <Brain className="h-8 w-8 text-primary-400" />
                <h1 className="text-xl font-bold text-white">AI Knowledge Assistant</h1>
              </div>
              <div className="flex space-x-4">
                <Link
                  to="/"
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-slate-300 hover:bg-slate-700/50 transition-colors"
                >
                  <Brain className="h-5 w-5" />
                  <span>Query</span>
                </Link>
                <Link
                  to="/documents"
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-slate-300 hover:bg-slate-700/50 transition-colors"
                >
                  <FileText className="h-5 w-5" />
                  <span>Documents</span>
                </Link>
                <Link
                  to="/stats"
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-slate-300 hover:bg-slate-700/50 transition-colors"
                >
                  <BarChart3 className="h-5 w-5" />
                  <span>Stats</span>
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<QueryInterface />} />
            <Route path="/documents" element={<DocumentUpload />} />
            <Route path="/stats" element={<SystemStats />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App

