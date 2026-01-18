import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface QueryRequest {
  query: string
  top_k?: number
  use_cache?: boolean
  return_sources?: boolean
}

export interface QueryResponse {
  query_id: string
  query: string
  answer: string
  status: string
  sources?: Source[]
  retrieval_stats?: {
    num_variants: number
    total_retrieved: number
    final_count: number
  }
  latency_ms: number
  cache_hit: boolean
  timestamp: string
}

export interface Source {
  chunk_id: number
  score: number
  metadata: Record<string, any>
}

export interface DocumentInput {
  text: string
  source: string
  metadata?: Record<string, any>
}

export interface DocumentResponse {
  status: string
  doc_id?: string
  num_chunks?: number
  message: string
}

export interface BatchDocumentInput {
  documents: DocumentInput[]
}

export interface BatchDocumentResponse {
  total: number
  success: number
  skipped: number
  errors: number
  message: string
}

export interface SystemStats {
  system: string
  initialized: boolean
  document_processor: {
    total_documents_processed: number
    vector_store_size: number
    embedding_version: string
  }
  embedding: {
    version: string
    model: string
  }
  cache?: {
    size: number
    hit_rate: number
  }
}

export interface MetricsResponse {
  total_queries: number
  successful_queries: number
  failed_queries: number
  cache_hit_rate: number
  avg_latency_ms: number
  p99_latency_ms: number
}

export const api = {
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await apiClient.post<QueryResponse>('/query', request)
    return response.data
  },

  addDocument: async (document: DocumentInput): Promise<DocumentResponse> => {
    const response = await apiClient.post<DocumentResponse>('/documents', document)
    return response.data
  },

  addDocumentsBatch: async (batch: BatchDocumentInput): Promise<BatchDocumentResponse> => {
    const response = await apiClient.post<BatchDocumentResponse>('/documents/batch', batch)
    return response.data
  },

  getStats: async (): Promise<SystemStats> => {
    const response = await apiClient.get<SystemStats>('/stats')
    return response.data
  },

  getMetrics: async (timePeriod: string = '1h'): Promise<MetricsResponse> => {
    const response = await apiClient.get<MetricsResponse>('/metrics', {
      params: { time_period: timePeriod },
    })
    return response.data
  },

  healthCheck: async () => {
    const response = await apiClient.get('/health')
    return response.data
  },
}

export default apiClient

