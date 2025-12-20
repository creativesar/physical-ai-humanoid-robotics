// Auto-generated TypeScript interfaces from Pydantic models

export interface ChatRequest {
  message: string;
  context?: string | null;
  chapter?: string | null;
  session_id?: string | null;
  user_id?: string | null;
}

export interface ChatResponse {
  response: string;
  sources: Record<string, any>[];
  timestamp: string;
  session_id: string;
  tokens_used?: number | null;
}

export interface StreamChatResponse {
  content: string;
  sources: Record<string, any>[];
  timestamp: string;
  session_id: string;
  is_end?: boolean;
}

export interface DocumentIngestionRequest {
  content: string;
  chapter: string;
  section?: string | null;
  metadata?: Record<string, any> | null;
}

export interface DocumentIngestionResponse {
  document_id: string;
  chunks_count: number;
  processed_at: string;
  status: string;
}

export interface DocumentQueryRequest {
  query: string;
  chapter?: string | null;
  limit?: number;
}

export interface DocumentQueryResponse {
  results: Record<string, any>[];
  query_time: number;
  total_results: number;
}

export interface DocumentInfoResponse {
  id: string;
  title: string;
  chapter: string;
  section?: string | null;
  content_hash: string;
  file_path?: string | null;
  file_type?: string | null;
  size_bytes?: number | null;
  chunk_count?: number | null;
  metadata?: Record<string, any> | null;
  created_at: string;
  updated_at: string;
  processed_at?: string | null;
  status: string;
}

export interface DocumentListResponse {
  documents: DocumentInfoResponse[];
  total: number;
}

export interface APIError {
  error: string;
  message: string;
  status_code: number;
  details?: Record<string, any>;
  timestamp?: string | null;
  request_id?: string | null;
}

export interface ValidationError {
  message: string;
  field?: string | null;
  details?: Record<string, any>;
}

export interface RateLimitError {
  message: string;
  retry_after?: number | null;
  details?: Record<string, any>;
}