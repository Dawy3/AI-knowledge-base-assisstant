"""
Document Processing Pipeline:
- Load documents from various sources
- Extract text and metadata
- Chunk documents
- Generate embeddings
- Store in vector store
"""

from typing import List, Dict, Optional
import hashlib
from datetime import datetime
from pathlib import Path
import mimetypes

import pypdf
import docx

from core.chunking import RecursiveChunker
from core.embeddings import EmbeddingManager
from core.vector_store import PineconeVectorStore
from config.settings import settings


class DocumentProcessor:
    """
    End-to-end document processing pipeline
    """
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embedding_manager: EmbeddingManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.chunker = RecursiveChunker(
            chunk_size= settings.CHUNK_SIZE,
            chunk_overlap= settings.CHUNK_OVERLAP
        )
        
        # Track processed documents to avoid duplicates
        self.processed_docs = set()
        
    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate unique document ID"""
        return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()
    
    def process_text(
        self,
        text:str,
        metadata: Optional[Dict] = None,
        source: str = "unknown"
    ) -> Dict:
        """
        Process a single text document
        
        Args:
            text: Document text content
            metadata: Optional metadata dict
            source: Document source identifier
            
        Returns:
            Processing result with stats
        """
        # Generate document ID
        doc_id = self._generate_doc_id(text, source)
        
        # Check if already processed
        if doc_id in self.processed_docs:
            return{
                "status" : "skipped",
                "reason" : "duplicated",
                "doc_id" : doc_id
            }
            
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "source" : source,
            "doc_id" : doc_id,
            "processed_at" : datetime.now().isoformat(),
            "embedding_version": self.embedding_manager.version
        })
        
        # Chunk document
        chunks = self.chunker.chunk_document(text, metadata)
        
        # Extract text from chunks
        chunk_texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_manager.embed_documents(chunk_texts)
        
        # Prepare metadata for each chunk
        chunk_metadata = []
        for chunk in chunks:
            chunk_meta = chunk["metadata"].copy()
            chunk_meta.update({
                "chunk_id": chunk["chunk_id"],
                "chunk_idx" : chunk["chunk_idx"],
                "total_chunks": chunk["total_chunks"],
                "token_count" : chunk["token_count"]
            })
            chunk_metadata.append(chunk_meta)
            
        # Add vector store
        chunk_ids = self.vector_store.add_document(embeddings, chunk_metadata)
        
        # Index for hybrid search (keyword) 'BM25'
        self.hybrdi_search.index_documents(chunk_ids, chunk_texts)
        
        # Mark as processed
        self.processed_docs.add(doc_id)
        
        return {
            "status" : "success",
            "doc_id" : doc_id,
            "num_chunks": len(chunks),
            "chunk_ids" : chunk_ids,
            "total_tokens": sum(c["token_count"] for c in chunks)
        }
    
    def process_file(self, file_path: str, metadata: Optional[Dict]= None) -> Dict:
        """
        Process a document file
        
        Args:
            file_path: Path to document file
            metadata: Optional metadata
            
        Returns:
            processing result
        """
        path = Path(file_path)
        
        if not path.exists():
            return {
                "status" : "error",
                "reason" : "file_not_found",
                "file_path": str(file_path)
            }
        
        # Detect file type
        mime_type, _ = mimetypes.guess_type(str(path))
        
        try:
            text = ""
            
            # 1. Handle PDF
            if mime_type == "application/pdf":
                reader = pypdf.PdfReader(path)
                text_parts = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_parts.append(extracted)
                text = "\n".join(text_parts)
            
            # 2. Handle DOCX
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(path)
                text = "\n".join([para.text for para in doc.paragraphs])
                
            # 3. Hanlde Plain Text
            elif mime_type and mime_type.startswith("text"):
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
            else:
                return{
                    "status": "error",
                    "reason": "unsupported_file_type",
                    "mime_type": mime_type
                }
                
            # Check if extraction resulted in empty content
            if not text.strip():
                 return {
                    "status": "error",
                    "reason": "empty_content_extracted",
                    "file_path": str(file_path)
                }
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "filename": path.name,
                "file_type": mime_type or "unknown",
                "file_size": path.stat().st_size
            })

            # Process the extracted text
            return self.process_text(text, metadata, source=str(path))
            
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "file_path": str(file_path)
            }
        
    def process_batch(
        self,
        documents: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> Dict:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of dicts with 'text', 'source', and optional 'metadata'
            show_progress: Show progress bar
            
        Returns:
            Batch processing statistics
        """
        
        results = {
            "total" : len(documents),
            "success" : 0,
            "skipped" : 0,
            "errors" : 0,
            "details": 0
        }
        
        for doc in documents:
            text = doc.get("text", "")
            source = doc.get("source", "batch_import")
            metadata = doc.get("metadata", {})
            
            result = self.process_text(text, metadata, source)
            results["details"].append(result)
            
            if result["status"] == "success":
                results["success"] += 1
            elif result["status"] == "skipped":
                results["skipped"] += 1
            else:
                results["errors"] += 1
            
            if show_progress:
                print(f"Processed {results['success'] + results['skipped'] + results['errors']}/{results['total']}")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            "total_documents_processed": len(self.processed_docs),
            "vector_store_size": self.vector_store.next_id,
            "embedding_version": self.embedding_manager.version
        }

            
            