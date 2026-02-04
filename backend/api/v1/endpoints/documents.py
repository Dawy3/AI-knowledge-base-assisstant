"""
Document Upload and Ingestion Endpoint.

FOCUS: Upload documents for RAG ingestion
MUST: Support PDF, DOCX, TXT, CSV, HTML
MUST: Chunk, embed, and store in vector database
PRODUCTION: Uses PostgreSQL for status, vector store for embeddings
"""

import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.v1.dependencies import (
    CurrentUserId,
    DbSession,
    RateLimited,
)
from backend.core.config import settings
from backend.db.models import Document, DocumentChunk
from backend.services.document_processor import DocumentProcessor
from backend.core.chunking.strategies import get_chunker
from backend.core.embedding.generator import create_embedding_generator
from backend.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    message: str
    chunks_created: Optional[int] = None
    processing_time_ms: Optional[float] = None


class DocumentStatus(BaseModel):
    """Document processing status."""
    document_id: str
    filename: str
    status: str
    chunks_created: int = 0
    chunks_embedded: int = 0
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: list[DocumentStatus]
    total: int
    page: int
    page_size: int


class ChunkingOptions(BaseModel):
    """Options for document chunking."""
    strategy: str = Field(
        default="recursive",
        description="Chunking strategy: recursive, fixed, semantic, sentence, page",
    )
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlap between chunks in tokens",
    )


# =============================================================================
# Background Processing
# =============================================================================

async def process_document_background(
    document_id: str,
    file_path: str,
    filename: str,
    metadata: Optional[dict],
    chunking_options: ChunkingOptions,
    user_id: Optional[str],
):
    """
    Background task to process uploaded document.

    PRODUCTION Flow:
    1. Extract text from document
    2. Chunk text into segments
    3. Generate embeddings for chunks
    4. Store chunks in PostgreSQL
    5. Store embeddings in vector database
    6. Update document status in PostgreSQL
    """
    from backend.db.session import get_async_session_local

    start_time = time.perf_counter()

    # Get database session
    session_factory = get_async_session_local()
    async with session_factory() as db:
        try:
            # Update status to processing
            await _update_document_status(db, document_id, "processing")

            # Step 1: Extract text
            logger.info(f"Processing document: {document_id} ({filename})")
            processor = DocumentProcessor()

            try:
                doc = processor.process(file_path)
            except Exception as e:
                logger.error(f"Failed to extract text: {e}")
                await _update_document_status(
                    db, document_id, "failed",
                    error_message=f"Extraction failed: {e}"
                )
                return

            if not doc.content or len(doc.content.strip()) < 10:
                await _update_document_status(
                    db, document_id, "failed",
                    error_message="Document appears to be empty"
                )
                return

            logger.info(f"Extracted {len(doc.content)} characters from {filename}")

            # Step 2: Chunk text
            chunker = get_chunker(
                strategy=chunking_options.strategy,
                chunk_size=chunking_options.chunk_size,
                chunk_overlap=chunking_options.chunk_overlap,
            )

            chunks = chunker.chunk(
                text=doc.content,
                document_id=document_id,
                source_file=filename,
                metadata={
                    **(metadata or {}),
                    "page_count": doc.page_count,
                    "file_type": doc.metadata.get("file_type"),
                    "uploaded_by": user_id,
                    "uploaded_at": datetime.utcnow().isoformat(),
                },
            )

            if not chunks:
                await _update_document_status(
                    db, document_id, "failed",
                    error_message="No chunks created"
                )
                return

            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            await _update_document_status(
                db, document_id, "embedding",
                chunks_created=len(chunks)
            )

            # Step 3: Store chunks in PostgreSQL
            db_chunks = []
            for i, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    id=uuid.uuid4(),  # Always generate a fresh UUID for DB
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk.content,
                    token_count=chunk.token_count if hasattr(chunk, 'token_count') else 0,
                    source_file=filename,
                    start_char=chunk.start_char if hasattr(chunk, 'start_char') else None,
                    end_char=chunk.end_char if hasattr(chunk, 'end_char') else None,
                    chunk_metadata=chunk.metadata if hasattr(chunk, 'metadata') else {},
                    embedding_model=settings.embedding.model_name,
                    is_embedded=False,
                )
                db_chunks.append(db_chunk)
                db.add(db_chunk)

            await db.commit()
            logger.info(f"Stored {len(db_chunks)} chunks in PostgreSQL")

            # Step 4: Generate embeddings
            try:
                embedding_generator = create_embedding_generator()

                chunk_texts = [chunk.content for chunk in chunks]
                result = await embedding_generator.embed_texts(chunk_texts)

                logger.info(
                    f"Generated {len(result.embeddings)} embeddings "
                    f"({result.cached_count} from cache)"
                )

            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                await _update_document_status(
                    db, document_id, "failed",
                    chunks_created=len(chunks),
                    error_message=f"Embedding failed: {e}"
                )
                return

            # Step 5: Store in vector database
            try:
                vector_store = get_vector_store()

                # Prepare vectors for upsert
                vectors = []
                for i, (chunk, embedding) in enumerate(zip(chunks, result.embeddings)):
                    chunk_id = str(db_chunks[i].id)
                    vectors.append({
                        "id": chunk_id,
                        "values": embedding,
                        "metadata": {
                            "document_id": document_id,
                            "chunk_index": i,
                            "content": chunk.content[:1000],  # Truncate for metadata
                            "source_file": filename,
                            **(chunk.metadata if hasattr(chunk, 'metadata') else {}),
                        }
                    })

                await vector_store.upsert(vectors)

                # Mark chunks as embedded
                for db_chunk in db_chunks:
                    db_chunk.is_embedded = True
                await db.commit()

                logger.info(f"Stored {len(vectors)} vectors in vector store")

            except Exception as e:
                logger.error(f"Failed to store in vector database: {e}")
                await _update_document_status(
                    db, document_id, "failed",
                    chunks_created=len(chunks),
                    error_message=f"Vector storage failed: {e}"
                )
                return

            # Step 6: Update final status
            processing_time = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"Document {document_id} processed successfully: "
                f"{len(chunks)} chunks in {processing_time:.2f}ms"
            )

            await _update_document_status(
                db, document_id, "completed",
                chunks_created=len(chunks),
                chunks_embedded=len(result.embeddings),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.exception(f"Document processing failed: {e}")
            await _update_document_status(
                db, document_id, "failed",
                error_message=str(e)
            )

        finally:
            # Clean up temp file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass


async def _update_document_status(
    db: AsyncSession,
    document_id: str,
    status: str,
    chunks_created: int = None,
    chunks_embedded: int = None,
    error_message: str = None,
    processing_time_ms: float = None,
):
    """Update document processing status in database."""
    result = await db.execute(
        select(Document).where(Document.id == uuid.UUID(document_id))
    )
    doc = result.scalar_one_or_none()

    if doc:
        doc.status = status
        doc.updated_at = datetime.utcnow()

        if chunks_created is not None:
            doc.chunks_created = chunks_created
        if chunks_embedded is not None:
            doc.chunks_embedded = chunks_embedded
        if error_message is not None:
            doc.error_message = error_message
        if processing_time_ms is not None:
            doc.processing_time_ms = processing_time_ms

        await db.commit()


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        200: {"description": "Document uploaded and processing started"},
        400: {"description": "Invalid file type"},
        413: {"description": "File too large"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def upload_document(
    background_tasks: BackgroundTasks,
    db: DbSession,
    file: UploadFile = File(..., description="Document file to upload"),
    title: Optional[str] = Form(None, description="Document title"),
    description: Optional[str] = Form(None, description="Document description"),
    tags: Optional[str] = Form(None, description="Comma-separated tags"),
    category: Optional[str] = Form(None, description="Document category"),
    chunk_strategy: str = Form("recursive", description="Chunking strategy"),
    chunk_size: int = Form(512, ge=100, le=2000, description="Chunk size in tokens"),
    chunk_overlap: int = Form(50, ge=0, le=200, description="Chunk overlap"),
    user_id: CurrentUserId = None,
    _rate_limit: RateLimited = None,
):
    """
    Upload a document for RAG ingestion.

    FOCUS: Accept documents, queue for processing
    MUST: Support PDF, DOCX, TXT, CSV, HTML
    PRODUCTION: Stores status in PostgreSQL, vectors in vector store

    Supported formats:
    - PDF (.pdf)
    - Word (.docx, .doc)
    - Text (.txt, .md)
    - CSV (.csv)
    - Excel (.xlsx)
    - HTML (.html)
    """
    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    supported_types = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".html"}

    if ext not in supported_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported_types)}",
        )

    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB

    # Read file content
    content = await file.read()
    file_size = len(content)

    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    # Generate document ID
    document_id = str(uuid.uuid4())

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{document_id}{ext}")

    with open(temp_path, "wb") as f:
        f.write(content)

    # Build metadata
    metadata = {
        "title": title or filename,
        "description": description,
        "tags": tags.split(",") if tags else [],
        "category": category,
    }

    # Build chunking options
    chunking_options = ChunkingOptions(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create document record in PostgreSQL
    db_document = Document(
        id=uuid.UUID(document_id),
        filename=filename,
        file_type=ext,
        file_size=file_size,
        status="queued",
        title=title or filename,
        description=description,
        tags=tags.split(",") if tags else [],
        category=category,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        user_id=user_id,
    )
    db.add(db_document)
    await db.commit()

    # Queue background processing
    background_tasks.add_task(
        process_document_background,
        document_id=document_id,
        file_path=temp_path,
        filename=filename,
        metadata=metadata,
        chunking_options=chunking_options,
        user_id=user_id,
    )

    logger.info(f"Document upload queued: {document_id} ({filename}, {file_size} bytes)")

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=ext,
        file_size=file_size,
        status="processing",
        message="Document uploaded successfully. Processing started.",
    )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatus,
    responses={
        200: {"description": "Document status"},
        404: {"description": "Document not found"},
    },
)
async def get_document_status(
    document_id: str,
    db: DbSession,
    user_id: CurrentUserId = None,
):
    """
    Get document processing status from PostgreSQL.

    Status values:
    - queued: Waiting to be processed
    - processing: Currently being processed
    - embedding: Generating embeddings
    - completed: Successfully processed
    - failed: Processing failed (check error_message)
    """
    result = await db.execute(
        select(Document).where(Document.id == uuid.UUID(document_id))
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return DocumentStatus(
        document_id=str(doc.id),
        filename=doc.filename,
        status=doc.status,
        chunks_created=doc.chunks_created or 0,
        chunks_embedded=doc.chunks_embedded or 0,
        error_message=doc.error_message,
        created_at=doc.created_at.isoformat(),
        updated_at=doc.updated_at.isoformat(),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={
        200: {"description": "List of documents"},
    },
)
async def list_documents(
    db: DbSession,
    page: int = 1,
    page_size: int = 20,
    status_filter: Optional[str] = None,
    user_id: CurrentUserId = None,
):
    """
    List uploaded documents with their processing status from PostgreSQL.
    """
    # Build query
    query = select(Document)

    if status_filter:
        query = query.where(Document.status == status_filter)

    if user_id:
        query = query.where(Document.user_id == user_id)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Paginate
    query = query.order_by(Document.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[
            DocumentStatus(
                document_id=str(doc.id),
                filename=doc.filename,
                status=doc.status,
                chunks_created=doc.chunks_created or 0,
                chunks_embedded=doc.chunks_embedded or 0,
                error_message=doc.error_message,
                created_at=doc.created_at.isoformat(),
                updated_at=doc.updated_at.isoformat(),
            )
            for doc in docs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/{document_id}",
    responses={
        200: {"description": "Document deleted"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    db: DbSession,
    user_id: CurrentUserId = None,
):
    """
    Delete a document and its chunks from the system.

    This removes:
    - Document metadata from PostgreSQL
    - All chunks from PostgreSQL
    - All vectors from vector store
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document ID format: {document_id}",
        )

    # Check document exists
    result = await db.execute(
        select(Document).where(Document.id == doc_uuid)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    # Delete vectors from vector store (don't fail if vector store has issues)
    try:
        vector_store = get_vector_store()
        await vector_store.connect()  # Ensure connected
        await vector_store.delete(filter={"document_id": document_id})
        logger.info(f"Deleted vectors for document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete vectors (continuing anyway): {e}")

    # Delete chunks from PostgreSQL (document_id is stored as string in chunks table)
    try:
        chunks_result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = chunks_result.scalars().all()
        for chunk in chunks:
            await db.delete(chunk)
        chunks_deleted = len(chunks)
    except Exception as e:
        logger.warning(f"Failed to delete chunks: {e}")
        chunks_deleted = 0

    # Delete document from PostgreSQL
    await db.delete(doc)
    await db.commit()

    logger.info(f"Document deleted: {document_id}")

    return {
        "document_id": document_id,
        "deleted": True,
        "chunks_deleted": chunks_deleted,
        "message": "Document and all chunks deleted successfully",
    }


@router.post(
    "/batch",
    responses={
        200: {"description": "Batch upload started"},
        400: {"description": "Invalid request"},
    },
)
async def batch_upload(
    background_tasks: BackgroundTasks,
    db: DbSession,
    files: list[UploadFile] = File(..., description="Multiple document files"),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    user_id: CurrentUserId = None,
    _rate_limit: RateLimited = None,
):
    """
    Upload multiple documents at once.

    All files will be processed in parallel.
    Returns list of document IDs for status tracking.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files per batch upload",
        )

    results = []

    for file in files:
        try:
            # Process each file
            response = await upload_document(
                background_tasks=background_tasks,
                db=db,
                file=file,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                user_id=user_id,
            )
            results.append({
                "filename": file.filename,
                "document_id": response.document_id,
                "status": "queued",
            })
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "document_id": None,
                "status": "failed",
                "error": e.detail,
            })

    return {
        "batch_id": str(uuid.uuid4()),
        "documents": results,
        "total": len(results),
        "queued": sum(1 for r in results if r["status"] == "queued"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
    }
