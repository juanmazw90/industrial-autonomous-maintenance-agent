-- Enable pgvector for RAG embeddings (used from V1)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for fuzzy text search on machine IDs / failure modes
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Separate database for Langfuse (avoids Prisma migration conflicts with AMIA tables)
CREATE DATABASE langfuse;

-- Placeholder schema — tables created by SQLAlchemy migrations in V1+
-- This file runs on first container startup to ensure extensions are ready.
