-- Ollama embeddings: switch vector dimension 1536 → 768
--
-- Context: OpenAI text-embedding-3-small produces 1536-dim vectors.
-- Ollama nomic-embed-text (default for sovereign/local deployments)
-- produces 768-dim vectors. This migration converts the column type
-- so sovereign deployments work out-of-the-box.
--
-- IMPACT: any existing embeddings will be DROPPED. Re-ingest content
-- via the document upload endpoint after this migration runs.
--
-- Compatibility: safe on empty Chunk table (fresh installs). For
-- existing deployments migrating to Ollama, back up first, then
-- truncate Chunk, then run this migration, then re-ingest.

-- Drop HNSW index (vector dim must match, rebuild after)
DROP INDEX IF EXISTS "chunk_embedding_hnsw_idx";

-- Reset embedding data (1536-dim vectors are incompatible with 768-dim column)
UPDATE "Chunk" SET "embedding" = NULL WHERE "embedding" IS NOT NULL;

-- Change vector dimension
ALTER TABLE "Chunk"
  ALTER COLUMN "embedding" TYPE vector(768)
  USING NULL::vector(768);

-- Rebuild HNSW index for semantic search (cosine similarity)
CREATE INDEX "chunk_embedding_hnsw_idx"
  ON "Chunk"
  USING hnsw ("embedding" vector_cosine_ops);
