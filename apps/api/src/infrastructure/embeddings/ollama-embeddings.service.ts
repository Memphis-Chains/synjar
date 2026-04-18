import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  IEmbeddingsService,
  EmbeddingResult,
} from '@/domain/document/embeddings.port';

/**
 * Ollama-based embeddings provider for sovereign/local RAG deployments.
 *
 * Uses Ollama's HTTP embeddings API (POST /api/embeddings) with any
 * locally-pulled embedding model. Default: nomic-embed-text (768-dim).
 *
 * Why this exists:
 * - OpenAI dependency is incompatible with air-gapped, sovereign, or
 *   budget-constrained deployments.
 * - Ollama runs locally on the same host (or a LAN peer) with zero
 *   per-token cost and no data leaving the network.
 * - Model choice is operator-controlled: set OLLAMA_EMBEDDING_MODEL
 *   to any model that supports the /api/embeddings endpoint
 *   (nomic-embed-text, mxbai-embed-large, bge-m3, etc.).
 *
 * Token counting:
 *   Ollama does not return token usage in the embeddings response.
 *   We approximate via word count (~1.3 tokens/word for English) for
 *   billing / rate-limit telemetry parity with the OpenAI adapter.
 *   Absolute exactness is not required — the field is primarily used
 *   for soft limits and cost visibility.
 *
 * Environment variables:
 *   OLLAMA_BASE_URL            default http://localhost:11434
 *   OLLAMA_EMBEDDING_MODEL     default nomic-embed-text
 *   OLLAMA_EMBEDDING_DIMENSIONS  default 768 (must match your chosen
 *                                model's output dim + the pgvector
 *                                column dim in Prisma schema).
 */
@Injectable()
export class OllamaEmbeddingsService implements IEmbeddingsService {
  private readonly logger = new Logger(OllamaEmbeddingsService.name);
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly expectedDim: number;

  constructor(private readonly configService: ConfigService) {
    this.baseUrl = this.configService.get<string>(
      'OLLAMA_BASE_URL',
      'http://localhost:11434',
    );
    this.model = this.configService.get<string>(
      'OLLAMA_EMBEDDING_MODEL',
      'nomic-embed-text',
    );
    this.expectedDim = Number(
      this.configService.get<string>('OLLAMA_EMBEDDING_DIMENSIONS', '768'),
    );

    this.logger.log(
      `Ollama embeddings: base=${this.baseUrl} model=${this.model} dim=${this.expectedDim}`,
    );
  }

  async generateEmbedding(text: string): Promise<EmbeddingResult> {
    if (!text || !text.trim()) {
      throw new Error(
        `Cannot generate embedding: input text is empty or whitespace-only (length: ${text?.length ?? 0})`,
      );
    }

    const embedding = await this.fetchEmbedding(text);
    this.assertDimension(embedding);

    return {
      embedding,
      tokenCount: this.approximateTokenCount(text),
    };
  }

  async generateEmbeddings(texts: string[]): Promise<EmbeddingResult[]> {
    if (texts.length === 0) {
      throw new Error('Cannot generate embeddings: input array is empty');
    }

    const emptyIndices = texts
      .map((t, i) => (!t || !t.trim() ? i : -1))
      .filter((i) => i !== -1);
    if (emptyIndices.length > 0) {
      throw new Error(
        `Cannot generate embeddings: ${emptyIndices.length} of ${texts.length} texts are empty at indices [${emptyIndices.join(', ')}]`,
      );
    }

    // Ollama's /api/embed endpoint (v0.5.0+) accepts an array, processes
    // server-side in a single round trip. Much faster than parallel
    // /api/embeddings calls (which serialize anyway at the model layer).
    const embeddings = await this.fetchBatchEmbeddings(texts);

    if (embeddings.length !== texts.length) {
      throw new Error(
        `Ollama returned ${embeddings.length} embeddings for ${texts.length} inputs — mismatch`,
      );
    }

    return embeddings.map((embedding, i) => {
      this.assertDimension(embedding);
      return {
        embedding,
        tokenCount: this.approximateTokenCount(texts[i]),
      };
    });
  }

  private async fetchEmbedding(text: string): Promise<number[]> {
    // Uses the newer /api/embed endpoint (v0.5.0+) even for single
    // inputs — it returns the same shape as batch and is more
    // performant than the legacy /api/embeddings single-prompt route.
    const [embedding] = await this.fetchBatchEmbeddings([text]);
    return embedding;
  }

  private async fetchBatchEmbeddings(texts: string[]): Promise<number[][]> {
    const url = `${this.baseUrl}/api/embed`;
    const body = JSON.stringify({
      model: this.model,
      input: texts,
    });

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => '(no body)');
      throw new Error(
        `Ollama /api/embed failed: HTTP ${response.status} ${response.statusText} — ${errorText.slice(0, 200)}`,
      );
    }

    const data = (await response.json()) as { embeddings?: number[][] };
    if (!data.embeddings || !Array.isArray(data.embeddings)) {
      throw new Error(
        `Ollama /api/embed response missing 'embeddings' array: ${JSON.stringify(data).slice(0, 200)}`,
      );
    }

    return data.embeddings;
  }

  private assertDimension(embedding: number[]): void {
    if (embedding.length !== this.expectedDim) {
      throw new Error(
        `Ollama returned ${embedding.length}-dim embedding, expected ${this.expectedDim}. ` +
          `Check OLLAMA_EMBEDDING_MODEL (${this.model}) output dimensions and ` +
          `align OLLAMA_EMBEDDING_DIMENSIONS env + pgvector column dim in schema.prisma.`,
      );
    }
  }

  /**
   * Approximate token count. Ollama does not report usage in embeddings
   * response. We use the heuristic 1 word ≈ 1.3 tokens (OpenAI norm for
   * English; roughly OK for mixed content). Exact counts are not
   * required — the field is used for billing/rate-limit visibility.
   */
  private approximateTokenCount(text: string): number {
    const words = text.trim().split(/\s+/).length;
    return Math.max(1, Math.ceil(words * 1.3));
  }
}
