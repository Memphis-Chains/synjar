import { Module, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { EMBEDDINGS_SERVICE } from '@/domain/document/embeddings.port';
import { OpenAIEmbeddingsService } from './openai-embeddings.service';
import { OllamaEmbeddingsService } from './ollama-embeddings.service';

/**
 * Embeddings provider selection via EMBEDDINGS_PROVIDER env var.
 *
 *   'ollama' (default) — sovereign/local, uses Ollama's /api/embeddings.
 *                        Zero external dependencies, zero per-token cost.
 *   'openai'           — uses OpenAI's text-embedding-3-small. Requires
 *                        OPENAI_API_KEY. Non-sovereign.
 *
 * Switching providers requires re-embedding existing documents (the two
 * models produce different vector spaces + usually different dimensions).
 * Run a migration that resets the embedding column before switching, and
 * re-ingest content via the Reindex endpoint.
 */
@Module({
  providers: [
    OpenAIEmbeddingsService,
    OllamaEmbeddingsService,
    {
      provide: EMBEDDINGS_SERVICE,
      inject: [ConfigService, OpenAIEmbeddingsService, OllamaEmbeddingsService],
      useFactory: (
        configService: ConfigService,
        openai: OpenAIEmbeddingsService,
        ollama: OllamaEmbeddingsService,
      ) => {
        const logger = new Logger('EmbeddingsModule');
        const provider = configService
          .get<string>('EMBEDDINGS_PROVIDER', 'ollama')
          .toLowerCase();

        if (provider === 'openai') {
          logger.log(
            'Using OpenAI embeddings (text-embedding-3-small, 1536-dim)',
          );
          return openai;
        }

        if (provider !== 'ollama') {
          logger.warn(
            `Unknown EMBEDDINGS_PROVIDER='${provider}', falling back to 'ollama'`,
          );
        }
        logger.log('Using Ollama embeddings (sovereign, local)');
        return ollama;
      },
    },
  ],
  exports: [EMBEDDINGS_SERVICE],
})
export class EmbeddingsModule {}
