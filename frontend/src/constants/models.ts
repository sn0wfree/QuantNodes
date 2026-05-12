export interface ModelInfo {
  id: string
  name: string
  provider: string
  contextWindow: number
  priceIn: number
  priceOut: number
  tags: string[]
}

export const MODEL_REGISTRY: ModelInfo[] = [
  {
    id: 'minimax/minimax-m2.5:free',
    name: 'MiniMax M2.5 (Free)',
    provider: 'MiniMax',
    contextWindow: 1000000,
    priceIn: 0,
    priceOut: 0,
    tags: ['free'],
  },
  {
    id: 'minimax/minimax-m2.5',
    name: 'MiniMax M2.5',
    provider: 'MiniMax',
    contextWindow: 1000000,
    priceIn: 0.15,
    priceOut: 1.15,
    tags: [],
  },
  {
    id: 'minimax/minimax-m2.7',
    name: 'MiniMax M2.7',
    provider: 'MiniMax',
    contextWindow: 1000000,
    priceIn: 0.30,
    priceOut: 1.20,
    tags: [],
  },
  {
    id: 'openai/gpt-4o',
    name: 'GPT-4o',
    provider: 'OpenAI',
    contextWindow: 128000,
    priceIn: 2.50,
    priceOut: 10.00,
    tags: ['reasoning'],
  },
  {
    id: 'openai/gpt-4o-mini',
    name: 'GPT-4o Mini',
    provider: 'OpenAI',
    contextWindow: 128000,
    priceIn: 0.15,
    priceOut: 0.60,
    tags: ['fast'],
  },
]

export function getModelInfo(id: string): ModelInfo | undefined {
  return MODEL_REGISTRY.find(m => m.id === id)
}

export function formatPrice(price: number): string {
  if (price === 0) return 'free'
  return `$${price.toFixed(2)}/M`
}

export function formatContextWindow(tokens: number): string {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(0)}M`
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`
  return String(tokens)
}
