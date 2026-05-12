import { onMounted, onUnmounted } from 'vue'

type KeyHandler = (e: KeyboardEvent) => void

export function useKeyboard() {
  const shortcuts = new Map<string, KeyHandler>()

  const register = (key: string, handler: KeyHandler) => {
    shortcuts.set(key, handler)
  }

  const unregister = (key: string) => {
    shortcuts.delete(key)
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement).tagName
    const isInput = tag === 'INPUT' || tag === 'TEXTAREA'

    const key = [
      e.ctrlKey ? 'ctrl' : '',
      e.metaKey ? 'meta' : '',
      e.shiftKey ? 'shift' : '',
      e.key.toLowerCase(),
    ].filter(Boolean).join('+')

    if (isInput && key !== 'ctrl+s' && key !== 'escape') return

    const handler = shortcuts.get(key)
    if (handler) {
      e.preventDefault()
      handler(e)
    }
  }

  onMounted(() => document.addEventListener('keydown', handleKeyDown))
  onUnmounted(() => document.removeEventListener('keydown', handleKeyDown))

  return { register, unregister }
}
