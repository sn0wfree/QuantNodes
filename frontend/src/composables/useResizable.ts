import { ref } from 'vue'

export interface UseResizableOptions {
  initialWidth: number
  minWidth: number
  maxWidth: number
  persistKey: string
}

export function useResizable(options: UseResizableOptions) {
  const panelWidth = ref(load())
  const isResizing = ref(false)

  function load(): number {
    try {
      const saved = localStorage.getItem(options.persistKey)
      if (saved) {
        const n = parseInt(saved, 10)
        if (!isNaN(n) && n >= options.minWidth && n <= options.maxWidth) return n
      }
    } catch { /* ignore */ }
    return options.initialWidth
  }

  function save() {
    try {
      localStorage.setItem(options.persistKey, String(Math.round(panelWidth.value)))
    } catch { /* ignore */ }
  }

  function startResize(updateWidth: (e: MouseEvent) => number) {
    isResizing.value = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.body.style.pointerEvents = 'none'

    const onMouseMove = (e: MouseEvent) => {
      const newWidth = updateWidth(e)
      panelWidth.value = Math.max(options.minWidth, Math.min(options.maxWidth, newWidth))
    }

    const onMouseUp = () => {
      isResizing.value = false
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.body.style.pointerEvents = ''
      save()
    }

    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
  }

  return { panelWidth, isResizing, startResize }
}
