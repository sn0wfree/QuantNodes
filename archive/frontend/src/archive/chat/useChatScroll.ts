import { ref, nextTick, watch, type Ref } from 'vue'

export function useChatScroll(
  containerRef: Ref<HTMLElement | undefined>,
  options: {
    watchSources?: (() => any)[]
    threshold?: number
  } = {}
) {
  const isAtBottom = ref(true)
  const threshold = options.threshold ?? 100

  const scrollToBottom = () => {
    nextTick(() => {
      if (containerRef.value) {
        containerRef.value.scrollTop = containerRef.value.scrollHeight
      }
    })
  }

  const handleScroll = () => {
    if (!containerRef.value) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.value
    isAtBottom.value = scrollHeight - scrollTop - clientHeight < threshold
  }

  // Auto-scroll when watch sources change
  if (options.watchSources) {
    for (const source of options.watchSources) {
      watch(source, () => {
        scrollToBottom()
      })
    }
  }

  return {
    isAtBottom,
    scrollToBottom,
    handleScroll,
  }
}
