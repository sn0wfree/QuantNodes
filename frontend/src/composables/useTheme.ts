import { ref, watch } from 'vue'

const STORAGE_KEY = 'quantnodes-theme'

export function useTheme() {
  const theme = ref<'light' | 'dark'>(
    (localStorage.getItem(STORAGE_KEY) as 'light' | 'dark') || 'light'
  )

  const toggleTheme = () => {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }

  watch(theme, (newTheme) => {
    localStorage.setItem(STORAGE_KEY, newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  })

  return {
    theme,
    toggleTheme,
  }
}
