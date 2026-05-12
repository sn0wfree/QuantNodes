import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'

export const useAppStore = defineStore('app', () => {
  const sidebarCollapsed = ref(false)
  const chatSidebarCollapsed = ref(true)
  const theme = ref<'light' | 'dark'>(
    (localStorage.getItem('quantnodes-theme') as 'light' | 'dark') || 'light'
  )
  const locale = ref(localStorage.getItem('quantnodes-locale') || 'en')

  const isDarkMode = computed(() => theme.value === 'dark')

  watch(theme, (newTheme) => {
    localStorage.setItem('quantnodes-theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }, { immediate: true })

  const toggleSidebar = () => {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  const toggleChatSidebar = () => {
    chatSidebarCollapsed.value = !chatSidebarCollapsed.value
  }

  const setTheme = (newTheme: 'light' | 'dark') => {
    theme.value = newTheme
    localStorage.setItem('quantnodes-theme', newTheme)
  }

  const setLocale = (newLocale: string) => {
    locale.value = newLocale
    localStorage.setItem('quantnodes-locale', newLocale)
  }

  return {
    sidebarCollapsed,
    chatSidebarCollapsed,
    theme,
    locale,
    isDarkMode,
    toggleSidebar,
    toggleChatSidebar,
    setTheme,
    setLocale,
  }
})
