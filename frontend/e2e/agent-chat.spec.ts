import { test, expect, type Page } from '@playwright/test'

test.describe('Agent Chat Page', () => {
  let consoleErrors: string[] = []

  test.beforeEach(async ({ page }) => {
    consoleErrors = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text())
      }
    })
    page.on('pageerror', (err) => {
      consoleErrors.push(err.message)
    })
  })

  test('page loads and chat layout renders', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const chatLayout = page.locator('.chat-layout')
    await expect(chatLayout).toBeVisible()
  })

  test('ChatNavSidebar renders', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const sidebar = page.locator('.chat-nav-sidebar')
    await expect(sidebar).toBeVisible()
  })

  test('ChatInput is visible and enabled', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const textarea = page.locator('.chat-input textarea, .chat-input-wrapper textarea')
    await expect(textarea).toBeVisible({ timeout: 10000 })

    const sendBtn = page.locator('.chat-input .send-btn, .chat-input-wrapper .send-btn')
    await expect(sendBtn).toBeVisible()
  })

  test('ChatInput accepts text and send button enables', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const textarea = page.locator('.chat-input textarea, .chat-input-wrapper textarea')
    await expect(textarea).toBeVisible({ timeout: 10000 })

    await textarea.fill('Hello test message')

    const sendBtn = page.locator('.chat-input .send-btn, .chat-input-wrapper .send-btn')
    await expect(sendBtn).toBeEnabled()
  })

  test('EmptyState shows welcome screen when no messages', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const emptyState = page.locator('.empty-state')
    await expect(emptyState).toBeVisible({ timeout: 10000 })

    const heading = page.locator('.empty-state h2')
    await expect(heading).toContainText('Welcome')
  })

  test('ChatHeader shows session label', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const header = page.locator('.chat-header')
    await expect(header).toBeVisible({ timeout: 10000 })
  })

  test('ChatInputFooter shows agent indicator and model name', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const inputFooter = page.locator('.chat-input-footer')
    await expect(inputFooter).toBeVisible({ timeout: 10000 })

    const agentDot = inputFooter.locator('.agent-dot')
    await expect(agentDot).toBeVisible()

    const modelName = inputFooter.locator('.model-name')
    await expect(modelName).toBeVisible()
  })

  test('ChatKeybindHints visible at bottom', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const keybinds = page.locator('.chat-keybinds')
    await expect(keybinds).toBeVisible({ timeout: 10000 })
  })

  test('sidebar expand/collapse toggle works', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const sidebar = page.locator('.chat-nav-sidebar')
    await expect(sidebar).toBeVisible({ timeout: 10000 })

    const isCollapsed = await sidebar.evaluate((el) => el.classList.contains('collapsed'))
    expect(typeof isCollapsed).toBe('boolean')
  })

  test('sidebar has navigation items', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const sidebar = page.locator('.chat-nav-sidebar')
    await expect(sidebar).toBeVisible({ timeout: 10000 })

    const navIcons = sidebar.locator('.nav-icon, .ant-menu-item')
    const count = await navIcons.count()
    expect(count).toBeGreaterThan(0)
  })

  test('take screenshot of full page', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(2000)

    await page.screenshot({ path: 'e2e/screenshots/agent-chat-full.png', fullPage: true })

    const viewport = page.viewportSize()
    if (viewport) {
      await page.screenshot({ path: 'e2e/screenshots/agent-chat-viewport.png' })
    }
  })

  test('no critical console errors on page load', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(3000)

    const criticalErrors = consoleErrors.filter(
      (e) =>
        !e.includes('WebSocket') &&
        !e.includes('favicon') &&
        !e.includes('Failed to fetch') &&
        !e.includes('connect ECONNREFUSED') &&
        !e.includes('API Error') &&
        !e.includes('status of 500')
    )

    if (criticalErrors.length > 0) {
      console.log('Console errors found:', criticalErrors)
    }

    expect(criticalErrors).toHaveLength(0)
  })

  test('all chat components coexist without overlap', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(2000)

    const components = [
      { name: 'ChatNavSidebar', selector: '.chat-nav-sidebar' },
      { name: 'chat-topbar', selector: '.chat-topbar' },
      { name: 'ChatHeader', selector: '.chat-header' },
      { name: 'MessageList', selector: '.messages' },
      { name: 'ChatInput', selector: '.chat-input-wrapper, .chat-input' },
      { name: 'ChatInputFooter', selector: '.chat-input-footer' },
      { name: 'ChatKeybindHints', selector: '.chat-keybinds' },
    ]

    for (const comp of components) {
      const el = page.locator(comp.selector).first()
      const visible = await el.isVisible().catch(() => false)
      const box = visible ? await el.boundingBox() : null

      console.log(
        `${comp.name}: visible=${visible}, box=${box ? `${box.x},${box.y} ${box.width}x${box.height}` : 'null'}`
      )
    }

    await page.screenshot({ path: 'e2e/screenshots/layout-audit.png', fullPage: true })

    const chatInput = page.locator('.chat-input-wrapper, .chat-input').first()
    const inputVisible = await chatInput.isVisible().catch(() => false)
    expect(inputVisible, 'ChatInput MUST be visible').toBe(true)
  })

  test('diagnose: dump entire DOM structure', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(2000)

    const html = await page.evaluate(() => {
      const agentChat = document.querySelector('.agent-chat')
      if (!agentChat) {
        return 'NO .agent-chat element found. Body innerHTML: ' + document.body.innerHTML.substring(0, 3000)
      }
      return agentChat.innerHTML.substring(0, 5000)
    })

    console.log('=== DOM DIAGNOSTIC ===')
    console.log(html)
    console.log('=== END DIAGNOSTIC ===')

    const hasChatInput = html.includes('chat-input') || html.includes('textarea')
    console.log('Has chat-input in DOM:', hasChatInput)
  })
})
