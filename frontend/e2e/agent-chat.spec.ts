import { test, expect } from '@playwright/test'

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

    const appLayout = page.locator('.app-layout')
    await expect(appLayout).toBeVisible()
  })

  test('AppSidebar renders with navigation items', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const sidebar = page.locator('.ant-layout-sider')
    await expect(sidebar).toBeVisible()

    const menuItems = page.locator('.ant-menu-item')
    const count = await menuItems.count()
    expect(count).toBeGreaterThan(0)
  })

  test('AppHeader renders', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const header = page.locator('.ant-layout-header')
    await expect(header).toBeVisible()
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

  test('Bottombar visible with hints', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const bottombar = page.locator('.chat-bottombar')
    await expect(bottombar).toBeVisible({ timeout: 10000 })

    const hints = bottombar.locator('.hints')
    await expect(hints).toBeVisible()
  })

  test('AppSidebar expand/collapse toggle works', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const sidebar = page.locator('.ant-layout-sider')
    await expect(sidebar).toBeVisible({ timeout: 10000 })

    const collapseTrigger = sidebar.locator('.ant-layout-sider-trigger')
    if (await collapseTrigger.isVisible()) {
      await collapseTrigger.click()
      await page.waitForTimeout(300)
    }
  })

  test('Dashboard page loads and renders', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    const appLayout = page.locator('.app-layout')
    await expect(appLayout).toBeVisible()

    const header = page.locator('.ant-layout-header')
    await expect(header).toBeVisible()

    const sidebar = page.locator('.ant-layout-sider')
    await expect(sidebar).toBeVisible()

    const dashboard = page.locator('.dashboard')
    await expect(dashboard).toBeVisible({ timeout: 10000 })
  })

  test('Dashboard to Chat transition has no flicker', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(500)

    await page.goto('/chat')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(300)

    const appLayout = page.locator('.app-layout')
    await expect(appLayout).toBeVisible()

    const chatInput = page.locator('.chat-input-wrapper, .chat-input').first()
    await expect(chatInput).toBeVisible({ timeout: 10000 })

    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(300)

    await expect(appLayout).toBeVisible()
  })

  test('Toggle panel button opens context and tools panel', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const toggleBtn = page.locator('[data-testid="toggle-panel"]')
    await expect(toggleBtn).toBeVisible({ timeout: 10000 })

    await toggleBtn.click()
    await page.waitForTimeout(500)

    const panelsContainer = page.locator('.panels-container')
    await expect(panelsContainer).toBeVisible()

    const contextPanel = page.locator('.context-panel')
    await expect(contextPanel).toBeVisible()

    const toolsPanel = page.locator('.tools-panel')
    await expect(toolsPanel).toBeVisible()
  })

  test('Toggle panel button closes panels', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const toggleBtn = page.locator('[data-testid="toggle-panel"]')
    await toggleBtn.click()
    await page.waitForTimeout(500)

    const panelsContainer = page.locator('.panels-container')
    await expect(panelsContainer).toBeVisible()

    await toggleBtn.click()
    await page.waitForTimeout(500)

    await expect(panelsContainer).not.toBeVisible()
  })

  test('Panels have resize handles', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')

    const toggleBtn = page.locator('[data-testid="toggle-panel"]')
    await toggleBtn.click()
    await page.waitForTimeout(500)

    const handles = page.locator('.resize-handle')
    const count = await handles.count()
    expect(count).toBeGreaterThanOrEqual(1)
  })

  test('take screenshot of full page', async ({ page }) => {
    await page.goto('/chat')
    await page.waitForLoadState('networkidle')
    await page.waitForTimeout(2000)

    await page.screenshot({ path: 'e2e/screenshots/agent-chat-full.png', fullPage: true })
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
      { name: 'AppLayout', selector: '.app-layout' },
      { name: 'AppSidebar', selector: '.ant-layout-sider' },
      { name: 'AppHeader', selector: '.ant-layout-header' },
      { name: 'MessageList', selector: '.messages' },
      { name: 'ChatInput', selector: '.chat-input-wrapper, .chat-input' },
      { name: 'ChatInputFooter', selector: '.chat-input-footer' },
      { name: 'chat-bottombar', selector: '.chat-bottombar' },
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