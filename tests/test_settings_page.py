"""
Playwright test suite for Settings page
Tests all 6 tabs + global actions (save, export, import, reset)
Target: http://10.67.10.50:18380/settings
"""

import json
import os
from playwright.sync_api import sync_playwright, Page

BASE_URL = "http://10.67.10.50:18380"
SETTINGS_URL = f"{BASE_URL}/settings"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

results = []


def log_result(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({"test": test_name, "status": status, "detail": detail})
    print(f"  [{status}] {test_name}" + (f" - {detail}" if detail else ""))


def screenshot(page: Page, name: str):
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    page.screenshot(path=path, full_page=True)
    return path


def wait_for_toast(page: Page, timeout_ms: int = 5000):
    try:
        toast = page.locator(".ant-message-notice").first
        toast.wait_for(state="visible", timeout=timeout_ms)
        return toast.inner_text()
    except Exception:
        return None


def dismiss_toasts(page: Page):
    page.evaluate("document.querySelectorAll('.ant-message-notice').forEach(e => e.remove())")


def get_form_item_input(page: Page, label_text: str):
    """Find input inside ant-form-item by label text."""
    return page.locator(
        f".ant-form-item:has(.ant-form-item-label label:text-is('{label_text}')) input"
    ).first


def get_form_item_switch(page: Page, label_text: str):
    """Find switch inside ant-form-item by label text."""
    return page.locator(
        f".ant-form-item:has(.ant-form-item-label label:text-is('{label_text}')) .ant-switch"
    ).first


def get_form_item_select(page: Page, label_text: str):
    """Find select inside ant-form-item by label text."""
    return page.locator(
        f".ant-form-item:has(.ant-form-item-label label:text-is('{label_text}')) .ant-select"
    ).first


def get_form_item_slider(page: Page, label_text: str):
    """Find slider inside ant-form-item by label text."""
    return page.locator(
        f".ant-form-item:has(.ant-form-item-label label:text-is('{label_text}')) .ant-slider"
    ).first


def get_form_item_radio_group(page: Page, label_text: str):
    """Find radio group inside ant-form-item by label text."""
    return page.locator(
        f".ant-form-item:has(.ant-form-item-label label:text-is('{label_text}')) .ant-radio-group"
    ).first


def run_tests():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        print("\n=== Settings Page Playwright Tests ===\n")

        # ── Test 1: Page Load ──
        print("[1] Page load")
        try:
            page.goto(SETTINGS_URL, wait_until="networkidle", timeout=15000)
            title = page.locator("text=Settings").first
            assert title.is_visible(), "Settings title not visible"

            tabs = ["Appearance", "API Connection", "Agent", "Editor", "Backtest", "Notifications"]
            for tab_name in tabs:
                tab = page.locator(f".ant-tabs-tab:has-text('{tab_name}')")
                assert tab.count() > 0, f"Tab '{tab_name}' not found"

            log_result("Page load", True)
        except Exception as e:
            screenshot(page, "01_page_load_fail")
            log_result("Page load", False, str(e))

        # ── Test 2: Tab Navigation ──
        print("[2] Tab navigation")
        try:
            for tab_name in ["API Connection", "Agent", "Editor", "Backtest", "Notifications", "Appearance"]:
                tab = page.locator(f".ant-tabs-tab:has-text('{tab_name}')")
                tab.click()
                page.wait_for_timeout(300)
                active = page.locator(f".ant-tabs-tab-active:has-text('{tab_name}')")
                assert active.count() > 0, f"Tab '{tab_name}' not active after click"
            log_result("Tab navigation", True)
        except Exception as e:
            screenshot(page, "02_tab_nav_fail")
            log_result("Tab navigation", False, str(e))

        # ── Test 3: API Connection Tab ──
        print("[3] API Connection tab")
        try:
            page.locator(".ant-tabs-tab:has-text('API Connection')").click()
            page.wait_for_timeout(300)

            base_url_input = get_form_item_input(page, "API Base URL")
            assert base_url_input.is_visible(), "API Base URL input not visible"
            base_val = base_url_input.input_value()
            print(f"    API Base URL default: {base_val}")

            ws_input = get_form_item_input(page, "WebSocket URL")
            assert ws_input.is_visible(), "WebSocket URL input not visible"

            timeout_input = get_form_item_input(page, "Request Timeout (ms)")
            assert timeout_input.is_visible(), "Timeout input not visible"

            test_btn = page.locator("button:has-text('Test Connection')")
            assert test_btn.is_visible(), "Test Connection button not visible"
            dismiss_toasts(page)
            test_btn.click()
            toast = wait_for_toast(page)
            print(f"    Test Connection toast: {toast}")
            assert toast and "success" in toast.lower(), f"Expected success toast, got: {toast}"

            log_result("API Connection tab", True)
        except Exception as e:
            screenshot(page, "03_api_conn_fail")
            log_result("API Connection tab", False, str(e))

        # ── Test 4: Agent Tab ──
        print("[4] Agent tab")
        try:
            page.locator(".ant-tabs-tab:has-text('Agent')").click()
            page.wait_for_timeout(300)

            provider_select = get_form_item_select(page, "LLM Provider")
            assert provider_select.is_visible(), "LLM Provider select not visible"
            provider_select.click()
            page.wait_for_timeout(200)
            options = page.locator(".ant-select-dropdown .ant-select-item")
            assert options.count() >= 5, f"Expected >= 5 provider options, got {options.count()}"
            # Verify Custom option exists
            custom_option = page.locator(".ant-select-dropdown .ant-select-item:has-text('Custom (OpenAI-Compatible)')")
            assert custom_option.count() > 0, "Custom (OpenAI-Compatible) option not found"
            page.locator(".ant-select-dropdown .ant-select-item:has-text('Anthropic')").click()
            page.wait_for_timeout(200)

            model_input = get_form_item_input(page, "Model")
            assert model_input.is_visible(), "Model input not visible"
            model_val = model_input.input_value()
            print(f"    Model default: {model_val}")

            api_key_input = get_form_item_input(page, "API Key")
            assert api_key_input.is_visible(), "API Key input not visible"
            api_key_input.fill("sk-test1234567890abcdef")
            page.wait_for_timeout(100)

            api_base = get_form_item_input(page, "API Base URL (optional)")
            assert api_base.is_visible(), "API Base URL (optional) not visible"

            max_iter = get_form_item_slider(page, "Max Iterations")
            assert max_iter.is_visible(), "Max Iterations slider not visible"

            temp = get_form_item_slider(page, "Temperature")
            assert temp.is_visible(), "Temperature slider not visible"

            llm_timeout = get_form_item_input(page, "LLM Request Timeout (seconds)")
            assert llm_timeout.is_visible(), "LLM Request Timeout input not visible"
            timeout_val = llm_timeout.input_value()
            print(f"    LLM Timeout default: {timeout_val}")

            max_retries = get_form_item_input(page, "Max Retries")
            assert max_retries.is_visible(), "Max Retries input not visible"
            retries_val = max_retries.input_value()
            print(f"    Max Retries default: {retries_val}")

            log_result("Agent tab", True)
        except Exception as e:
            screenshot(page, "04_agent_fail")
            log_result("Agent tab", False, str(e))

        # ── Test 5: Editor Tab ──
        print("[5] Editor tab")
        try:
            page.locator(".ant-tabs-tab:has-text('Editor')").click()
            page.wait_for_timeout(300)

            font_input = get_form_item_input(page, "Font Size")
            assert font_input.is_visible(), "Font Size input not visible"
            font_input.fill("16")
            page.wait_for_timeout(100)
            assert font_input.input_value() == "16", f"Font size not updated, got: {font_input.input_value()}"

            tab_size_group = get_form_item_radio_group(page, "Tab Size")
            assert tab_size_group.is_visible(), "Tab Size radio group not visible"
            tab_4 = page.locator(
                ".ant-form-item:has(.ant-form-item-label label:text-is('Tab Size')) label:has-text('4')"
            )
            tab_4.click()
            page.wait_for_timeout(100)

            word_wrap = get_form_item_switch(page, "Word Wrap")
            assert word_wrap.is_visible(), "Word Wrap switch not visible"
            initial_wrap = word_wrap.get_attribute("aria-checked")
            word_wrap.click()
            page.wait_for_timeout(100)
            new_wrap = word_wrap.get_attribute("aria-checked")
            assert initial_wrap != new_wrap, f"Word Wrap toggle failed: {initial_wrap} -> {new_wrap}"
            word_wrap.click()
            page.wait_for_timeout(100)

            minimap = get_form_item_switch(page, "Minimap")
            assert minimap.is_visible(), "Minimap switch not visible"

            auto_save = get_form_item_switch(page, "Auto Save")
            assert auto_save.is_visible(), "Auto Save switch not visible"

            log_result("Editor tab", True)
        except Exception as e:
            screenshot(page, "05_editor_fail")
            log_result("Editor tab", False, str(e))

        # ── Test 6: Backtest Tab ──
        print("[6] Backtest tab")
        try:
            page.locator(".ant-tabs-tab:has-text('Backtest')").click()
            page.wait_for_timeout(300)

            cash = get_form_item_input(page, "Default Initial Cash")
            assert cash.is_visible(), "Default Initial Cash input not visible"
            cash_val = cash.input_value()
            print(f"    Default cash: {cash_val}")

            commission = get_form_item_input(page, "Default Commission Rate")
            assert commission.is_visible(), "Default Commission Rate input not visible"
            commission_val = commission.input_value()
            print(f"    Default commission: {commission_val}")

            auto_save = get_form_item_switch(page, "Auto Save Results")
            assert auto_save.is_visible(), "Auto Save Results switch not visible"

            log_result("Backtest tab", True)
        except Exception as e:
            screenshot(page, "06_backtest_fail")
            log_result("Backtest tab", False, str(e))

        # ── Test 7: Notifications Tab ──
        print("[7] Notifications tab")
        try:
            page.locator(".ant-tabs-tab:has-text('Notifications')").click()
            page.wait_for_timeout(300)

            enable = get_form_item_switch(page, "Enable Notifications")
            assert enable.is_visible(), "Enable Notifications switch not visible"

            sound = get_form_item_switch(page, "Sound")
            assert sound.is_visible(), "Sound switch not visible"

            desktop = get_form_item_switch(page, "Desktop Notifications")
            assert desktop.is_visible(), "Desktop Notifications switch not visible"

            enable_state = enable.get_attribute("aria-checked")
            if enable_state == "true":
                enable.click()
                page.wait_for_timeout(300)
                sound_cls = sound.get_attribute("class") or ""
                desktop_cls = desktop.get_attribute("class") or ""
                sound_disabled = "ant-switch-disabled" in sound_cls
                desktop_disabled = "ant-switch-disabled" in desktop_cls
                print(f"    After disable: sound_disabled={sound_disabled}, desktop_disabled={desktop_disabled}")
                assert sound_disabled, f"Sound not disabled when notifications off: class={sound_cls}"
                assert desktop_disabled, f"Desktop not disabled when notifications off: class={desktop_cls}"

                enable.click()
                page.wait_for_timeout(300)
                sound_cls2 = sound.get_attribute("class") or ""
                desktop_cls2 = desktop.get_attribute("class") or ""
                assert "ant-switch-disabled" not in sound_cls2, f"Sound still disabled after re-enable: {sound_cls2}"
                assert "ant-switch-disabled" not in desktop_cls2, f"Desktop still disabled after re-enable: {desktop_cls2}"

            log_result("Notifications tab (cascading)", True)
        except Exception as e:
            screenshot(page, "07_notifications_fail")
            log_result("Notifications tab (cascading)", False, str(e))

        # ── Test 8: Appearance Tab ──
        print("[8] Appearance tab")
        try:
            page.locator(".ant-tabs-tab:has-text('Appearance')").click()
            page.wait_for_timeout(300)

            dark_btn = page.locator("label:has-text('Dark')")
            assert dark_btn.is_visible(), "Dark theme button not visible"
            dark_btn.click()
            page.wait_for_timeout(500)

            theme_attr = page.evaluate("document.documentElement.getAttribute('data-theme')")
            print(f"    data-theme after Dark click: {theme_attr}")

            light_btn = page.locator("label:has-text('Light')")
            light_btn.click()
            page.wait_for_timeout(500)
            theme_attr2 = page.evaluate("document.documentElement.getAttribute('data-theme')")
            print(f"    data-theme after Light click: {theme_attr2}")

            lang = get_form_item_select(page, "Language")
            assert lang.is_visible(), "Language selector not visible"

            compact = get_form_item_switch(page, "Compact Mode")
            assert compact.is_visible(), "Compact Mode switch not visible"

            sidebar = get_form_item_switch(page, "Sidebar Collapsed by Default")
            assert sidebar.is_visible(), "Sidebar Collapsed switch not visible"

            log_result("Appearance tab", True)
        except Exception as e:
            screenshot(page, "08_appearance_fail")
            log_result("Appearance tab", False, str(e))

        # ── Test 9: Save Settings ──
        print("[9] Save settings")
        try:
            dismiss_toasts(page)
            save_btn = page.locator("button:has-text('Save Settings')")
            assert save_btn.is_visible(), "Save Settings button not visible"
            save_btn.click()
            toast = wait_for_toast(page)
            print(f"    Save toast: {toast}")
            assert toast and "saved" in toast.lower(), f"Expected save success toast, got: {toast}"
            log_result("Save settings", True)
        except Exception as e:
            screenshot(page, "09_save_fail")
            log_result("Save settings", False, str(e))

        # ── Test 10: Export Settings ──
        print("[10] Export settings")
        try:
            dismiss_toasts(page)
            with page.expect_download(timeout=5000) as download_info:
                export_btn = page.locator("button:has-text('Export')")
                export_btn.click()
            download = download_info.value
            print(f"    Downloaded: {download.suggested_filename}")
            assert download.suggested_filename == "quantnodes-settings.json", f"Wrong filename: {download.suggested_filename}"
            log_result("Export settings", True)
        except Exception as e:
            screenshot(page, "10_export_fail")
            log_result("Export settings", False, str(e))

        # ── Test 11: Reset Settings ──
        print("[11] Reset settings")
        try:
            dismiss_toasts(page)
            reset_btn = page.locator("button:has-text('Reset')")
            reset_btn.click()
            page.wait_for_timeout(300)

            confirm_btn = page.locator(".ant-popconfirm .ant-btn-primary")
            if confirm_btn.count() > 0:
                confirm_btn.click()
                toast = wait_for_toast(page)
                print(f"    Reset toast: {toast}")
                assert toast and "reset" in toast.lower(), f"Expected reset toast, got: {toast}"
            else:
                page.locator("button:has-text('OK')").click()
                toast = wait_for_toast(page)
                print(f"    Reset toast (alt): {toast}")

            page.locator(".ant-tabs-tab:has-text('Editor')").click()
            page.wait_for_timeout(300)
            font_input = get_form_item_input(page, "Font Size")
            font_val = font_input.input_value()
            print(f"    Font size after reset: {font_val}")

            log_result("Reset settings", True)
        except Exception as e:
            screenshot(page, "11_reset_fail")
            log_result("Reset settings", False, str(e))

        # ── Test 12: Import Settings ──
        print("[12] Import settings")
        try:
            dismiss_toasts(page)
            import_btn = page.locator("button:has-text('Import')")
            import_btn.click()
            page.wait_for_timeout(500)

            modal = page.locator(".ant-modal:has-text('Import Settings')")
            assert modal.is_visible(), "Import modal not visible"

            import_data = json.dumps({"editor": {"font_size": 18, "tab_size": 4}})
            textarea = modal.locator("textarea")
            textarea.fill(import_data)
            page.wait_for_timeout(200)

            modal.locator(".ant-modal-footer .ant-btn-primary").click()
            page.wait_for_timeout(1000)
            toast = wait_for_toast(page)
            print(f"    Import toast: {toast}")
            assert toast and "import" in toast.lower(), f"Expected import toast, got: {toast}"

            page.locator(".ant-tabs-tab:has-text('Editor')").click()
            page.wait_for_timeout(500)
            font_input = get_form_item_input(page, "Font Size")
            font_val = font_input.input_value()
            print(f"    Font size after import: {font_val}")
            assert font_val == "18", f"Expected font_size=18 after import, got: {font_val}"

            log_result("Import settings", True)
        except Exception as e:
            screenshot(page, "12_import_fail")
            log_result("Import settings", False, str(e))

        # ── Console Errors ──
        print("\n[Console Errors]")
        real_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        if real_errors:
            for err in real_errors[:10]:
                print(f"  ERROR: {err}")
        else:
            print("  No console errors detected")

        # ── Summary ──
        print("\n=== Test Summary ===")
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        total = len(results)
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if r["status"] == "FAIL":
                    print(f"  - {r['test']}: {r['detail']}")
            print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")

        # Reset settings back to defaults after test
        try:
            page.goto(SETTINGS_URL, wait_until="networkidle", timeout=10000)
            page.locator("button:has-text('Reset')").click()
            page.wait_for_timeout(300)
            confirm = page.locator(".ant-popconfirm .ant-btn-primary")
            if confirm.count() > 0:
                confirm.click()
                wait_for_toast(page)
        except Exception:
            pass

        browser.close()
        return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
