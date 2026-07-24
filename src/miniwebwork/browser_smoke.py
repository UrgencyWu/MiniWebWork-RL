"""
Playwright browser smoke test for MiniWebWork-RL.

Performs:
  1. Launch headless Chromium
  2. Open local web app
  3. Verify page title and initial state
  4. Type into input field
  5. Click button
  6. Verify result text changed
  7. Read DOM information
  8. Save JSON result and screenshot
"""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("MINIWEBWORK_URL", "http://127.0.0.1:18080") + "/smoke"
ARTIFACTS_DIR = os.environ.get(
    "MINIWEBWORK_ARTIFACTS",
    str(Path(__file__).resolve().parent.parent.parent / "artifacts"),
)


def main():
    result = {
        "success": False,
        "url": "",
        "title": "",
        "input_count": 0,
        "button_count": 0,
        "result_text": "",
        "browser": "chromium",
        "headless": True,
    }

    exit_code = 0
    artifacts = Path(ARTIFACTS_DIR)
    artifacts.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            print(f"Chromium executable: {p.chromium.executable_path}")

            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )

            page = browser.new_page()

            # Open the local web app
            page.goto(BASE_URL, timeout=10000)
            result["url"] = page.url

            # Verify title
            result["title"] = page.title()
            print(f"Page title: {result['title']}")

            # Verify input and button counts
            result["input_count"] = page.locator("input").count()
            result["button_count"] = page.locator("button").count()
            print(f"Input count: {result['input_count']}")
            print(f"Button count: {result['button_count']}")

            # Verify initial result text
            initial_result = page.locator("#result").inner_text()
            print(f"Initial #result text: {initial_result}")
            assert initial_result == "ready", \
                f"Expected 'ready', got '{initial_result}'"

            # Type into input and click button
            page.locator("#query").fill("RTX PRO 6000")
            page.locator("#search-button").click()

            # Wait for DOM update (synchronous in this case)
            time.sleep(0.1)

            # Verify result changed
            result["result_text"] = page.locator("#result").inner_text()
            print(f"Updated #result text: {result['result_text']}")
            assert result["result_text"] == "searched: RTX PRO 6000", \
                f"Expected 'searched: RTX PRO 6000', got '{result['result_text']}'"

            # Save screenshot
            screenshot_path = artifacts / "browser_smoke.png"
            page.screenshot(path=str(screenshot_path))
            print(f"Screenshot saved: {screenshot_path}")

            result["success"] = True
            browser.close()
            print("Browser smoke test PASSED")

    except Exception as e:
        print(f"Browser smoke test FAILED: {e}")
        result["error"] = str(e)
        exit_code = 1

    finally:
        # Save JSON result
        json_path = artifacts / "browser_smoke_result.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved: {json_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
