// Verify the real browser viewer. Install Playwright outside the project environment.
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.BATTLE_PLAYWRIGHT_MODULE || "playwright");

async function main() {
  const [url, output, frameCount = "300"] = process.argv.slice(2);
  if (!url || !output) throw new Error("Usage: node check_pipette_demo_browser.cjs URL OUTPUT");
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.BATTLE_CHROME || "/usr/bin/google-chrome",
    headless: true,
    args: ["--no-sandbox", ...(process.env.BATTLE_PUBLIC_IP ?
      [`--host-resolver-rules=MAP ${new URL(url).hostname} ${process.env.BATTLE_PUBLIC_IP}`] : [])],
  });
  const errors = [];
  const consoleMessages = [];
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) consoleMessages.push(message.text());
  });
  try {
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.locator("canvas").waitFor({ state: "visible", timeout: 240000 });
    await page.waitForFunction(() => window._handle &&
      window._handle.get_active_recording_id() && !window._handle.has_panicked(),
      null, { timeout: 240000 });
    await page.waitForFunction((last) => {
      const h = window._handle, id = h.get_active_recording_id();
      return h.get_timeline_time_range(id, "raw_frame")?.max === last;
    }, Number(frameCount) - 1, { timeout: 240000 });
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(output, "browser-initial.png") });
    const environment = await page.evaluate(() => ({
      secureContext: window.isSecureContext,
      webgpu: Boolean(navigator.gpu),
      canvasCount: document.querySelectorAll("canvas").length,
      title: document.title,
      recordingId: window._handle.get_active_recording_id(),
      viewerKeys: Object.keys(window).filter((key) => /rerun|viewer/i.test(key)),
    }));
    const range = await page.evaluate(() => {
      const h = window._handle;
      return h.get_timeline_time_range(h.get_active_recording_id(), "raw_frame");
    });
    const seekResults = [];
    const frames = [...new Set([0, 100, 200, 599, 600, Number(frameCount) - 1])]
      .filter((frame) => frame < Number(frameCount)).sort((a, b) => a - b);
    for (const frame of frames) {
      await page.evaluate((frame) => {
        const h = window._handle;
        const id = h.get_active_recording_id();
        h.set_playing(id, false);
        h.set_active_timeline(id, "raw_frame");
        h.set_time_for_timeline(id, "raw_frame", frame);
      }, frame);
      await page.waitForTimeout(1000);
      const time = await page.evaluate(() => {
        const h = window._handle;
        return h.get_time_for_timeline(h.get_active_recording_id(), "raw_frame");
      });
      await page.screenshot({ path: path.join(output, `browser-${frame}.png`) });
      seekResults.push({ requested: frame, observed: time });
    }
    await page.evaluate(() => {
      const h = window._handle, id = h.get_active_recording_id();
      h.set_time_for_timeline(id, "raw_frame", 100);
      h.set_playing(id, true);
    });
    await page.waitForTimeout(2000);
    const playbackTime = await page.evaluate(() => {
      const h = window._handle, id = h.get_active_recording_id();
      h.set_playing(id, false);
      return h.get_time_for_timeline(id, "raw_frame");
    });
    await page.screenshot({ path: path.join(output, "browser-playback.png") });
    await page.reload({ waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window._handle &&
      window._handle.get_active_recording_id() && !window._handle.has_panicked(),
      null, { timeout: 240000 });
    await page.waitForFunction((last) => {
      const h = window._handle, id = h.get_active_recording_id();
      return h.get_timeline_time_range(id, "raw_frame")?.max === last;
    }, Number(frameCount) - 1, { timeout: 240000 });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(output, "browser-reconnect.png") });
    await page.mouse.click(130, 40);
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(output, "browser-cameras.png") });
    await page.evaluate(() => {
      const h = window._handle, id = h.get_active_recording_id();
      h.set_playing(id, false);
      h.set_active_timeline(id, "raw_frame");
      h.set_time_for_timeline(id, "raw_frame", 100);
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(output, "browser-cameras-100.png") });
    await page.mouse.click(230, 40);
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(output, "browser-evidence.png") });
    const report = { url, publicRelayIp: process.env.BATTLE_PUBLIC_IP || null,
      environment, range, seekResults, playbackTime, errors, consoleMessages,
      checks: "Screenshots require direct visual review; canvas presence alone is not success" };
    fs.writeFileSync(path.join(output, "browser-check.json"), JSON.stringify(report, null, 2) + "\n");
    console.log(JSON.stringify(report, null, 2));
    if (errors.length) throw new Error("Browser reported JavaScript errors");
    if (!(playbackTime > 100)) throw new Error("Playback did not advance");
    if (seekResults.some(({requested, observed}) => requested !== observed)) {
      throw new Error("Seek did not reach the requested native frame");
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
