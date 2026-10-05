// Render canvas_attention.html to an MP4, frame by frame, at a fixed timestep.
//
// Drives headless Chrome over the DevTools protocol, steps the page's own clock
// exactly 1/FPS per frame through window.__canvasAttention, composites the stage,
// the overlay sentence and the simulator inset into one image, and pipes PNG frames
// into ffmpeg (libx264).
//
// Usage (from presentation/animations, with `python3 -m http.server 8765` running there):
//   FFMPEG=/path/to/ffmpeg node canvas_attention/pipeline/export_mp4.mjs [out.mp4]
// Env: FPS (30), WIDTH (1920), HEIGHT (1080), SECONDS (one full loop), PORT (8765).

import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const FPS = +(process.env.FPS || 30);
const WIDTH = +(process.env.WIDTH || 1920), HEIGHT = +(process.env.HEIGHT || 1080);
const PORT = +(process.env.PORT || 8765), DBG = 9333;
const OUT = process.argv[2] || "canvas_attention.mp4";
const FFMPEG = process.env.FFMPEG || "ffmpeg";
const CHROME = process.env.CHROME || "google-chrome";
const sleep = ms => new Promise(r => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  "--headless=new", `--remote-debugging-port=${DBG}`, `--window-size=${WIDTH},${HEIGHT}`,
  "--force-device-scale-factor=1", "--hide-scrollbars", "--disable-gpu",
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "ca-export-"))}`, "about:blank",
], { stdio: "ignore" });

let target;
for (let k = 0; k < 50 && !target; k++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${DBG}/json/list`)).json()).find(t => t.type === "page"); } catch {}
}
if (!target) throw new Error("Chrome did not expose a page target");

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener("open", r, { once: true }));
let seq = 0; const pending = new Map();
ws.addEventListener("message", ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
});
const send = (method, params = {}) => new Promise(r => { const id = ++seq; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const m = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (m.result?.exceptionDetails) throw new Error(JSON.stringify(m.result.exceptionDetails).slice(0, 500));
  return m.result?.result?.value;
};

await send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1, mobile: false });
await send("Page.enable");
await send("Page.navigate", { url: `http://127.0.0.1:${PORT}/canvas_attention.html` });
for (let k = 0; ; k++) {
  await sleep(300);
  if (await evaluate("!!(window.__canvasAttention && window.__canvasAttention.ready())")) break;
  if (k > 200) throw new Error("page never became ready");
}
await evaluate("document.fonts.ready.then(() => true)");
const total = +(process.env.SECONDS || await evaluate("window.__canvasAttention.total()"));
const frames = Math.round(total * FPS);
await evaluate("window.__canvasAttention.reset()");

// one composited frame: stage canvas + simulator inset + the overlay sentence
const FRAME_JS = `(() => {
  const A = window.__canvasAttention; A.step(${1 / FPS});
  const out = window.__exportCanvas || (window.__exportCanvas = Object.assign(document.createElement("canvas"), { width: ${WIDTH}, height: ${HEIGHT} }));
  const c = out.getContext("2d");
  c.globalAlpha = 1; c.fillStyle = "#fff"; c.fillRect(0, 0, out.width, out.height);
  const st = document.getElementById("stage"), sr = st.getBoundingClientRect();
  c.drawImage(st, sr.left, sr.top, sr.width, sr.height);
  const sim = document.querySelector(".sim");
  c.globalAlpha = +getComputedStyle(sim).opacity;
  for (const id of ["top", "ego"]) {
    const e = document.getElementById(id), r = e.getBoundingClientRect();
    c.drawImage(e, r.left, r.top, r.width, r.height);
    if (id === "top") { c.strokeStyle = "#141414"; c.lineWidth = 1; c.strokeRect(r.left + 0.5, r.top + 0.5, r.width - 1, r.height - 1); }
  }
  // the sentence, wrapped from the DOM's own line boxes so it matches the page
  const el = document.getElementById("line"), cs = getComputedStyle(el);
  c.globalAlpha = +cs.opacity; c.fillStyle = "#141414"; c.textBaseline = "alphabetic";
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const ps = getComputedStyle(n.parentElement);
    c.font = ps.fontStyle + " " + ps.fontWeight + " " + ps.fontSize + " " + ps.fontFamily;
    const re = /\\S+\\s*/g; let m;
    while ((m = re.exec(n.data))) {
      range.setStart(n, m.index); range.setEnd(n, m.index + m[0].trimEnd().length);
      const rs = range.getClientRects(); if (!rs.length) continue;
      const r = rs[0];
      c.fillText(m[0].trimEnd(), r.left, r.bottom - parseFloat(ps.fontSize) * 0.24);
    }
  }
  return out.toDataURL("image/png");
})()`;

const ff = spawn(FFMPEG, [
  "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
  "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
  "-movflags", "+faststart", OUT,
], { stdio: ["pipe", "inherit", "inherit"] });

const t0 = Date.now();
for (let k = 0; k < frames; k++) {
  const url = await evaluate(FRAME_JS);
  const buf = Buffer.from(url.slice(url.indexOf(",") + 1), "base64");
  if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once("drain", r));
  if (k % (FPS * 5) === 0) console.log(`frame ${k}/${frames}  (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
ff.stdin.end();
await new Promise(r => ff.on("close", r));
ws.close(); chrome.kill();
console.log(`wrote ${OUT}: ${frames} frames, ${total.toFixed(1)} s at ${FPS} fps`);
