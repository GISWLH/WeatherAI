// Generate the WeatherAI logo, icon, banner and social preview in an Excalidraw-like hand-drawn style (white background).
//
//   cd scripts/brand && npm install          # roughjs, playwright-core, @excalidraw/excalidraw (for its OFL fonts)
//   node make_assets.mjs [--font path/to/Excalifont-or-Virgil.(ttf|woff2)] [--chromium /path/to/chrome]
//
// Shapes are drawn with rough.js (the library Excalidraw uses) inside headless Chromium, so text is measured with the
// real handwriting font and every box fits its label. Seeds are fixed: re-running gives the same images.
// Writes docs/assets/{logo,icon,banner,social_preview}.png and logo.svg (the logo has no text, so its SVG renders anywhere).
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "..", "..", "docs", "assets");
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };

function defaultFont() {
  try {
    const dir = path.join(path.dirname(require.resolve("@excalidraw/excalidraw/package.json")), "dist", "prod", "fonts", "Virgil");
    return path.join(dir, fs.readdirSync(dir).find((f) => f.endsWith(".woff2")));
  } catch { return null; }
}
const FONT = arg("--font", process.env.WEATHERAI_FONT || defaultFont());
if (!FONT || !fs.existsSync(FONT)) throw new Error("handwriting font not found: npm install, or pass --font <Excalifont/Virgil file>");
const CHROME = arg("--chromium", process.env.CHROMIUM || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome");
const ROUGH = fs.readFileSync(require.resolve("roughjs/bundled/rough.js"), "utf8");
const fontUrl = "data:font/" + (FONT.endsWith(".ttf") ? "ttf" : "woff2") + ";base64," + fs.readFileSync(FONT).toString("base64");

// Runs in the page: draws into <svg id="c"> and returns its markup.
function draw(kind) {
  const INK = "#1e1e1e", BLUE = "#1971c2", ORANGE = "#e8590c";
  const svg = document.getElementById("c");
  const rc = rough.svg(svg);
  const NS = "http://www.w3.org/2000/svg";
  const add = (n) => (svg.appendChild(n), n);
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const bg = document.createElementNS(NS, "rect");
  bg.setAttribute("width", W); bg.setAttribute("height", H); bg.setAttribute("fill", "#ffffff"); add(bg);
  const text = (s, x, y, size, fill = INK, anchor = "start") => {
    const t = document.createElementNS(NS, "text");
    t.textContent = s; t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("font-size", size);
    t.setAttribute("fill", fill); t.setAttribute("text-anchor", anchor); t.setAttribute("font-family", "Hand");
    return add(t);
  };
  const spiral = (cx, cy, rot, r0, r1, sweep, n = 28) => {
    const pts = [], k = Math.log(r1 / r0) / sweep;
    for (let i = 0; i <= n; i++) {
      const a = sweep * i / n, r = r0 * Math.exp(k * a);
      pts.push([cx + r * Math.cos(rot + a), cy - r * Math.sin(rot + a)]);
    }
    return pts;
  };

  // Logo: hatched globe, one meridian + parallels, two-arm cyclone, a few network nodes.
  function logo(ox, oy, s, seed = 7) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("transform", `translate(${ox} ${oy}) scale(${s})`);
    const put = (n) => g.appendChild(n);
    const c = 256, R = 200;
    put(rc.circle(c, c, 2 * R, { seed, stroke: INK, strokeWidth: 3.2, roughness: 1.1, fill: "#a5d8ff", fillStyle: "hachure", hachureGap: 15, hachureAngle: -41, fillWeight: 1.8 }));
    put(rc.ellipse(c, c, 170, 2 * R - 6, { seed: seed + 1, stroke: BLUE, strokeWidth: 2, roughness: 1.2 }));
    for (const [y, bow] of [[c, 18]]) {                                    // equator
      const half = Math.sqrt(R * R - (y - c) ** 2) - 6;
      put(rc.curve([[c - half, y], [c, y + bow], [c + half, y]], { seed: seed + 2 + y, stroke: BLUE, strokeWidth: 2, roughness: 1.1 }));
    }
    const eye = [c + 4, c - 2], rots = [200, 20].map((d) => d * Math.PI / 180);
    const arms = rots.map((r) => spiral(eye[0], eye[1], r, 20, 150, 1.4 * Math.PI));
    arms.forEach((p, i) => put(rc.curve(p, { seed: seed + 20 + i, stroke: INK, strokeWidth: 9, roughness: 0.9 })));
    put(rc.circle(eye[0], eye[1], 58, { seed: seed + 30, stroke: INK, strokeWidth: 5, roughness: 0.8, fill: "#ffffff", fillStyle: "solid" }));
    arms.forEach((p) => [12, 24].forEach((j, k) => put(rc.circle(p[j][0], p[j][1], k ? 36 : 30, {
      seed: seed + 40 + j, stroke: ORANGE, strokeWidth: 3, roughness: 0.9, fill: "#ffc078", fillStyle: "solid" }))));
    return add(g);
  }

  const box = (x, y, label, fill, seed, size = 24) => {
    const t = text(label, 0, 0, size);
    const w = t.getComputedTextLength() + 44, h = size + 30;
    t.setAttribute("x", x + w / 2); t.setAttribute("y", y + h / 2 + size * 0.34); t.setAttribute("text-anchor", "middle");
    svg.insertBefore(rc.rectangle(x, y, w, h, { seed, stroke: INK, strokeWidth: 2, roughness: 1.3, fill, fillStyle: "hachure", hachureGap: 7, fillWeight: 1.6 }), t);
    return w;
  };

  function banner(oy = 0) {
    logo(64, oy + 34, 0.46);
    const title = text("Weather", 330, oy + 150, 112);
    const tw = title.getComputedTextLength();
    const ai = text("AI", 330 + tw + 6, oy + 150, 112, BLUE);
    const aw = ai.getComputedTextLength();
    add(rc.curve([[330 + tw, oy + 168], [330 + tw + aw / 2 + 6, oy + 178], [330 + tw + aw + 18, oy + 166]], { seed: 3, stroke: "#f08c00", strokeWidth: 5, roughness: 1.6 }));
    text("20 peer-reviewed weather & climate AI models,", 334, oy + 214, 31, "#343a40");
    text("one readable PyTorch library.", 334, oy + 254, 31, "#343a40");
    const chips = [["PyTorch-native", "#a5d8ff"], ["JAX · ONNX · CUDA", "#d0bfff"], ["Verified vs official code", "#b2f2bb"], ["Lite models for small GPUs", "#ffec99"]];
    const gap = 18, ws = chips.map(([s, f], i) => box(0, -500, s, f, 11 + i, 21));     // measure, then lay out centred
    [...svg.querySelectorAll("g, text")].slice(-2 * chips.length).forEach((n) => n.remove());
    let x = (W - ws.reduce((a, b) => a + b, 0) - gap * (chips.length - 1)) / 2;
    chips.forEach(([s, f], i) => { box(x, oy + 300, s, f, 11 + i, 21); x += ws[i] + gap; });
  }

  if (kind === "logo") logo(0, 0, 1);
  else if (kind === "icon") logo(56, 56, 0.78);
  else if (kind === "banner") banner();
  else if (kind === "social") { banner(110); text("github.com/GISWLH/WeatherAI", W / 2, 590, 28, "#868e96", "middle"); }
  return svg.outerHTML;
}

const JOBS = { logo: [512, 512, 2], icon: [512, 512, 2], banner: [1280, 400, 2], social: [1280, 640, 1] };
const browser = await chromium.launch({ executablePath: CHROME });
for (const [kind, [w, h, scale]] of Object.entries(JOBS)) {
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: scale });
  await page.setContent(`<!doctype html><html><head><style>@font-face{font-family:Hand;src:url(${fontUrl})}
    html,body{margin:0;background:#fff}</style><script>${ROUGH}</script></head>
    <body><svg id="c" xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"></svg></body></html>`);
  await page.evaluate(() => document.fonts.load("40px Hand").then(() => document.fonts.ready));
  const markup = await page.evaluate(draw, kind);
  const name = kind === "social" ? "social_preview" : kind;
  await page.locator("#c").screenshot({ path: path.join(OUT, `${name}.png`) });
  if (kind === "logo") fs.writeFileSync(path.join(OUT, "logo.svg"), markup + "\n");
  await page.close();
}
await browser.close();
console.log("wrote", fs.readdirSync(OUT).sort().join(", "));
