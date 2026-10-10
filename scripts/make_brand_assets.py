"""Generate the WeatherAI logo / banner (SVG) and their PNG renders.

    python scripts/make_brand_assets.py            # writes docs/assets/*.svg and, if Playwright is installed, *.png

The mark is a globe with a two-arm cyclone whose arms carry network nodes (weather + AI). PNGs are rendered with
headless Chromium so the banner text uses the Inter font regardless of the viewer's system fonts.
"""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")

NAVY, NAVY2, NAVY3 = "#06111F", "#0A2140", "#0D3358"
FONT = "Inter, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"


def _arm_polar(t, r0=34, r1=152, sweep=1.3 * math.pi):
    """Logarithmic spiral: radius and angle offset at arm fraction t in [0, 1]."""
    k = math.log(r1 / r0) / sweep
    a = sweep * t
    return r0 * math.exp(k * a), a


def _spiral_arm(cx, cy, rot, w0=36, w1=3, n=90):
    """Tapered cyclone arm as a closed polygon (outer edge out, inner edge back)."""
    outer, inner = [], []
    for i in range(n + 1):
        t = i / n
        r, a = _arm_polar(t)
        w = (w0 + (w1 - w0) * t ** 0.7) * min(1.0, t / 0.1) ** 0.5 + 0.5   # pointed start, tapered tail
        ca, sa = math.cos(rot + a), math.sin(rot + a)
        outer.append((cx + (r + w / 2) * ca, cy - (r + w / 2) * sa))
        inner.append((cx + (r - w / 2) * ca, cy - (r - w / 2) * sa))
    pts = outer + inner[::-1]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"


def _arm_point(cx, cy, rot, t):
    r, a = _arm_polar(t)
    return cx + r * math.cos(rot + a), cy - r * math.sin(rot + a)


def mark(prefix="m", size=512):
    """Logo mark (no text) in a 512x512 coordinate system; returns (defs, body)."""
    c = 256
    defs = f"""
    <radialGradient id="{prefix}-globe" cx="34%" cy="28%" r="80%">
      <stop offset="0" stop-color="#5EEAD4"/>
      <stop offset="0.35" stop-color="#0EA5E9"/>
      <stop offset="0.75" stop-color="#1D4ED8"/>
      <stop offset="1" stop-color="#172554"/>
    </radialGradient>
    <linearGradient id="{prefix}-arm" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFFFFF"/>
      <stop offset="1" stop-color="#CFFAFE"/>
    </linearGradient>
    <radialGradient id="{prefix}-node" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#FEF3C7"/>
      <stop offset="0.55" stop-color="#FBBF24"/>
      <stop offset="1" stop-color="#F97316"/>
    </radialGradient>
    <clipPath id="{prefix}-clip"><circle cx="{c}" cy="{c}" r="206"/></clipPath>
    <filter id="{prefix}-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>"""
    grat = []
    for k in (-2, -1, 0, 1, 2):                                   # parallels
        y = c + k * 70
        half = math.sqrt(max(206 ** 2 - (y - c) ** 2, 0))
        grat.append(f'<path d="M{c - half:.1f},{y:.1f} Q{c},{y + 16:.1f} {c + half:.1f},{y:.1f}"/>')
    for rx in (206, 150, 78):                                     # meridians
        grat.append(f'<ellipse cx="{c}" cy="{c}" rx="{rx}" ry="206"/>')
    grat.append(f'<line x1="{c}" y1="50" x2="{c}" y2="462"/>')
    ex, ey = c + 4, c - 2                                          # cyclone eye
    rots = (math.radians(200), math.radians(20))
    arms = [_spiral_arm(ex, ey, rot) for rot in rots]
    nodes = [_arm_point(ex, ey, rot, t) for rot in rots for t in (0.42, 0.7, 0.97)]
    ring = sorted(nodes, key=lambda p: math.atan2(p[1] - ey, p[0] - ex))   # hexagonal "mesh" around the storm
    links = [f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>' for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])]
    body = f"""
    <circle cx="{c}" cy="{c}" r="214" fill="none" stroke="#67E8F9" stroke-opacity="0.35" stroke-width="4"/>
    <circle cx="{c}" cy="{c}" r="206" fill="url(#{prefix}-globe)"/>
    <g clip-path="url(#{prefix}-clip)" fill="none" stroke="#FFFFFF" stroke-opacity="0.18" stroke-width="3">{''.join(grat)}</g>
    <g stroke="#FDE68A" stroke-opacity="0.5" stroke-width="3" stroke-dasharray="2 7" stroke-linecap="round">{''.join(links)}</g>
    <g fill="url(#{prefix}-arm)">{''.join(f'<path d="{a}"/>' for a in arms)}</g>
    <circle cx="{ex}" cy="{ey}" r="30" fill="#0B1E3F" stroke="#FFFFFF" stroke-width="11"/>
    <g filter="url(#{prefix}-glow)">{''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{13 if k % 3 == 2 else 10}" fill="url(#{prefix}-node)"/>' for k, (x, y) in enumerate(nodes))}</g>"""
    return defs, body


def logo_svg():
    defs, body = mark("logo")
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512"><defs>{defs}</defs>{body}</svg>\n'


def icon_svg():
    """Rounded-square app icon (dark tile) for avatars / HF / favicons."""
    defs, body = mark("icon")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512"><defs>{defs}
    <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{NAVY3}"/><stop offset="1" stop-color="{NAVY}"/></linearGradient></defs>
    <rect width="512" height="512" rx="112" fill="url(#tile)"/><g transform="translate(36 36) scale(0.86)">{body}</g></svg>\n"""


def _sst_map(x0, y0, w, h, pid):
    """Stylised global anomaly map (ellipse, warm/cold blobs, graticule) for the banner background."""
    blobs = [  # (u, v, radius, colour, opacity) in map-relative coordinates
        (0.60, 0.50, 0.22, "#EF4444", 0.85), (0.80, 0.42, 0.14, "#F97316", 0.7), (0.30, 0.38, 0.16, "#3B82F6", 0.8),
        (0.18, 0.62, 0.13, "#06B6D4", 0.6), (0.45, 0.72, 0.15, "#2563EB", 0.6), (0.72, 0.70, 0.12, "#FB7185", 0.6),
        (0.50, 0.25, 0.12, "#F59E0B", 0.5), (0.90, 0.60, 0.10, "#3B82F6", 0.5), (0.08, 0.45, 0.10, "#EF4444", 0.45)]
    cx, cy, rx, ry = x0 + w / 2, y0 + h / 2, w / 2, h / 2
    g = [f'<radialGradient id="{pid}-b{i}"><stop offset="0" stop-color="{col}" stop-opacity="{op}"/><stop offset="1" stop-color="{col}" stop-opacity="0"/></radialGradient>'
         for i, (_, _, _, col, op) in enumerate(blobs)]
    defs = "".join(g) + f'<clipPath id="{pid}-clip"><ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"/></clipPath>'
    circles = "".join(f'<circle cx="{x0 + u * w:.1f}" cy="{y0 + v * h:.1f}" r="{r * w:.1f}" fill="url(#{pid}-b{i})"/>' for i, (u, v, r, _, _) in enumerate(blobs))
    grid = "".join(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx * f:.1f}" ry="{ry}"/>' for f in (0.2, 0.45, 0.7, 0.9))
    grid += "".join(f'<line x1="{x0}" y1="{y0 + h * f:.1f}" x2="{x0 + w}" y2="{y0 + h * f:.1f}"/>' for f in (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6))
    body = f"""<g clip-path="url(#{pid}-clip)"><rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#0B2A4A"/>{circles}
      <g fill="none" stroke="#E2E8F0" stroke-opacity="0.16" stroke-width="1.2">{grid}</g></g>
      <ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="#7DD3FC" stroke-opacity="0.45" stroke-width="2"/>"""
    return defs, body


def banner_svg(w=1280, h=400):
    mdefs, mbody = mark("bn")
    sdefs, sbody = _sst_map(905, 92, 330, 190, "sst")
    chips = ["PyTorch-native", "JAX · ONNX · CUDA", "Verified vs official code", "Lite models for small GPUs"]
    cx, cy, parts = 92, 302, []
    for t in chips:
        cw = 26 + 9.4 * len(t)
        parts.append(f'<rect x="{cx}" y="{cy}" width="{cw:.0f}" height="38" rx="19" fill="#FFFFFF" fill-opacity="0.08" stroke="#7DD3FC" stroke-opacity="0.45"/>'
                     f'<text x="{cx + cw / 2:.0f}" y="{cy + 25}" text-anchor="middle" font-size="16" font-weight="600" fill="#E0F2FE">{t}</text>')
        cx += cw + 12
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{NAVY}"/><stop offset="0.55" stop-color="{NAVY2}"/><stop offset="1" stop-color="{NAVY3}"/></linearGradient>
    <linearGradient id="ai" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#22D3EE"/><stop offset="1" stop-color="#5EEAD4"/></linearGradient>
    {mdefs}{sdefs}
  </defs>
  <rect width="{w}" height="{h}" rx="28" fill="url(#bg)"/>
  <g opacity="0.9">{sbody}</g>
  <g fill="none" stroke="#38BDF8" stroke-opacity="0.10" stroke-width="2">
    <path d="M0,350 C220,300 420,390 640,330 S1040,280 1280,330"/><path d="M0,372 C260,330 460,400 700,352 S1080,310 1280,352"/>
  </g>
  <g transform="translate(84 62) scale(0.36)">{mbody}</g>
  <text x="290" y="152" font-size="92" font-weight="800" fill="#FFFFFF" letter-spacing="-2">Weather<tspan fill="url(#ai)">AI</tspan></text>
  <text x="294" y="200" font-size="25" font-weight="500" fill="#CBD5E1">20 peer-reviewed weather &amp; climate AI models.</text>
  <text x="294" y="236" font-size="25" font-weight="500" fill="#CBD5E1">One readable PyTorch library.</text>
  {''.join(parts)}
</svg>
"""


def social_svg(w=1280, h=640):
    """GitHub social preview (1280x640): banner centred on a taller canvas."""
    inner = banner_svg().split(">", 1)[1].rsplit("</svg>", 1)[0]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
  <rect width="{w}" height="{h}" fill="{NAVY}"/><g transform="translate(0 120)">{inner}</g>
  <text x="{w / 2}" y="590" text-anchor="middle" font-size="24" font-weight="600" fill="#7DD3FC">github.com/GISWLH/WeatherAI</text>
</svg>
"""


def render_png(svg_path, png_path, w, h, scale=2):
    from playwright.sync_api import sync_playwright

    exe = os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe if os.path.exists(exe) else None)
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
        pg.goto("file://" + svg_path)
        pg.screenshot(path=png_path, omit_background=True, clip={"x": 0, "y": 0, "width": w, "height": h})
        b.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = {"logo.svg": (logo_svg(), 512, 512), "icon.svg": (icon_svg(), 512, 512), "banner.svg": (banner_svg(), 1280, 400),
             "social_preview.svg": (social_svg(), 1280, 640)}
    for name, (svg, w, h) in files.items():
        path = os.path.join(OUT, name)
        open(path, "w").write(svg)
        try:
            render_png(path, path[:-4] + ".png", w, h, scale=1 if name == "social_preview.svg" else 2)
        except ImportError:
            print("playwright not installed: skipped", name[:-4] + ".png")
    print("wrote", sorted(os.listdir(OUT)))
