import os, re, math

ROOT = os.path.dirname(os.path.abspath(__file__))
def W(rel, txt):
    p = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w').write(txt)
    print('wrote', rel)

# ---------------- palette ----------------
C = dict(
    indigo   = '#20305f',
    cream    = '#f2ead6',
    cyan     = '#3a8cc4',
    red      = '#e0455c',
    # derived
    indigo900='#0e1730', indigo800='#16213f', indigo600='#2c3f73', indigo400='#5566a0',
    cream50  ='#faf6ea', paper='#f4ecd9', cream200='#e7dcc1', cream300='#d8c9a6',
    cyan_dark='#2a6c9c', cyan_light='#74b2dd',
    red_dark ='#c23047', red_light='#ec7a89',
    muted    ='#50608c', muted_dark='#aab3cf', border_dark='#2a3a66',
)

# ---------------- badge inner (for embedding) ----------------
def badge_inner():
    s = open(os.path.join(ROOT, 'full_logo.svg')).read()
    s = re.sub(r'<\?xml.*?\?>', '', s, flags=re.S)
    # take everything between the opening <svg ...> and closing </svg>
    inner = s[s.index('>', s.index('<svg'))+1 : s.rindex('</svg>')]
    return inner

# ================= colors/tokens.css =================
tokens = f""":root {{
  /* SIGWAY brand palette */
  --sig-indigo: {C['indigo']};
  --sig-cream:  {C['cream']};
  --sig-cyan:   {C['cyan']};
  --sig-red:    {C['red']};

  /* indigo ramp */
  --sig-indigo-900: {C['indigo900']};
  --sig-indigo-800: {C['indigo800']};
  --sig-indigo-600: {C['indigo600']};
  --sig-indigo-400: {C['indigo400']};
  /* paper ramp */
  --sig-cream-50:  {C['cream50']};
  --sig-paper:     {C['paper']};
  --sig-cream-200: {C['cream200']};
  --sig-cream-300: {C['cream300']};
  /* accents */
  --sig-cyan-dark:  {C['cyan_dark']};
  --sig-cyan-light: {C['cyan_light']};
  --sig-red-dark:   {C['red_dark']};
  --sig-red-light:  {C['red_light']};
}}

/* ---- light (default) semantic roles ---- */
:root, [data-theme="light"] {{
  --bg:       {C['paper']};
  --surface:  {C['cream50']};
  --text:     {C['indigo']};
  --muted:    {C['muted']};
  --border:   {C['cream300']};
  --link:     {C['cyan_dark']};
  --accent:   {C['cyan']};
  --danger:   {C['red_dark']};
}}

/* ---- dark semantic roles ---- */
[data-theme="dark"] {{
  --bg:       {C['indigo900']};
  --surface:  {C['indigo800']};
  --text:     {C['cream']};
  --muted:    {C['muted_dark']};
  --border:   {C['border_dark']};
  --link:     {C['cyan_light']};
  --accent:   {C['cyan']};
  --danger:   {C['red_light']};
}}
"""
W('colors/tokens.css', tokens)

# ================= favicon =================
# The favicon is derived from the user's trimmed artwork
# (favicon_source_less_waves.svg -> favicon/favicon.svg) and exported to PNG/ICO
# in the build commands. build_suite.py intentionally does NOT overwrite it.

# ================= background tiles =================
# Generated separately by bg_gen.py (seamless horizontal waves + stars +
# inflaton ball-on-potential). Run `python3 bg_gen.py` to regenerate.

# ================= palette swatch sheet =================
def swatches(items, title):
    x=40; y=90; w=150; gap=22; rows=[]
    rows.append(f'<text x="40" y="54" font-family="Righteous, sans-serif" font-size="32" fill="{C["indigo"]}">{title}</text>')
    for i,(name,hexv,textc) in enumerate(items):
        cx = 40 + (i%5)*(w+gap)
        cyy = 90 + (i//5)*150
        rows.append(f'<rect x="{cx}" y="{cyy}" width="{w}" height="100" rx="10" fill="{hexv}" stroke="{C["cream300"]}"/>')
        rows.append(f'<text x="{cx+14}" y="{cyy+58}" font-family="Futura, sans-serif" font-size="20" fill="{textc}">{name}</text>')
        rows.append(f'<text x="{cx+14}" y="{cyy+82}" font-family="monospace" font-size="17" fill="{textc}">{hexv}</text>')
    h = 90 + ((len(items)+4)//5)*150 + 20
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 {h}" role="img"><title>{title}</title><rect width="940" height="{h}" fill="{C["cream50"]}"/>' + "".join(rows) + '</svg>'

pal_items = [
    ('Indigo', C['indigo'], C['cream']),
    ('Cream', C['cream'], C['indigo']),
    ('Cyan', C['cyan'], C['indigo']),
    ('Red', C['red'], C['cream']),
    ('Indigo 900', C['indigo900'], C['cream']),
    ('Paper', C['paper'], C['indigo']),
    ('Cyan dark', C['cyan_dark'], C['cream']),
    ('Red dark', C['red_dark'], C['cream']),
    ('Indigo 400', C['indigo400'], C['cream']),
    ('Cream 300', C['cream300'], C['indigo']),
]
W('colors/palette.svg', swatches(pal_items, 'SIGWAY palette'))

# ================= lockup (badge + wordmark, live text -> outlined later) =================
inner = badge_inner()
TAG = "Scalar-Induced Gravitational Wave AnalYsis"
def lockup(textcolor, tagcolor):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 400" role="img">
<title>SIGWAY — {TAG}</title>
<g transform="translate(8,20) scale(0.288)">{inner}</g>
<text x="430" y="214" font-family="Righteous" font-size="140" letter-spacing="4" fill="{textcolor}">SIGWAY</text>
<text x="436" y="268" font-family="Avenir Next" font-weight="500" font-size="31" letter-spacing="1.2" fill="{tagcolor}">{TAG}</text>
</svg>
"""
W('brand/lockup_horizontal.svg', lockup(C['indigo'], C['muted']))
W('brand/lockup_horizontal_dark.svg', lockup(C['cream'], C['muted_dark']))

# wordmark only
W('brand/wordmark.svg', f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 220" role="img"><title>SIGWAY</title>
<text x="12" y="162" font-family="Righteous" font-size="150" letter-spacing="4" fill="{C['indigo']}">SIGWAY</text></svg>
""")

# ================= social card 1280x640 =================
def swave(baseline, amp, periods, width=1280, step=10):
    pts = []; x = 0
    while x <= width:
        pts.append(f"{x} {baseline+amp*math.sin(2*math.pi*periods*x/width):.1f}"); x += step
    return "M " + " L ".join(pts)
def sstar(cx, cy, s):
    return (f"M{cx} {cy-s} L{cx+0.28*s:.1f} {cy-0.28*s:.1f} L{cx+s} {cy} L{cx+0.28*s:.1f} {cy+0.28*s:.1f} "
            f"L{cx} {cy+s} L{cx-0.28*s:.1f} {cy+0.28*s:.1f} L{cx-s} {cy} L{cx-0.28*s:.1f} {cy-0.28*s:.1f} Z")
sw_paths = "".join(f'<path d="{swave(b,a,p)}"/>' for b,a,p in [(118,18,4),(330,14,5),(522,20,3)])
ss_paths = "".join(f'<path d="{sstar(x,y,s)}"/>' for x,y,s in
                   [(980,92,11),(1090,200,8),(900,300,8),(1150,420,9),(1030,520,8),(848,150,7),(1190,118,7),(940,452,7)])
social = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" role="img">
<title>SIGWAY — {TAG}</title>
<rect width="1280" height="640" fill="{C['paper']}"/>
<g fill="none" stroke="{C['indigo']}" stroke-opacity="0.05" stroke-width="3.4" stroke-linecap="round">{sw_paths}</g>
<g fill="{C['indigo']}" fill-opacity="0.06">{ss_paths}</g>
<g transform="translate(70,120) scale(0.319)">{inner}</g>
<text x="586" y="300" font-family="Righteous" font-size="120" letter-spacing="4" fill="{C['indigo']}">SIGWAY</text>
<text x="590" y="362" font-family="Avenir Next" font-weight="500" font-size="28" letter-spacing="1" fill="{C['muted']}">{TAG}</text>
</svg>
"""
W('social/social_card.svg', social)

# ================= mkdocs snippet =================
mkdocs_yml = """# --- paste/merge into your mkdocs.yml ---
theme:
  name: material
  logo: assets/images/mark.svg          # the badge (square)
  favicon: assets/images/favicon.svg
  font:
    text: Inter
    code: JetBrains Mono
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      toggle: { icon: material/weather-night, name: Switch to dark mode }
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      toggle: { icon: material/weather-sunny, name: Switch to light mode }
  features:
    - navigation.instant
    - navigation.sections
    - content.code.copy

extra_css:
  - stylesheets/extra.css

# Suggested file placement:
#   docs/assets/images/full_logo.svg      (README + docs hero)
#   docs/assets/images/mark.svg           (copy of full_logo.svg, used as header logo)
#   docs/assets/images/favicon.svg
#   docs/assets/images/bg_tile_light.svg, bg_tile_dark.svg
#   docs/stylesheets/extra.css
"""
W('mkdocs/mkdocs.snippet.yml', mkdocs_yml)

# ================= extra.css for Material =================
extra = f"""/* SIGWAY theme for MkDocs-Material */
@import url('https://fonts.googleapis.com/css2?family=Righteous&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- LIGHT (default) ---------- */
[data-md-color-scheme="default"] {{
  --md-primary-fg-color:        {C['indigo']};
  --md-primary-fg-color--light: {C['indigo600']};
  --md-primary-fg-color--dark:  {C['indigo800']};
  --md-accent-fg-color:         {C['cyan_dark']};
  --md-default-bg-color:        {C['paper']};
  --md-default-fg-color:        {C['indigo']};
  --md-typeset-a-color:         {C['cyan_dark']};
  --md-code-bg-color:           {C['cream50']};
}}

/* ---------- DARK (slate) ---------- */
[data-md-color-scheme="slate"] {{
  --md-primary-fg-color:        {C['indigo800']};
  --md-accent-fg-color:         {C['cyan_light']};
  --md-default-bg-color:        {C['indigo900']};
  --md-default-fg-color:        {C['cream']};
  --md-typeset-a-color:         {C['cyan_light']};
  --md-code-bg-color:           {C['indigo800']};
}}

/* brand headings in Righteous (applied to h1/h2 + header title; h3/h4 stay Inter for legibility) */
.md-typeset h1, .md-typeset h2, .md-header__title {{
  font-family: 'Righteous', system-ui, sans-serif;
  font-weight: 400;
  letter-spacing: .3px;
}}

/* subtle background watermark */
.md-main {{
  background-image: url('../assets/images/bg_tile_light.svg');
  background-repeat: repeat;
}}
[data-md-color-scheme="slate"] .md-main {{
  background-image: url('../assets/images/bg_tile_dark.svg');
}}

/* admonitions tuned to brand accents */
.md-typeset .admonition.note,    .md-typeset details.note    {{ border-color: {C['cyan']}; }}
.md-typeset .note > .admonition-title, .md-typeset .note > summary {{ background: {C['cyan']}1a; }}
.md-typeset .admonition.warning, .md-typeset details.warning {{ border-color: {C['red']}; }}
.md-typeset .warning > .admonition-title, .md-typeset .warning > summary {{ background: {C['red']}1a; }}
"""
W('mkdocs/extra.css', extra)

print('source files generated.')
