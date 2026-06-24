# SIGWAY design suite

**SIGWAY** = **S**calar-**I**nduced **G**ravitational **W**ave **A**nal**Y**sis.

Visual identity assets derived from `full_logo.svg` (the master badge). Everything
here is regenerable with `python3 build_suite.py` (sources) + `python3 bg_gen.py`
(backgrounds) + the inkscape/PIL steps noted at the bottom.

## Palette

| Role        | Name        | Hex       |
|-------------|-------------|-----------|
| Primary ink | Indigo      | `#20305f` |
| Paper       | Cream       | `#f2ead6` |
| Accent      | Cyan        | `#3a8cc4` |
| Energy      | Red         | `#e0455c` |
| Dark bg     | Indigo 900  | `#0e1730` |
| Page paper  | Paper       | `#f4ecd9` |
| Link (light)| Cyan dark   | `#2a6c9c` (AA on paper) |
| Link (dark) | Cyan light  | `#74b2dd` |
| Muted text  | Indigo 400  | `#5566a0` |
| Borders     | Cream 300   | `#d8c9a6` |

Full token set + semantic light/dark roles: `colors/tokens.css`.
Reference swatch sheet: `colors/palette.svg`.

## Typography

- **Wordmark "SIGWAY":** Righteous (outlined to paths in the brand SVGs — no font dependency).
- **Tagline:** Avenir Next (outlined in the brand SVGs).
- **Docs headings (h1/h2 + header title):** Righteous, via Google Fonts. h3/h4 stay Inter for legibility.
- **Docs body:** Inter. **Code:** JetBrains Mono.

Wired up in `mkdocs/extra.css` (Google Fonts `@import`) and `mkdocs/mkdocs.snippet.yml`.

## Files

```
full_logo.svg                 master badge — README + docs hero
brand/
  mark.svg                    = master, used as MkDocs header logo (square)
  lockup_horizontal.svg/.png  badge + SIGWAY wordmark (light bg)
  lockup_horizontal_dark.svg  cream wordmark for dark backgrounds
  wordmark.svg                "SIGWAY" only (outlined)
favicon/
  favicon.svg                 simplified Segway-in-ring
  favicon.ico                 16/32/48 bundle
  icon-16/32/48/192/512.png   apple-touch-icon.png (180)
background/
  bg_tile_light.svg           subtle seamless watermark (tiles via repeat)
  bg_tile_dark.svg            dark-mode variant
social/
  social_card.svg/.png        1280×640 GitHub social-preview / OG card
colors/
  tokens.css  palette.svg
mkdocs/
  mkdocs.snippet.yml  extra.css
```

## Usage

**README header:** use `brand/lockup_horizontal.svg` (or `.png`) at the top; full badge
`full_logo.svg` elsewhere.

**GitHub social preview:** repo → Settings → General → Social preview → upload
`social/social_card.png`.

**Favicon (plain HTML):**
```html
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="icon" href="favicon.ico" sizes="any">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
```

**MkDocs:** copy `mkdocs/extra.css` → `docs/stylesheets/extra.css`, the images →
`docs/assets/images/`, and merge `mkdocs/mkdocs.snippet.yml` into your `mkdocs.yml`.

## Notes

- The tagline ("Scalar-Induced Gravitational Wave AnalYsis") is set via the `TAG`
  variable in `build_suite.py`.
- Preview helper PNGs are prefixed `_` and can be deleted.

## Regenerate

Font note: outlining the wordmark needs **Righteous** installed in `~/Library/Fonts/`
(Inkscape's fontconfig does *not* scan `~/.fonts`, and silently falls back to a serif).

```
python3 build_suite.py            # writes all source SVG/CSS/YAML
python3 bg_gen.py                 # writes the background tiles
# then: outline wordmarks (inkscape object-to-path), export PNGs, build favicon.ico
```
