import math, os
ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 540, 320

def wave(baseline, amp, periods, step=6):
    pts = []
    x = 0
    while x <= W:
        y = baseline + amp*math.sin(2*math.pi*periods*x/W)
        pts.append(f"{x:.1f} {y:.1f}"); x += step
    return "M " + " L ".join(pts)

def star(cx, cy, s):
    return (f"M{cx} {cy-s} L{cx+0.28*s:.1f} {cy-0.28*s:.1f} L{cx+s} {cy} "
            f"L{cx+0.28*s:.1f} {cy+0.28*s:.1f} L{cx} {cy+s} "
            f"L{cx-0.28*s:.1f} {cy+0.28*s:.1f} L{cx-s} {cy} L{cx-0.28*s:.1f} {cy-0.28*s:.1f} Z")

STARS = [(70,46,9),(300,60,8),(150,150,8),(432,120,10),(110,266,7),(404,250,8),(256,38,7)]
DOTS  = [(352,162,3.4),(58,182,3.4),(470,206,3.4)]

def tile(stroke, op):
    waves = (f'<path d="{wave(95,12,2)}"/>'
             f'<path d="{wave(212,9,3)}"/>'
             f'<path d="{wave(300,11,2)}"/>')
    stars = "".join(f'<path d="{star(*s)}"/>' for s in STARS)
    dots  = "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>' for x,y,r in DOTS)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<g fill="none" stroke="{stroke}" stroke-opacity="{op}" stroke-width="3.4" '
            f'stroke-linecap="round" stroke-linejoin="round">{waves}</g>'
            f'<g fill="{stroke}" fill-opacity="{op}">{stars}{dots}</g>'
            f'</svg>')

open(os.path.join(ROOT,'background/bg_tile_light.svg'),'w').write(tile('#20305f','0.04'))
open(os.path.join(ROOT,'background/bg_tile_dark.svg'),'w').write(tile('#f2ead6','0.035'))
print('wrote bg tiles')
