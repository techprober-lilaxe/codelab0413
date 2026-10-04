# J-20 版：地面上的 J-20 + 前景虚化士兵；天空向上延展并压暗成钢蓝，
# 白色大字「人民空军」立在天空里，鸭翼 / 垂尾 / 座舱压住字的下沿
from common import *

SX0, SX1 = 330, 1800
src = load(3)
sw, sh = src.size
S = W / (SX1 - SX0)
ph = arr(src.resize((int(sw * S), int(sh * S)), Image.LANCZOS))
ox = int(SX0 * S)
OY = H - ph.shape[0]
TY = int(os.environ.get('TY', 1010))

# --- sky model (quadratic in x,y) fitted on known-sky samples of the source ---
a = arr(src)
yy, xx = np.mgrid[0:sh, 0:sw]
sky = np.zeros((sh, sw), bool)
sky[:215] = True
sky[:930, 1810:] = True
sky[:680, :90] = True
sky[300:380, 700:1080] = True
sky[230:380, 1000:1300] = True
def basis(x, y):
    x = x / sw; y = y / sh
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
B = basis(xx[sky].astype(float), yy[sky].astype(float))
coef = np.linalg.lstsq(B, a[sky], rcond=None)[0]
model = basis(xx.astype(float), yy.astype(float)) @ coef
diff = np.abs(a - model).max(2)
pm = smoothstep(0.03, 0.06, diff)
pm[950:] = 1.0                                  # below the tree line everything is "foreground"
# canopy glass reads like sky: fill each canopy column from its first hit downwards
for x in range(1100, 1480):
    hit = 380 + np.nonzero(pm[380:700, x] > 0.5)[0]
    if len(hit):
        pm[hit[0]:700, x] = 1.0
pmi = Image.fromarray((pm * 255).astype(np.uint8))
pmi = pmi.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
# the out-of-focus fin tip is pale and patchy: close it locally so it reads as one solid shape
zpts = [(508, 72), (480, 150), (440, 270), (380, 420), (320, 580), (285, 720),
        (150, 720), (230, 520), (300, 330), (360, 200), (420, 120), (470, 85)]
SS = 4
poly = Image.new('L', (sw * SS // 4, sh * SS // 4), 0)
poly = Image.new('L', (220 * SS, 240 * SS), 0)
ImageDraw.Draw(poly).polygon([((zx / 3) * SS, (zy / 3) * SS) for zx, zy in zpts], fill=255)
poly = poly.resize((220, 240), Image.LANCZOS)
reg = pmi.crop((500, 280, 720, 520))
pmi.paste(Image.fromarray(np.maximum(np.asarray(reg), np.asarray(poly))), (500, 280))
pmi = pmi.filter(ImageFilter.GaussianBlur(0.8))
pmi = pmi.resize(src.size and (int(sw * S), int(sh * S)), Image.LANCZOS)
fgp = arr(pmi)

canvas = np.zeros((H, W, 3), np.float32)
fg = np.zeros((H, W), np.float32)
canvas[OY:] = ph[:, ox:ox + W]
fg[OY:] = fgp[:, ox:ox + W]
# sky extension: evaluate the fitted model above the frame (y<0) – smooth by construction,
# but clamp the extrapolation by blending to the top-row colour
ext_y = (np.arange(-OY, 0) / S)[:, None] * np.ones((1, W))
ext_x = ((np.arange(W) + ox) / S)[None, :] * np.ones((OY, 1))
toprow = (basis(((np.arange(W) + ox) / S), np.zeros(W)) @ coef)       # (W,3)
canvas[:OY] = toprow[None]
# faint overcast texture in the extension so it doesn't read as a flat block
rng = np.random.default_rng(11)
tex = np.zeros((OY, W), np.float32)
for cell, amp in [(360, 0.020), (150, 0.012), (60, 0.006)]:
    n = rng.normal(0, 1, (OY // cell + 3, W // cell + 3)).astype(np.float32)
    n = np.asarray(Image.fromarray(n, 'F').resize((W + 3 * cell, OY + 3 * cell), Image.BICUBIC))[:OY, :W]
    tex += amp * n
tex *= smoothstep(0, 1, np.arange(OY)[::-1] / 200.0)[:, None]   # fade out toward the seam
canvas[:OY] += tex[..., None]
# blend seam
for i in range(120):
    w = 1 - smoothstep(0, 1, (i + 1) / 120)
    canvas[OY + i] = canvas[OY + i] * (1 - w) + toprow * w * (1 - fg[OY + i, :, None]) + canvas[OY + i] * w * fg[OY + i, :, None]

# --- cinematic grade: graduated steel-blue darkening from the top ---
yv = np.arange(H)[:, None, None] / H
g = 1 - smoothstep(0.30, 0.62, yv)            # 1 at top -> 0 at ~62% height
deep = np.array([18, 28, 44]) / 255.0
mult = 1 - 0.80 * g
canvas = canvas * mult + deep * (0.80 * g) * 0.9
# overall: slight S-curve + cool shadows / warm highlights
lum = canvas.mean(2, keepdims=True)
canvas = canvas + 0.10 * (canvas - 0.5) * (1 - np.abs(canvas - 0.5) * 2)
canvas = canvas * (1 - 0.06 * (1 - lum)) + np.array([0.0, 0.01, 0.03]) * (1 - lum) + np.array([0.02, 0.01, -0.01]) * lum

# ---------------- type ----------------
CREAM = np.array([246, 242, 232]) / 255.0
margin = 140
title = render_cjk('人民空军', 'noto-serif-sc', 900, 470, tracking=40)
sc = (W - 2 * margin) / title.width
title = title.resize((int(title.width * sc), int(title.height * sc)), Image.LANCZOS)
L = Image.new('L', (W, H), 0)
place(L, title, margin, TY)

eng = render_latin('CHINA AIR FORCE', barlow(800, 200), tracking=6)
sc = (W - 2 * margin) / eng.width
eng = eng.resize((int(eng.width * sc), int(eng.height * sc)), Image.LANCZOS)
EY = TY - eng.height - 60
place(L, eng, margin, EY)

S2 = Image.new('L', (W, H), 0)
sub = render_cjk('长春航展', 'noto-serif-sc', 700, 110, tracking=26)
cs = render_latin('CHANGCHUN  AIRSHOW', barlow(600, 54), tracking=9)
SUBY = int(os.environ.get('SUBY', EY - 300))
place(S2, sub, (W - sub.width) // 2, SUBY)
place(S2, cs, (W - cs.width) // 2, SUBY + sub.height + 30)

out = canvas.copy()
# soft shadow under the type for depth
sh_ = Image.fromarray(np.maximum(np.asarray(L), np.asarray(S2))).filter(ImageFilter.GaussianBlur(18))
sa = arr(sh_)[..., None] * 0.35 * (1 - fg[..., None])
out = out * (1 - sa)
for M in [L, S2]:
    al = arr(M)[..., None] * (1 - fg[..., None])
    out = out * (1 - al) + CREAM * al

out += grain(out.shape[:2], 0.014, 5)[..., None]
name = os.environ.get('OUT', 'v2.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
img(fg).resize((540, 720)).save('v2_fg.png')
print('OY', OY, 'EY', EY, 'TY', TY, title.height, 'SUBY', SUBY)
