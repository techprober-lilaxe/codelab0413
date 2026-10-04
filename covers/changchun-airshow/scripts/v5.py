# 运-20 诱饵弹版：巨型「人民 / 空军」两行立在天空里，飞机和诱饵弹烟迹从字前面穿过（Su-57 参考的主体压字）
from common import *

src = load(8)
S = H / src.height
SX = int(os.environ.get('SX', 45))
a = arr(src)
h, w = a.shape[:2]

# foreground alpha in source space
lum = a.mean(2); rb = a[..., 0] - a[..., 2]
yy, xx = np.mgrid[0:h, 0:w]
sky = (np.abs(rb) < 0.02) & (lum > 0.85)
def basis(x, y):
    x = x / w; y = y / h
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
coef = np.linalg.lstsq(basis(xx[sky] * 1., yy[sky] * 1.), lum[sky], rcond=None)[0]
model = basis(xx * 1., yy * 1.) @ coef
dark = smoothstep(0.03, 0.10, model - lum)
warm = smoothstep(0.015, 0.05, rb)
# the airframe: dark mask closed hard so white smoke puffs on the fuselage don't punch holes
body = Image.fromarray((smoothstep(0.08, 0.16, model - lum) * 255).astype(np.uint8))
# round closing (blur -> threshold, twice) instead of square Max/Min, which leaves blocky notches
b = arr(body.filter(ImageFilter.GaussianBlur(9)))
b = smoothstep(0.22, 0.32, b)                                   # dilate + fill
b = arr(Image.fromarray((b * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)))
b = smoothstep(0.68, 0.78, b)                                   # erode back
body = Image.fromarray((b * 255).astype(np.uint8))
TR = float(os.environ.get("TR", 0.22))   # flare trails: only partly in front of the type
bodyA = arr(body)
bright = smoothstep(0.025, 0.07, lum - model)                    # dense white smoke puffs, brighter than the sky
fgm = np.maximum(np.maximum(np.maximum(dark, warm) * TR, bodyA), bright * 0.9)
fgi = Image.fromarray((fgm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))

ph = arr(src.resize((int(w * S), H), Image.LANCZOS))
fg = arr(fgi.resize((int(w * S), H), Image.LANCZOS))
ox = int(SX * S)
canvas = ph[:, ox:ox + W].copy()
fg = fg[:, ox:ox + W]

lumc = canvas.mean(2, keepdims=True)
canvas = canvas + 0.08 * (canvas - 0.5) * (1 - np.abs(canvas - 0.5) * 2)
canvas = canvas + np.array([0.0, 0.004, 0.015]) * (1 - lumc)

NAVY = np.array([24, 36, 54]) / 255.0
L = Image.new('L', (W, H), 0)
margin = 110
row1 = render_cjk('人民', 'noto-serif-sc', 900, 900, tracking=60)
row2 = render_cjk('空军', 'noto-serif-sc', 900, 900, tracking=60)
tw = W - 2 * margin
r1 = row1.resize((tw, int(row1.height * tw / row1.width)), Image.LANCZOS)
r2 = row2.resize((tw, int(row2.height * tw / row2.width)), Image.LANCZOS)
TY = int(os.environ.get('TY', 420))
GAP = 90
place(L, r1, margin, TY)
place(L, r2, margin, TY + r1.height + GAP)

S2 = Image.new('L', (W, H), 0)
eng = render_latin('CHINA AIR FORCE', barlow(800, 120), tracking=8)
place(S2, eng, W - margin - eng.width, 200)
sub = render_cjk('长春航展', 'noto-serif-sc', 700, 110, tracking=26)
cs = render_latin('CHANGCHUN  AIRSHOW', barlow(600, 54), tracking=9)
by = TY + r1.height + GAP + r2.height + 90
place(S2, sub, W - margin - sub.width, by)
place(S2, cs, W - margin - cs.width, by + sub.height + 28)

out = canvas.copy()
al = arr(L)[..., None] * (1 - fg[..., None])
out = out * (1 - al) + NAVY * al
al = arr(S2)[..., None]
out = out * (1 - al) + NAVY * al
out += grain(out.shape[:2], 0.012, 13)[..., None]
name = os.environ.get('OUT', 'v5.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('rows', r1.height, r2.height, 'bottom block', by, by + sub.height + 28 + cs.height)
