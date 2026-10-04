# 拼贴版：上 = 八一编队彩烟整张；中 = 深蓝标题带；下 = J-20 / 拍飞机的我 两张并排，照片直接相接
from common import *

NAVY = np.array([20, 30, 52]) / 255.0
CREAM = np.array([246, 240, 226]) / 255.0
A_H = 1440
BAND_H = int(os.environ.get('BAND', 440))
B_Y = A_H + BAND_H
B_H = H - B_Y

canvas = np.zeros((H, W, 3), np.float32)
p1 = load(1).resize((W, A_H), Image.LANCZOS)
canvas[:A_H] = arr(p1)
canvas[A_H:B_Y] = NAVY

def fit(im, box, w, h):
    c = im.crop(box)
    s = max(w / c.width, h / c.height)
    c = c.resize((int(round(c.width * s)), int(round(c.height * s))), Image.LANCZOS)
    x0 = (c.width - w) // 2; y0 = (c.height - h) // 2
    return arr(c.crop((x0, y0, x0 + w, y0 + h)))

half = W // 2
j20 = fit(load(3), (760, 230, 1880, 1270), half, B_H)
me = fit(load(2), (200, 150, 1300, 1170), half, B_H)
canvas[B_Y:, :half] = j20
canvas[B_Y:, half:] = me

# unify the three photos a little: gentle contrast + same cool shadows
lum = canvas.mean(2, keepdims=True)
canvas = canvas + 0.08 * (canvas - 0.5) * (1 - np.abs(canvas - 0.5) * 2)
canvas = canvas + np.array([0.0, 0.004, 0.02]) * (1 - lum) * 0.8

# ---------------- type in the band ----------------
L = Image.new('L', (W, H), 0)
margin = 140
title = render_cjk('人民空军', 'noto-serif-sc', 900, 300, tracking=22)
eng = render_latin('CHINA AIR FORCE', barlow(800, 120), tracking=4)
sub = render_cjk('长春航展', 'noto-serif-sc', 700, 70, tracking=14)
cs = render_latin('CHANGCHUN  AIRSHOW', barlow(600, 40), tracking=6)
# left: big Chinese, vertically centred in the band
th = 280
title = title.resize((int(title.width * th / title.height), th), Image.LANCZOS)
ty = A_H + (BAND_H - th) // 2
place(L, title, margin, ty)
# right column: English over the subtitle, right-aligned to the margin and to the title's top/bottom
rx = W - margin
ew = rx - (margin + title.width + 90)
eng = eng.resize((ew, int(eng.height * ew / eng.width)), Image.LANCZOS)
place(L, eng, rx - eng.width, ty)
place(L, sub, rx - sub.width, ty + th - sub.height)
place(L, cs, rx - sub.width - 40 - cs.width, ty + th - sub.height + (sub.height - cs.height) // 2)

out = canvas.copy()
al = arr(L)[..., None]
out = out * (1 - al) + CREAM * al
out += grain(out.shape[:2], 0.012, 7)[..., None]
name = os.environ.get('OUT', 'v3.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('title w', title.width, 'eng w', ew)
