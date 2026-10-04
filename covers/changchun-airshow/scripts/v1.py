# 彩烟版：八一编队满版置底，上方天空延展放标题，领队机压住标题下沿
from common import *

S = float(os.environ.get('S', 1.40))
SX = int(os.environ.get('SX', 330))      # source x shown at canvas x=0
OY = int(os.environ.get('OY', 1000))     # canvas y of photo top

src = load(1)
sw, sh = src.size
ph = src.resize((int(sw * S), int(sh * S)), Image.LANCZOS)
pa = arr(ph)
ox = -int(SX * S)

canvas = np.zeros((H, W, 3), np.float32)
# photo region
y0, y1 = OY, min(H, OY + pa.shape[0])
canvas[y0:y1] = pa[0:y1 - y0, -ox:-ox + W]
# sky extension above: per-column colour from clean top rows of the photo, blurred
top = pa[5:60, -ox:-ox + W].mean(0)                        # (W,3)
k = np.ones(301) / 301
top = np.stack([np.convolve(np.pad(top[:, c], 150, mode='edge'), k, 'valid') for c in range(3)], 1)
ys = np.arange(OY)[:, None, None]
# toward the very top: slightly cooler / lighter (long, gentle)
t = (1 - ys / OY) ** 1.6
cool = np.array([0.93, 0.955, 0.975])
ext = top[None] * (1 - 0.55 * t) + cool * 0.55 * t
canvas[:OY] = ext
# seam blend: the first rows of the photo -> blend toward extension colour
blend = 160
for i in range(blend):
    w = 1 - smoothstep(0, 1, (i + 1) / blend)
    canvas[OY + i] = canvas[OY + i] * (1 - w) + top * w
# if photo shorter than canvas bottom, extend with last rows (not expected)

# foreground (jets + smoke) alpha over the full canvas
mx, mn, lum = canvas.max(2), canvas.min(2), canvas.mean(2)
fg = np.maximum(smoothstep(0.10, 0.30, mx - mn), smoothstep(0.80, 0.55, lum))
# solid silhouette for the jets near the title: beige wings are low-saturation, key them by r-b
bx0, bx1, by0, by1 = int((540 - SX) * S), int((930 - SX) * S), OY, OY + int(260 * S)
sub_ = canvas[by0:by1, max(0, bx0):bx1]
rb = smoothstep(0.03, 0.07, sub_[..., 0] - sub_[..., 2])
fg[by0:by1, max(0, bx0):bx1] = np.maximum(fg[by0:by1, max(0, bx0):bx1], rb)
fgi = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
fg = arr(fgi)

# ---------------- type ----------------
NAVY = np.array([22, 32, 56]) / 255.0
title = render_cjk('人民空军', 'noto-serif-sc', 900, 470, tracking=40)
margin = 150
sc = (W - 2 * margin) / title.width
title = title.resize((int(title.width * sc), int(title.height * sc)), Image.LANCZOS)
TY = int(os.environ.get('TY', 640))
L = Image.new('L', (W, H), 0)
place(L, title, margin, TY)

eng = render_latin('CHINA AIR FORCE', barlow(800, 200), tracking=6)
sc = (W - 2 * margin) / eng.width
eng = eng.resize((int(eng.width * sc), int(eng.height * sc)), Image.LANCZOS)
EY = TY - eng.height - 60
E = Image.new('L', (W, H), 0)
place(E, eng, margin, EY)

sub = render_cjk('长春航展', 'noto-serif-sc', 700, 120, tracking=28)
S2 = Image.new('L', (W, H), 0)
SUBY = TY + title.height + 80
place(S2, sub, W - margin - sub.width, SUBY)
cs = render_latin('CHANGCHUN  AIRSHOW', barlow(600, 58), tracking=9)
place(S2, cs, W - margin - cs.width, SUBY + sub.height + 34)

out = canvas.copy()
for M, col in [(E, NAVY), (L, NAVY), (S2, NAVY)]:
    a = arr(M)[..., None] * (1 - fg[..., None])
    out = out * (1 - a) + col * a

# soft grain
out += grain(out.shape[:2], 0.012, 3)[..., None]
name = os.environ.get('OUT', 'v1.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print(EY, TY, title.height, SUBY)
