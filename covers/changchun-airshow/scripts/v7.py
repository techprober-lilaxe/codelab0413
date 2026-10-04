# 诱饵弹 · 蓝天版（原图重做）：运-20 + 诱饵弹从未裁切的原片抠出、缩小居中，背景换成补全的蓝天和云；
# 黑体「人民 / 空军」在飞机后面，被挡住的部分在机身上留一圈细描边（Su-57 参考的做法）
from common import *

S = float(os.environ.get('S', 1.45))
CX, CY = int(os.environ.get('CX', 930)), int(os.environ.get('CY', 729))                         # source point (airframe centre) that goes to canvas centre
src = load(11)
a = arr(src)
h, w = a.shape[:2]

# ---------- matte (source space) ----------
lum = a.mean(2); rb = a[..., 0] - a[..., 2]
yy, xx = np.mgrid[0:h, 0:w]
sky = (np.abs(rb) < 0.03) & (lum > 0.70)
# iterate: refit on pixels close to the current model so smoke doesn't bias it

def basis(x, y):
    x = x / w; y = y / h
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3, x ** 4, y ** 4, x * x * y * y], -1)
B = basis(xx * 1., yy * 1.)
# non-parametric sky: normalized convolution of "sky-like" pixels (downsampled), refined 3x
def boxblur(x, r):
    # three box passes ~ gaussian; edge-padded, along both axes
    for _ in range(3):
        for ax in (0, 1):
            p = np.pad(x, [(r + 1, r)] * 1 + [(0, 0)] * (x.ndim - 1) if ax == 0 else [(0, 0), (r + 1, r)] + [(0, 0)] * (x.ndim - 2), mode='edge')
            c = np.cumsum(p, axis=ax)
            if ax == 0:
                x = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
            else:
                x = (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
    return x
def nblur(img_, m_, r):
    sm = (w // 4, h // 4)
    I = np.stack([np.asarray(Image.fromarray(((img_[..., c] * m_) * 255).astype(np.uint8)).resize(sm, Image.BOX)) for c in range(3)], -1).astype(np.float32)
    M = np.asarray(Image.fromarray((m_ * 255).astype(np.uint8)).resize(sm, Image.BOX)).astype(np.float32)
    I = boxblur(I, r); M = boxblur(M, r)
    out = I / np.maximum(M, 1e-3)[..., None]
    return np.stack([np.asarray(Image.fromarray(out[..., c].astype(np.float32), 'F').resize((w, h), Image.BICUBIC)) for c in range(3)], -1)
m = sky.astype(np.float32)
for r in (14, 10, 7):
    skyc = nblur(a, m, r)
    m = (np.abs(a - skyc).max(2) < 0.02).astype(np.float32)
skyl = skyc.mean(2)
d = np.abs(a - skyc).max(2)
smoke = smoothstep(0.022, 0.075, d)                       # trails & puffs (any difference from sky)
body = smoothstep(0.08, 0.16, skyl - lum)
b = arr(Image.fromarray((body * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)))
b = smoothstep(0.22, 0.32, b)
b = arr(Image.fromarray((b * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)))
body = smoothstep(0.68, 0.78, b)
# the lower-right wing is wrapped in white smoke and doesn't key: trace it by hand
# (points read off a 2x zoom of the source crop at (950,650))
wing = Image.new('L', (w * 2, h * 2), 0)
dw = ImageDraw.Draw(wing)
up = [(60, 100), (140, 112), (400, 262), (700, 402), (970, 545), (1040, 598)]
lo = [(960, 612), (900, 612), (780, 555), (600, 465), (400, 375), (220, 290), (60, 230)]
dw.polygon([((950 + zx / 2) * 2, (650 + zy / 2) * 2) for zx, zy in up + lo], fill=255)
dw.ellipse(((1068) * 2, (838) * 2, (1146) * 2, (890) * 2), fill=255)     # inner engine pod
wing = arr(wing.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2)))
body = np.maximum(body, wing)
# fade the smoke out toward the photo border so the trails dissolve into the new sky
FE = int(os.environ.get('FE', 200))
ex = np.minimum(xx, w - 1 - xx); ey = np.minimum(yy, h - 1 - yy)
edge = smoothstep(0, FE, np.minimum(ex, ey).astype(np.float32)) ** 1.5
body = body * (np.minimum(ex, ey) > 8)                 # border pixels of the source are junk
alpha = np.maximum(smoke * edge, body)
# decontaminate: remove the old white sky from semi-transparent pixels
af = np.clip(alpha, 1e-3, 1)[..., None]
fgc = np.clip((a - (1 - af) * skyc) / af, 0, 1)
fgc = np.where(alpha[..., None] > 0.15, fgc, a)            # tiny alphas: keep original colour (white smoke)
# airframe: a bit more weight so it holds against the cream letters
bm = body[..., None]
fgc = fgc * (1 - bm) + np.clip((fgc - 0.5) * 1.18 + 0.5 - 0.07, 0, 1) * bm
# smoke over blue reads better slightly brighter / whiter
fgc = np.where(body[..., None] > 0.5, fgc, fgc * 0.85 + 0.15 * np.array([0.98, 0.96, 0.93]))

def to_canvas(x):
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    im = im.resize((int(w * S), int(h * S)), Image.LANCZOS)
    out = np.zeros((H, W) + (() if x.ndim == 2 else (3,)), np.float32)
    ox, oy = int(W / 2 - CX * S), int(H / 2 - CY * S)
    sa = arr(im)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(H, oy + sa.shape[0]), min(W, ox + sa.shape[1])
    out[y0:y1, x0:x1] = sa[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out
FG = to_canvas(fgc)
A = to_canvas(alpha)
BODY = to_canvas(body)

# ---------- sky ----------
rng = np.random.default_rng(int(os.environ.get('SEED', 4)))
def fbm(shape, cells, amps):
    t = np.zeros(shape, np.float32)
    for c, amp in zip(cells, amps):
        n = rng.normal(0, 1, (shape[0] // c + 4, shape[1] // c + 4)).astype(np.float32)
        n = np.asarray(Image.fromarray(n, 'F').resize((shape[1] + 4 * c, shape[0] + 4 * c), Image.BICUBIC))
        t += amp * n[2 * c:2 * c + shape[0], 2 * c:2 * c + shape[1]]
    return t
yv = (np.arange(H) / H)[:, None]
TOP = np.array([58, 104, 150]) / 255.0
BOT = np.array([170, 198, 216]) / 255.0
t = smoothstep(0, 1, yv)[..., None]
skyimg = TOP * (1 - t) + BOT * t
skyimg = skyimg * np.ones((1, W, 1))
# clouds: soft, out-of-focus fBm banks, mostly low in the frame and in the corners
n = fbm((H // 4, W // 4), [160, 80, 40, 18, 8], [1.0, 0.6, 0.35, 0.18, 0.09])
n = np.asarray(Image.fromarray(n, 'F').resize((W, H), Image.BICUBIC))
xv = (np.arange(W) / W)[None, :]
bias = 1.15 * smoothstep(0.45, 1.0, yv) + 0.45 * (np.abs(xv - 0.5) * 2) ** 3 * (1 - 0.6 * smoothstep(0.3, 0.6, yv)) - 0.55
dens = smoothstep(0.2, 1.3, n * 0.9 + bias * 1.8)
dens = arr(Image.fromarray((dens * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10)))
# cheap volume: the underside (where density grows downward) a little grey-blue, tops bright
up = arr(Image.fromarray((dens * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(40)))
under = np.clip((up - dens) * 2.2 + 0.25 * dens * smoothstep(0.6, 1.0, yv), 0, 1)
cloud = np.array([0.975, 0.978, 0.985]) * (1 - 0.18 * under[..., None]) + np.array([0.62, 0.70, 0.80]) * 0.18 * under[..., None]
sky_out = skyimg * (1 - dens[..., None] * 0.9) + cloud * dens[..., None] * 0.9

# ---------- type ----------
CREAM = np.array([246, 240, 226]) / 255.0
TW = int(os.environ.get('TW', 1300))
GAP = int(os.environ.get('GAP', 70))
row1 = render_cjk('人民', 'noto-sans-sc', 900, 800, tracking=50)
row2 = render_cjk('空军', 'noto-sans-sc', 900, 800, tracking=50)
r1 = row1.resize((TW, int(row1.height * TW / row1.width)), Image.LANCZOS)
r2 = row2.resize((TW, int(row2.height * TW / row2.width)), Image.LANCZOS)
th = r1.height + GAP + r2.height
TY = (H - th) // 2 + int(os.environ.get('DY', -100))
L = Image.new('L', (W, H), 0)
place(L, r1, (W - TW) // 2, TY)
place(L, r2, (W - TW) // 2, TY + r1.height + GAP)
# outline = dilate - erode of the fill
SW = int(os.environ.get('SW', 7))
outer = L.filter(ImageFilter.MaxFilter(SW | 1))
inner = L.filter(ImageFilter.MinFilter(SW | 1))
STROKE = np.clip(arr(outer) - arr(inner), 0, 1)
STROKE = arr(Image.fromarray((STROKE * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)))
FILL = arr(L)

S2 = Image.new('L', (W, H), 0)
eng = render_latin('CHINA AIR FORCE', barlow(700, 96), tracking=18)
place(S2, eng, (W - eng.width) // 2, TY - eng.height - 130)
sub = render_cjk('长 春 航 展', 'noto-sans-sc', 700, 84, tracking=0)
cs = render_latin('CHANGCHUN  AIRSHOW', barlow(600, 46), tracking=10)
by = TY + th + 130
place(S2, sub, (W - sub.width) // 2, by)
place(S2, cs, (W - cs.width) // 2, by + sub.height + 26)

# ---------- composite ----------
out = sky_out.copy()
# soft shadow of the letters on the sky (very light) for a little lift
shd = arr(L.filter(ImageFilter.GaussianBlur(22)))[..., None]
out = out * (1 - 0.18 * shd)
fa = FILL[..., None]
out = out * (1 - fa) + CREAM * fa                           # letters on the sky
TRL = float(os.environ.get('TRL', 0.35))
Aa = (np.maximum(A * (1 - FILL * (1 - TRL)), BODY * A))[..., None]
out = out * (1 - Aa) + FG * Aa                              # aircraft & smoke in front
# Su-57 trick: where the airframe hides a letter, keep its outline on top
sa = (STROKE * np.clip(BODY + A * 0.6, 0, 1))[..., None] * 0.95
out = out * (1 - sa) + CREAM * sa
s2 = arr(S2)[..., None]
sh2 = arr(S2.filter(ImageFilter.GaussianBlur(10)))[..., None]
out = out * (1 - 0.30 * sh2) + np.array([0.08, 0.16, 0.26]) * 0.30 * sh2
out = out * (1 - s2) + CREAM * s2
out += grain(out.shape[:2], 0.012, 21)[..., None]
name = os.environ.get('OUT', 'v6.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('TY', TY, 'title', th, 'eng y', TY - eng.height - 130, 'bottom', by + sub.height + 26 + cs.height)
