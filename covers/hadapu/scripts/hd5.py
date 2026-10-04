# 哈达铺 · 纪念碑 · 五角星：纪念碑缩小置底居中，四周用原片天空补全；
# 碑后一颗五角星（碑尖从星前穿过），顶端横排「哈达铺」+ 向下渐隐的色带；整体饱和度收回来
from common import *

src = Image.open(os.path.join(HERE, 'hd_mon.png')).convert('RGB')
a = arr(src)
h, w = a.shape[:2]

def boxblur(x, r):
    for _ in range(3):
        for ax in (0, 1):
            pad = [(0, 0)] * x.ndim; pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(x, pad, mode='edge'), axis=ax)
            x = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return x

# ---------- matte (source res) ----------
r_, b_ = a[..., 0], a[..., 2]
sky = smoothstep(-0.10, 0.04, b_ - r_)
sky = arr(Image.fromarray((sky * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)))
# only sky connected to the left / right / top edges counts (white inscriptions on the monument are not sky)
small = sky > 0.5
mark = np.zeros_like(small)
mark[0, :] = small[0, :]; mark[:, 0] = small[:, 0]; mark[:, -1] = small[:, -1]
while True:
    g = mark.copy()
    g[1:] |= mark[:-1]; g[:-1] |= mark[1:]; g[:, 1:] |= mark[:, :-1]; g[:, :-1] |= mark[:, 1:]
    g &= small
    if (g == mark).all():
        break
    mark = g
conn = arr(Image.fromarray((mark * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)))
sky = sky * conn
mon = 1 - sky

# ---------- grade (milder than v1: user found the red over-saturated) ----------
yv = (np.arange(h) / h)[:, None]
lum = a.mean(2)
cloud = smoothstep(0.76, 0.86, lum)
deep = np.array([52, 102, 176]) / 255.0
amt = (float(os.environ.get('SKYA', 0.42)) - 0.2 * yv) * (1 - 0.85 * cloud)
skyc = np.clip(a * (1 - amt[..., None]) + deep * amt[..., None], 0, 1)
out = a * (1 - sky[..., None]) + skyc * sky[..., None]
m_ = out.mean(2, keepdims=True)
SAT = float(os.environ.get('SAT', 0.86))           # < 1: pull the red back
rich = np.clip(m_ + (out - m_) * SAT, 0, 1)
rich = np.clip((rich - 0.5) * 1.04 + 0.5, 0, 1)
out = out * (1 - mon[..., None]) + rich * mon[..., None]

# de-block the sky (normalized blur among sky pixels only)
core = arr(Image.fromarray(((sky > 0.95) * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)))
num = boxblur(out * core[..., None], 4); den = boxblur(core, 4)[..., None]
wv = (core * np.clip(den[..., 0] * 1.5, 0, 1))[..., None]
out = out * (1 - wv) + (num / np.maximum(den, 1e-3)) * wv

# ---------- background sky: same frame, bigger, with the monument painted out ----------
fill = boxblur(out * core[..., None], 40) / np.maximum(boxblur(core, 40), 1e-3)[..., None]
fill2 = boxblur(out * core[..., None], 90) / np.maximum(boxblur(core, 90), 1e-3)[..., None]
holew = smoothstep(0.0, 0.25, boxblur(core, 40))[..., None]
fill = fill * holew + fill2 * (1 - holew)
# feathered blend: real sky only well away from the hole, so the old silhouette doesn't ghost through
far = smoothstep(0.92, 1.0, boxblur(core, int(os.environ.get('FEA', 26))))
bgsrc = out * far[..., None] + fill * (1 - far[..., None])
SB = W / w                                            # 3.0: fill the width
bg = arr(img(bgsrc).resize((W, int(h * SB)), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3)))
bg = bg[:H]                                           # top of the sky

# ---------- monument, smaller, bottom-centre ----------
SF = float(os.environ.get('SF', 1.8))
fw, fh = int(w * SF), int(h * SF)
fg = arr(img(out).resize((fw, fh), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2, percent=50, threshold=2)))
fa = arr(Image.fromarray((mon * 255).astype(np.uint8)).resize((fw, fh), Image.BICUBIC))
fa = smoothstep(0.3, 0.7, fa)
MCX = 400                                             # source x of the monument's centre line
fx = W // 2 - int(MCX * SF) + int(os.environ.get('DX', 0))
fy = H - fh
FA = np.zeros((H, W), np.float32); FG = np.zeros((H, W, 3), np.float32)
x0, x1 = max(0, fx), min(W, fx + fw)
FA[fy:, x0:x1] = fa[:, x0 - fx:x1 - fx]; FG[fy:, x0:x1] = fg[:, x0 - fx:x1 - fx]

outc = bg.copy()

# ---------- star behind the monument ----------
SCX, SCY = int(os.environ.get('SCX', W // 2)), int(os.environ.get('SCY', 1300))
SR = int(os.environ.get('SR', 720))
pts = []
for k in range(10):
    ang = -np.pi / 2 + k * np.pi / 5
    rr = SR if k % 2 == 0 else SR * 0.382
    pts.append((SCX + rr * np.cos(ang), SCY + rr * np.sin(ang)))
SS = 3
st = Image.new('L', (W * SS // 2, H * SS // 2), 0)
ImageDraw.Draw(st).polygon([(x * SS / 2, y * SS / 2) for x, y in pts], fill=255)
st = arr(st.resize((W, H), Image.LANCZOS))
STC = np.array([int(v) for v in os.environ.get('STC', '244,196,84').split(',')]) / 255.0
STA = float(os.environ.get('STA', 0.95))
shd = arr(Image.fromarray((st * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28)))
outc = outc * (1 - 0.18 * shd[..., None]) + np.array([0.05, 0.12, 0.28]) * 0.18 * shd[..., None]
outc = outc * (1 - STA * st[..., None]) + STC * STA * st[..., None]

# monument on top
outc = outc * (1 - FA[..., None]) + FG * FA[..., None]

# ---------- top band: gradient fading to transparent + 哈达铺 ----------
BANDC = np.array([int(v) for v in os.environ.get('BANDC', '128,20,16').split(',')]) / 255.0
BH = int(os.environ.get('BH', 760))
yy = np.arange(H)[:, None]
ba = (float(os.environ.get('BA', 0.92)) * (1 - smoothstep(0, BH, yy)) ** 1.3)[..., None]
outc = outc * (1 - ba) + BANDC * ba

HS = int(os.environ.get('HS', 250))
t = render_cjk('哈达铺', os.environ.get('HF', 'noto-serif-sc'), int(os.environ.get('HW', 900)), HS,
               tracking=int(os.environ.get('HT', 110)))
T = Image.new('L', (W, H), 0)
TY = int(os.environ.get('TY', 210))
place(T, t, (W - t.width) // 2, TY)
Ta = arr(T)
CREAM = np.array([250, 238, 208]) / 255.0
outc = outc * (1 - Ta[..., None]) + CREAM * Ta[..., None]

outc += grain(outc.shape[:2], 0.010, 47)[..., None]
name = os.environ.get('OUT', 'hd5.png')
img(outc).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('monument x', fx, fx + fw, 'top', fy, 'spike tip ~', fy + int(255 * SF), '| title', TY, TY + t.height, t.width)
