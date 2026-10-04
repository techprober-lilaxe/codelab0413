# 哈达铺 · 胶片书法版（参考：人民万岁 / 红星照中国 等胶片风海报）。底图沿用五角星版的缩小纪念碑；
# 去掉五角星和色带，加胶片调色（橙红、褪色青蓝天、提灰暗部、雾、粗颗粒），鸿雷行书竖排「哈达铺」+ 汇文明朝体小字
# 原注释：哈达铺 · 纪念碑 · 五角星：纪念碑缩小置底居中，四周用原片天空补全；
# 碑后一颗五角星（碑尖从星前穿过），顶端横排「哈达铺」+ 向下渐隐的色带；整体饱和度收回来
from common import *

src = Image.open(os.path.join(HERE, 'hd_mon.png')).convert('RGB')
a = arr(src)
h, w = a.shape[:2]
# patch the missing tile on the pedestal (bluish hole at x303-354, y1167-1207): copy the neighbouring tile
PX0, PX1, PY0, PY1 = 300, 358, 1164, 1209
donor = a[PY0:PY1, PX1 + 4:PX1 + 4 + (PX1 - PX0)].copy()
pm = np.zeros((PY1 - PY0, PX1 - PX0), np.float32)
pm[3:-3, 3:-3] = 1
pm = np.asarray(Image.fromarray((pm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))).astype(np.float32)[..., None] / 255
a[PY0:PY1, PX0:PX1] = a[PY0:PY1, PX0:PX1] * (1 - pm) + donor * pm

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
SF = float(os.environ.get('SF', 2.0))
fw, fh = int(w * SF), int(h * SF)
fg = arr(img(out).resize((fw, fh), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2, percent=50, threshold=2)))
fa = arr(Image.fromarray((mon * 255).astype(np.uint8)).resize((fw, fh), Image.BICUBIC))
fa = smoothstep(0.3, 0.7, fa)
MCX = 400                                             # source x of the monument's centre line
fx = W // 2 - int(MCX * SF) + int(os.environ.get('DX', -230))
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
STA = float(os.environ.get('STA', 0.0))
shd = arr(Image.fromarray((st * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28)))
outc = outc * (1 - 0.18 * shd[..., None]) + np.array([0.05, 0.12, 0.28]) * 0.18 * shd[..., None]
outc = outc * (1 - STA * st[..., None]) + STC * STA * st[..., None]

# monument on top
outc = outc * (1 - FA[..., None]) + FG * FA[..., None]

# ---------- top band: gradient fading to transparent + 哈达铺 ----------
BANDC = np.array([int(v) for v in os.environ.get('BANDC', '128,20,16').split(',')]) / 255.0
BH = int(os.environ.get('BH', 760))
yy = np.arange(H)[:, None]
ba = (float(os.environ.get('BA', 0.0)) * (1 - smoothstep(0, BH, yy)) ** 1.3)[..., None]
outc = outc * (1 - ba) + BANDC * ba

# ---------- orange-red haze rising from below (sky mostly, a little over the monument) ----------
yh = (np.arange(H) / H)[:, None]
xh = (np.arange(W) / W)[None, :]
hz = smoothstep(float(os.environ.get('HZ0', 0.30)), float(os.environ.get('HZ1', 0.85)), yh + 0.08 * np.sin(xh * 5.0))
hz = hz * float(os.environ.get('HZA', 0.72))
HZC = np.array([214, 78, 38]) / 255.0
wgt = (hz * (1 - 0.85 * FA))[..., None]
outc = outc * (1 - wgt) + HZC * wgt
# ---------- film grade ----------
o = outc
lum = o.mean(2, keepdims=True)
redness = smoothstep(0.08, 0.30, o[..., 0:1] - o[..., 2:3])
# reds -> warmer orange-red, everything else desaturated
orange = np.clip(o * np.array([1.06, 0.92, 0.70]) + np.array([0.02, 0.05, 0.0]), 0, 1)
o = o * (1 - redness) + orange * redness
sat = 0.62 + 0.38 * redness
o = lum + (o - lum) * sat
# sky -> faded cyan-blue
skyness = smoothstep(0.02, 0.15, o[..., 2:3] - o[..., 0:1])
o = o * (1 - 0.35 * skyness) + (o * np.array([0.86, 1.0, 1.0]) + np.array([0.02, 0.05, 0.06])) * 0.35 * skyness
# haze + bloom
blur = arr(img(o).filter(ImageFilter.GaussianBlur(14)))
o = o * 0.82 + blur * 0.18
bright = np.clip(blur - 0.6, 0, 1)
o = o + 0.25 * bright
# faded blacks, warm toe
o = o * 0.86 + np.array([0.075, 0.05, 0.035])
# darker, warmer lower third (ground)
yy2 = (np.arange(H) / H)[:, None, None]
o = o * (1 - 0.25 * smoothstep(0.75, 1.0, yy2))
xx2 = (np.arange(W) / W)[None, :, None]
vig = smoothstep(0.45, 0.95, np.sqrt((xx2 - 0.5) ** 2 * 1.4 + (yy2 - 0.5) ** 2) * 1.6)
o = o * (1 - 0.22 * vig)
outc = o

# ---------- type: big vertical brush 哈达铺 + small 到陕北去 1935 ----------
CREAM = np.array([242, 233, 212]) / 255.0
HF = os.environ.get('HF', 'file:~/fonts/user/zqcyh.ttf')   # 钟齐蔡云汉毛笔行书（用户提供，不入库；非商用授权）
HS = int(os.environ.get('HS', 470))
chars = [render_cjk(ch, HF, 0, HS) for ch in '哈达铺']
colw = max(c.width for c in chars)
cellh = int(HS * float(os.environ.get('HLH', 1.02)))
HX = int(os.environ.get('HX', 1420)); HY = int(os.environ.get('HY', 200))
T = Image.new('L', (W, H), 0)
for i, c in enumerate(chars):
    place(T, c, HX + (colw - c.width) // 2 + int(os.environ.get('JIT', 30)) * (1 if i == 1 else 0), HY + i * cellh)
SF_ = os.environ.get('SFONT', 'glob:cf/hwmct/package/dist/*/*.woff2')
SSZ = int(os.environ.get('SSZ', 96))
small_chars = [render_cjk(ch, SF_, 0, SSZ) for ch in '到陕北去']
sx = HX - SSZ - int(os.environ.get('SGAP', 70))
title_bottom = int(np.nonzero(np.asarray(T).max(1) > 8)[0].max())     # ink bottom of 铺
step = int(SSZ * 1.25)
sy = title_bottom - (3 * step + small_chars[3].height) + 1             # bottoms of the two columns line up
for i, c in enumerate(small_chars):
    place(T, c, sx + (SSZ - c.width) // 2, sy + i * int(SSZ * 1.25))
Ta = arr(T)
outc = outc * (1 - Ta[..., None] * 0.96) + CREAM * Ta[..., None] * 0.96

# ---------- grain over everything (text included) ----------
rng = np.random.default_rng(57)
def grain2(sig, blur_):
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    n = np.asarray(Image.fromarray(n, 'F').filter(ImageFilter.GaussianBlur(blur_))) if False else n
    return n * sig
g1 = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
g1 = np.asarray(Image.fromarray(g1, 'F').resize((W, H), Image.BICUBIC))
g2 = rng.normal(0, 1, (H, W)).astype(np.float32)
lumc = outc.mean(2, keepdims=True)
amp = (0.05 + 0.05 * (1 - np.abs(lumc - 0.5) * 2))
outc = outc + (g1[..., None] * 0.7 + g2[..., None] * 0.5) * amp
chroma = rng.normal(0, 1, (H // 3, W // 3, 3)).astype(np.float32)
chroma = np.stack([np.asarray(Image.fromarray(chroma[..., c], 'F').resize((W, H), Image.BICUBIC)) for c in range(3)], -1)
outc = outc + chroma * 0.018
name = os.environ.get('OUT', 'hd6.png')
img(outc).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('title', HX, HX + colw, HY, HY + 3 * cellh, 'small x', sx, 'y', sy)
