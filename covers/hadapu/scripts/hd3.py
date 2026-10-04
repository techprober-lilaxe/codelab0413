# 哈达铺 · 纪念碑版：片尾 44.5s「到陕北去」纪念碑满版（720x1280 放大 3x 竖裁 3:4），
# 天空压深成饱和的蓝、碑体红更浓；右侧碑尖旁的天空里竖排米白毛笔大字「哈达铺」
from common import *

src = Image.open(os.path.join(HERE, 'hd_mon.png')).convert('RGB')
S = 3.0
CY0 = int(os.environ.get('CY0', 230))            # source y at canvas top (keeps the spike tips)
a = arr(src)

# ---------- grade (source res) ----------
r, g_, b = a[..., 0], a[..., 1], a[..., 2]
sky = smoothstep(-0.10, 0.04, b - r)              # blue sky + clouds (clouds are slightly blue too)
sky = arr(Image.fromarray((sky * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)))
yv = (np.arange(a.shape[0]) / a.shape[0])[:, None]
# deepen the sky: polariser-like — pull the blue down & richer, more toward the top, clouds keep their whites
lum = a.mean(2)
cloud = smoothstep(0.76, 0.86, lum)
deep = np.array([38, 92, 178]) / 255.0
amt = (0.60 - 0.25 * yv) * (1 - 0.85 * cloud)
skyc = a * (1 - amt[..., None]) + deep * amt[..., None] * (a / np.maximum(a.mean(2, keepdims=True), 1e-3)) ** 0.0
skyc = np.clip(skyc, 0, 1)
out = a * (1 - sky[..., None]) + skyc * sky[..., None]
# the monument: a touch richer red, deeper shadows
red = (1 - sky)[..., None]
m_ = out.mean(2, keepdims=True)
rich = np.clip(m_ + (out - m_) * 1.15, 0, 1)
rich = np.clip((rich - 0.5) * 1.08 + 0.5 - 0.02, 0, 1)
out = out * (1 - red) + rich * red
# de-block the sky: blur only among sky pixels (normalized), so the red never bleeds in
def boxblur(x, r):
    for _ in range(3):
        for ax in (0, 1):
            pad = [(0, 0)] * x.ndim; pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(x, pad, mode='edge'), axis=ax)
            x = (np.take(c, range(2 * r + 1, c.shape[ax]), axis=ax) - np.take(c, range(0, c.shape[ax] - 2 * r - 1), axis=ax)) / (2 * r + 1)
    return x
core = (sky > 0.95).astype(np.float32)
core = arr(Image.fromarray((core * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)))   # stay off the edge
R = int(os.environ.get('DB', 4))
num = boxblur(out * core[..., None], R); den = boxblur(core, R)[..., None]
smooth = num / np.maximum(den, 1e-3)
w_ = (core * np.clip(den[..., 0] * 1.5, 0, 1))[..., None]
out = out * (1 - w_) + smooth * w_
# the 1-2px rim between sky and red: pull it toward the nearby sky so there's no light halo
rim = np.clip(sky - core, 0, 1) * (sky > 0.15)
wide = boxblur(out * core[..., None], 3) / np.maximum(boxblur(core, 3), 1e-3)[..., None]
out = out * (1 - 0.6 * rim[..., None]) + wide * 0.6 * rim[..., None]
SKY = sky
# warm late light overall
out = out * np.array([1.02, 1.0, 0.98])

# ---------- upscale + crop ----------
im = img(out)
big = im.resize((int(720 * S), int(1280 * S)), Image.LANCZOS)
sharp = arr(big.filter(ImageFilter.UnsharpMask(radius=3, percent=60, threshold=2)))
skyB = arr(img(SKY).resize(big.size, Image.BICUBIC))[..., 0] if SKY.ndim == 2 else None
skyB = arr(Image.fromarray((SKY * 255).astype(np.uint8)).resize(big.size, Image.BICUBIC))
monB = smoothstep(0.6, 0.1, skyB)                      # sharpen the monument only (no halo into the sky)
monB = arr(Image.fromarray((monB * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(7)))[..., None]
bigA = arr(big) * (1 - monB) + sharp * monB
canvas = bigA[int(CY0 * S):int(CY0 * S) + H, :W].copy()

# ---------- title ----------
CREAM = np.array([252, 244, 224]) / 255.0
TS = int(os.environ.get('TS', 520))
chars = [render_cjk(ch, 'ma-shan-zheng', 400, TS) for ch in '哈达铺']
colw = max(c.width for c in chars)
cellh = int(TS * float(os.environ.get('LH', 1.0)))
TX = int(os.environ.get('TX', W - 150 - colw))
TY = int(os.environ.get('TY', 330))
T = Image.new('L', (W, H), 0)
for i, c in enumerate(chars):
    place(T, c, TX + (colw - c.width) // 2, TY + i * cellh + (cellh - c.height) // 2)
Ta = arr(T)
sh = arr(T.filter(ImageFilter.GaussianBlur(22)))
canvas = canvas * (1 - 0.40 * sh[..., None]) + np.array([0.04, 0.10, 0.22]) * 0.40 * sh[..., None]
canvas = canvas * (1 - Ta[..., None]) + CREAM * Ta[..., None]

canvas += grain(canvas.shape[:2], 0.016, 41)[..., None]
name = os.environ.get('OUT', 'hd3.png')
img(canvas).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('title x', TX, TX + colw, 'y', TY, TY + 3 * cellh)
