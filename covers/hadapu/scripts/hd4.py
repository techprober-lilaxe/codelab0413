# 哈达铺 · 纪念碑 + 长征：底图来自 hd_base.py（调好色的纪念碑 + 天空蒙版），
# 思源黑体 Black 巨型「长征」放在纪念碑后面（碑尖从字前面穿过），「哈达铺」竖排在右下天空
from common import *

base = np.load(os.path.join(HERE, 'hd_base.npy')).astype(np.float32)
sky = np.load(os.path.join(HERE, 'hd_sky.npy')).astype(np.float32)
out = base.copy()

# ---------- 长征 behind the monument ----------
CF = os.environ.get('CF', 'noto-sans-sc')
CW = int(os.environ.get('CW', 900))
cz = render_cjk('长征', CF, CW, 1100, tracking=int(os.environ.get('CT', 30)))
margin = int(os.environ.get('CM', 90))
tw = W - 2 * margin
cz = cz.resize((tw, int(cz.height * tw / cz.width)), Image.LANCZOS)
CY = int(os.environ.get('CZY', 330))
C = Image.new('L', (W, H), 0)
place(C, cz, margin, CY)
Ca = arr(C) * sky                                   # only where the sky is -> the monument stays in front
CCOL = np.array([int(v) for v in os.environ.get('CCOL', '250,246,236').split(',')]) / 255.0
CA = float(os.environ.get('CA', 0.92))
shd = arr(C.filter(ImageFilter.GaussianBlur(30))) * sky
out = out * (1 - 0.22 * shd[..., None]) + np.array([0.05, 0.12, 0.28]) * 0.22 * shd[..., None]
out = out * (1 - CA * Ca[..., None]) + CCOL * CA * Ca[..., None]

# ---------- 哈达铺 ----------
HF = os.environ.get('HF', 'noto-serif-sc')
HW = int(os.environ.get('HW', 900))
HS = int(os.environ.get('HS', 330))
txt = os.environ.get('HT', '哈达铺')
chars = [render_cjk(ch, HF, HW, HS) for ch in txt]
colw = max(c.width for c in chars)
cellh = int(HS * float(os.environ.get('HLH', 1.04)))
HX = int(os.environ.get('HX', W - 110 - colw))
HY = int(os.environ.get('HY', 1380))
T = Image.new('L', (W, H), 0)
for i, c in enumerate(chars):
    place(T, c, HX + (colw - c.width) // 2, HY + i * cellh + (cellh - c.height) // 2)
Ta = arr(T)
sh = arr(T.filter(ImageFilter.GaussianBlur(16)))
out = out * (1 - 0.45 * sh[..., None]) + np.array([0.04, 0.10, 0.22]) * 0.45 * sh[..., None]
HCOL = np.array([int(v) for v in os.environ.get('HCOL', '252,244,224').split(',')]) / 255.0
out = out * (1 - Ta[..., None]) + HCOL * Ta[..., None]

out += grain(out.shape[:2], 0.010, 43)[..., None]
name = os.environ.get('OUT', 'hd4.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('长征 y', CY, CY + cz.height, '哈达铺 x', HX, HX + colw, 'y', HY, HY + len(txt) * cellh)
