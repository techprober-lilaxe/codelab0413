# 哈达铺 · 到陕北去：五六十年代宣传画风
# 满版红色放射光芒（光源在举起的手所指的右上方画面外），举臂石雕做成"印在纸上"的金色单色版画，
# 左侧竖排米黄毛笔大字「到陕北去」+ 深红错位实心投影，旁配小字「哈达铺 · 一九三五」
from common import *

src = Image.open(os.path.join(HERE, 'hd_statue.png')).convert('RGB')    # 5.76s frame, 720x1280
a = arr(src)
CROPB = 1108                                    # stop above the power lines
S = float(os.environ.get('S', 2.6))
FX = int(os.environ.get('FX', 360))             # canvas x of source x=0

# ---------- matte ----------
br = a[..., 2] - a[..., 0]
m = smoothstep(0.08, 0.0, br)
mi = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
m = arr(mi)
m[:, :20] *= 0                                  # sliver of the neighbouring statue at the left edge

# ---------- print treatment of the statue (source res, then upscale) ----------
g = a.mean(2)
g = arr(Image.fromarray((g * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)))
lo, hi = np.percentile(g[m > 0.5], [3, 97])
g = np.clip((g - lo) / (hi - lo), 0, 1)
g = smoothstep(0, 1, g) * 0.6 + g * 0.4             # a little more punch
levels = int(os.environ.get('LV', 7))
gp = np.round(g * (levels - 1)) / (levels - 1)       # flat poster tones
INK = np.array([70, 18, 12]) / 255.0                 # deep red-brown ink
MID = np.array([196, 104, 40]) / 255.0               # burnt orange
GOLD = np.array([252, 214, 128]) / 255.0             # warm gold highlight
def ramp(t):
    t = t[..., None]
    return np.where(t < 0.5, INK + (MID - INK) * (t / 0.5), MID + (GOLD - MID) * ((t - 0.5) / 0.5))
col = ramp(gp)

def up(x, size):
    im = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
    return arr(im.resize(size, Image.BICUBIC))
sw, sh = int(720 * S), int(CROPB * S)
COL = up(col[:CROPB], (sw, sh))
M = up(m[:CROPB], (sw, sh))
M = smoothstep(0.35, 0.65, M)                        # crisp silhouette after upscaling

# ---------- background: radiating rays ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
LX, LY = float(os.environ.get('LX', 1830)), float(os.environ.get('LY', -300))   # light source (off canvas, up-right)
ang = np.arctan2(yy - LY, xx - LX)
NR = int(os.environ.get('NR', 44))
ray = (np.floor((ang + np.pi) / (2 * np.pi) * NR * 2) % 2)
R1 = np.array([190, 36, 28]) / 255.0
R2 = np.array([214, 60, 38]) / 255.0
bg = R1 + (R2 - R1) * ray[..., None]
dist = np.hypot(xx - LX, yy - LY)
glow = smoothstep(1400, 300, dist)[..., None]        # warmer / lighter toward the source
bg = bg + (np.array([236, 120, 56]) / 255.0 - bg) * 0.45 * glow
sink = smoothstep(0.55, 1.0, yy / H)[..., None]      # a touch deeper toward the bottom
bg = bg * (1 - 0.18 * sink)

out = bg.copy()
# statue
ox, oy = FX, 0
canvas_col = np.zeros((H, W, 3), np.float32); canvas_m = np.zeros((H, W), np.float32)
x0, x1 = max(0, ox), min(W, ox + sw); y1 = min(H, oy + sh)
canvas_col[oy:y1, x0:x1] = COL[:y1 - oy, x0 - ox:x1 - ox]
canvas_m[oy:y1, x0:x1] = M[:y1 - oy, x0 - ox:x1 - ox]
if y1 < H:                                           # bleed the bottom rows down if the crop is short
    canvas_col[y1:] = canvas_col[y1 - 1]; canvas_m[y1:] = canvas_m[y1 - 1]

# halftone dots in the shadows of the statue (45 degrees)
P = 14
u = (xx + yy) / np.sqrt(2); v = (xx - yy) / np.sqrt(2)
cell = np.hypot((u % P) - P / 2, (v % P) - P / 2) / (P / 2)
lumc = canvas_col.mean(2)
dot = (cell < 1.15 * smoothstep(0.55, 0.25, lumc)).astype(np.float32)
canvas_col = canvas_col * (1 - 0.35 * dot[..., None]) + INK * 0.35 * dot[..., None]
mm = canvas_m[..., None]
out = out * (1 - mm) + canvas_col * mm
# misregistration: shift the red-ish edge 2px for the slightly-off two-colour print feel
out[..., 0] = np.roll(out[..., 0], 2, axis=1) * 0.5 + out[..., 0] * 0.5

# ---------- type ----------
CREAM = np.array([250, 232, 172]) / 255.0
SHADOW = np.array([110, 14, 10]) / 255.0
TS = int(os.environ.get('TS', 470))
chars = []
for ch in '到陕北去':
    chars.append(render_cjk(ch, 'ma-shan-zheng', 400, TS))
colw = max(c.width for c in chars)
cellh = int(TS * float(os.environ.get('LH', 0.98)))
TX, TY = int(os.environ.get('TX', 120)), int(os.environ.get('TY', 330))
T = Image.new('L', (W, H), 0)
for i, c in enumerate(chars):
    place(T, c, TX + (colw - c.width) // 2, TY + i * cellh + (cellh - c.height) // 2)
Ta = arr(T)
off = int(os.environ.get('OFF', 14))
Sh = np.roll(np.roll(Ta, off, axis=0), off, axis=1)
out = out * (1 - Sh[..., None]) + SHADOW * Sh[..., None]
out = out * (1 - Ta[..., None]) + CREAM * Ta[..., None]

# small type: 哈达铺 · 一九三五 (vertical, serif), right of the title's lower half
S2 = Image.new('L', (W, H), 0)
small = int(os.environ.get('SS', 92))
sy = TY + 4 * cellh - 40
sx = TX + colw + 40
txt = os.environ.get('SUB', '哈达铺一九三五')
yy0 = int(os.environ.get('SY', 360))
k = 0
for ch in txt:
    gch = render_cjk(ch, 'noto-serif-sc', 900, small)
    place(S2, gch, sx + (small - gch.width) // 2, yy0 + k)
    k += int(small * 1.18) + (int(small * 0.6) if ch == '铺' else 0)
S2a = arr(S2)
Sh2 = np.roll(np.roll(S2a, 5, axis=0), 5, axis=1)
out = out * (1 - Sh2[..., None]) + SHADOW * Sh2[..., None]
out = out * (1 - S2a[..., None]) + CREAM * S2a[..., None]

# paper: very slight warm wash + grain, edges a hair darker
out = out * 0.97 + np.array([0.03, 0.025, 0.015])
vig = smoothstep(0.55, 1.2, np.hypot((xx - W / 2) / W, (yy - H / 2) / H) * 1.6)[..., None]
out = out * (1 - 0.18 * vig)
out += grain(out.shape[:2], 0.02, 31)[..., None]
name = os.environ.get('OUT', 'hd1.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('title bottom', TY + 4 * cellh, 'sub', yy0, yy0 + k, 'statue', ox, ox + sw, sh)
