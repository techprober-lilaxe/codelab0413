# 飞行员挥手版：座舱里的飞行员占下半，上方大片阴天留白放标题
from common import *

src = load(6)
S = H / src.height                       # 1.6
SX = int(os.environ.get('SX', 60))       # source x at canvas x=0
ph = arr(src.resize((int(src.width * S), H), Image.LANCZOS))
ox = int(SX * S)
canvas = ph[:, ox:ox + W].copy()

# gentle grade: a touch of contrast, cool shadows; a soft grad to give the sky some weight at the top
yv = np.arange(H)[:, None, None] / H
g = 1 - smoothstep(0.0, 0.42, yv)
canvas = canvas * (1 - 0.10 * g) + np.array([0.0, 0.01, 0.025]) * 0.10 * g
lum = canvas.mean(2, keepdims=True)
canvas = canvas + 0.10 * (canvas - 0.5) * (1 - np.abs(canvas - 0.5) * 2)
canvas = canvas + np.array([0.0, 0.004, 0.015]) * (1 - lum)

NAVY = np.array([24, 36, 54]) / 255.0
margin = 150
TY = int(os.environ.get("TY", 705))
L = Image.new('L', (W, H), 0)
title = render_cjk('人民空军', 'noto-serif-sc', 900, 470, tracking=40)
sc = (W - 2 * margin) / title.width
title = title.resize((int(title.width * sc), int(title.height * sc)), Image.LANCZOS)
place(L, title, margin, TY)
eng = render_latin('CHINA AIR FORCE', barlow(800, 200), tracking=6)
sc = (W - 2 * margin) / eng.width
eng = eng.resize((int(eng.width * sc), int(eng.height * sc)), Image.LANCZOS)
EY = TY - eng.height - 50
place(L, eng, margin, EY)
S2 = Image.new('L', (W, H), 0)
sub = render_cjk('长春航展', 'noto-serif-sc', 700, 110, tracking=26)
cs = render_latin('CHANGCHUN  AIRSHOW', barlow(600, 54), tracking=9)
SUBY = EY - sub.height - cs.height - 30 - 100
place(S2, sub, (W - sub.width) // 2, SUBY)
place(S2, cs, (W - cs.width) // 2, SUBY + sub.height + 30)

out = canvas.copy()
for M in [L, S2]:
    al = arr(M)[..., None]
    out = out * (1 - al) + NAVY * al
out += grain(out.shape[:2], 0.012, 9)[..., None]
name = os.environ.get('OUT', 'v4.png')
img(out).save(name)
thumb(name, name.replace('.png', '_300.png'))
print('SUBY', SUBY, 'EY', EY, 'title bottom', TY + title.height)
