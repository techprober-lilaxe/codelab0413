import sys
from common import *
ims = []
for p in sys.argv[1:]:
    im = Image.open(p).convert('RGB').resize((375, 500), Image.LANCZOS)
    a = arr(im)
    g = smoothstep(0.72, 1.0, np.arange(500) / 500)[:, None, None]
    a = a * (1 - 0.45 * g)
    im = img(a); d = ImageDraw.Draw(im)
    f = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 22) if os.path.exists('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc') else None
    d.rounded_rectangle((8, 8, 70, 38), 6, fill=(254, 214, 60))
    d.text((16, 11), '置顶', font=f, fill=(40, 30, 0))
    d.text((14, 462), '▷ 5.8万', font=f, fill=(255, 255, 255))
    ims.append(im)
out = Image.new('RGB', (len(ims) * 375 + (len(ims) - 1) * 4, 500), (20, 20, 20))
for i, im in enumerate(ims): out.paste(im, (i * 379, 0))
out.save('douyin_sim.png')
