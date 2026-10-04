import sys, os
sys.path.insert(0, os.path.expanduser('~/fonts'))
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import fontlib

W, H = 2160, 2880
HERE = os.path.dirname(os.path.abspath(__file__))


def load(i):
    return Image.open(os.path.join(HERE, f'p{i}.png')).convert('RGB')


def arr(im):
    return np.asarray(im).astype(np.float32) / 255.0


def img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def barlow(weight, size):
    return ImageFont.truetype(
        fontlib.latin('barlow-condensed', f'barlow-condensed-latin-{weight}-normal.woff2'), size)


def ink_bbox(mask_img):
    a = np.asarray(mask_img)
    ys, xs = np.nonzero(a > 8)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def render_latin(text, font, tracking=0):
    """Render text into an L mask cropped to ink; tracking in px between glyphs."""
    glyphs = []
    for ch in text:
        if ch == ' ':
            glyphs.append(None)
            continue
        bb = font.getbbox(ch)
        m = Image.new('L', (bb[2] + 40, font.size * 2), 0)
        ImageDraw.Draw(m).text((20, 0), ch, font=font, fill=255)
        glyphs.append((m, font.getlength(ch)))
    space = font.getlength(' ')
    asc, desc = font.getmetrics()
    total = sum(g[1] if g else space for g in glyphs) + tracking * (len(text) - 1)
    out = Image.new('L', (int(total) + 80, font.size * 2), 0)
    x = 0
    for g in glyphs:
        if g is None:
            x += space + tracking
            continue
        out.paste(g[0], (int(round(x)) + 20, 0), g[0])
        x += g[1] + tracking
    return out.crop(ink_bbox(out))


def render_cjk(text, pkg, weight, size, tracking=0, by_ink=True):
    """Render CJK text char by char (fontsource subsets); spaces use size*0.5."""
    pieces = []
    for ch in text:
        if ch == ' ':
            pieces.append(None)
            continue
        f = fontlib.glob_font(pkg[5:], ch, size) if pkg.startswith('glob:') else fontlib.cjk_font(pkg, weight, ch, size)
        m = Image.new('L', (size * 2, size * 2), 0)
        ImageDraw.Draw(m).text((size // 2, size // 4), ch, font=f, fill=255)
        pieces.append((m, f.getlength(ch)))
    # vertical: keep common baseline -> crop all with the same vertical window
    allm = [p[0] for p in pieces if p]
    stack = np.max([np.asarray(m) for m in allm], axis=0)
    ys = np.nonzero(stack.max(1) > 8)[0]
    y0, y1 = ys.min(), ys.max() + 1
    out_parts = []
    for p in pieces:
        if p is None:
            out_parts.append(None)
            continue
        m, adv = p
        a = np.asarray(m)
        if by_ink:
            xs = np.nonzero(a.max(0) > 8)[0]
            out_parts.append(m.crop((xs.min(), y0, xs.max() + 1, y1)))
        else:
            out_parts.append(m.crop((size // 2, y0, size // 2 + int(adv), y1)))
    tw = sum(p.width if p else size // 2 for p in out_parts) + tracking * (len(out_parts) - 1)
    out = Image.new('L', (tw, y1 - y0), 0)
    x = 0
    for p in out_parts:
        if p is None:
            x += size // 2 + tracking
            continue
        out.paste(p, (x, 0))
        x += p.width + tracking
    return out


def place(canvas_mask, m, x, y):
    canvas_mask.paste(m, (int(x), int(y)), m)


def grain(shape, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, sigma, shape).astype(np.float32)


def thumb(path, out, w=300):
    im = Image.open(path)
    im.resize((w, int(im.height * w / im.width)), Image.LANCZOS).save(out)
