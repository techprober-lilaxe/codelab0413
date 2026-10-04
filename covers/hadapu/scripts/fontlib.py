import os, glob, json
from fontTools.ttLib import TTFont
from PIL import ImageFont
ROOT=os.path.expanduser('~/fonts')
_cache={}
def _ttf(woff):
    out=os.path.join(ROOT,'ttf',os.path.basename(woff).replace('.woff2','.ttf'))
    if not os.path.exists(out):
        f=TTFont(woff); f.flavor=None; f.save(out)
    return out
def latin(family_dir, name):
    return _ttf(glob.glob(f'{ROOT}/fontsource-{family_dir}-*/files/{name}')[0])
_cmaps={}
def cjk_path(pkg, weight, ch):
    key=(pkg,weight)
    if key not in _cmaps:
        lst=[]
        for w in sorted(glob.glob(f'{ROOT}/fontsource-{pkg}-*/files/{pkg}-*-{weight}-normal.woff2')):
            lst.append((w,set(TTFont(w).getBestCmap().keys())))
        _cmaps[key]=lst
    for w,cm in _cmaps[key]:
        if ord(ch) in cm: return _ttf(w)
    raise KeyError(ch)
def cjk_font(pkg, weight, ch, size):
    return ImageFont.truetype(cjk_path(pkg,weight,ch), size)
_gcm = {}
def glob_font(pattern, ch, size):
    """any split CJK webfont: pattern is a glob of woff2 files under ~/fonts"""
    if pattern not in _gcm:
        _gcm[pattern] = [(w, set(TTFont(w).getBestCmap().keys())) for w in sorted(glob.glob(f'{ROOT}/{pattern}'))]
    for w, cm in _gcm[pattern]:
        if ord(ch) in cm:
            return ImageFont.truetype(_ttf(w), size)
    raise KeyError(ch)
