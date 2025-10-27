# utils/figs_utils.py
import os, re
from typing import Iterable, Optional
import matplotlib
import matplotlib.pyplot as plt

FIGS_DIR = os.environ.get("FIGS_DIR", "figs")

def ensure_dir(path: str = FIGS_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def slug(name: str) -> str:
    """nombre-seguro: minúsculas, sin espacios/acentos"""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    return s.strip("_") or "fig"

def safe_savefig(name: str, dpi: int = 150, folder: str = FIGS_DIR) -> str:
    """Guarda la figura actual con nombre seguro en /figs"""
    ensure_dir(folder)
    fname = slug(name) + ".png"
    path = os.path.join(folder, fname)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"✔ guardado: {path}")
    return path

def save_all_open(prefix: str = "fig", dpi: int = 150, folder: str = FIGS_DIR) -> list[str]:
    """Guarda TODAS las figuras abiertas (si existen)"""
    ensure_dir(folder)
    out = []
    nums = plt.get_fignums()
    if not nums:
        print("No hay figuras abiertas. Ejecutá antes las celdas que generan gráficos.")
        return out
    for i, num in enumerate(nums, 1):
        fig = plt.figure(num)
        name = f"{slug(prefix)}_{i:02d}.png"
        path = os.path.join(folder, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        out.append(path)
        print(f"✔ guardado: {path}")
    return out

# (Opcional) Re-encode de PNGs existentes a carpeta figs/
def reencode_pngs(src_dirs: Iterable[str], folder: str = FIGS_DIR) -> None:
    from PIL import Image
    ensure_dir(folder)
    for d in src_dirs:
        if not os.path.isdir(d): 
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".png"):
                src = os.path.join(d, fn)
                try:
                    im = Image.open(src); im.load()
                    out = os.path.join(folder, slug(os.path.splitext(fn)[0]) + ".png")
                    im.save(out, format="PNG", optimize=True)
                    print("✔ re-encodeado:", out)
                except Exception as e:
                    print("⚠ no pude abrir:", src, "->", e)
