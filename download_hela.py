import urllib.request, gzip, shutil
from pathlib import Path

URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-11756/map/emd_11756.map.gz"
GZ  = Path("emd_11756.map.gz")
MAP = Path("emd_11756.map")

print("Скачиваю полную клетку HeLa (~500 MB)...")
print("Это займёт 5-15 минут в зависимости от интернета\n")

def progress(n, bs, total):
    mb = n*bs/1024/1024
    pct = min(mb/(total/1024/1024)*100, 100)
    bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
    print(f"\r  [{bar}] {pct:.0f}%  {mb:.0f}/{total/1024/1024:.0f} MB", end="", flush=True)

urllib.request.urlretrieve(URL, GZ, reporthook=progress)
print("\n  ✓ скачано")

print("  Распаковываю...")
with gzip.open(GZ,"rb") as f_in, open(MAP,"wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
GZ.unlink()
print(f"  ✓ готово: {MAP} ({MAP.stat().st_size/1024/1024:.0f} MB)")
