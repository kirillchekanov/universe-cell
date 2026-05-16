import urllib.request, json, gzip, shutil
from pathlib import Path

print("Скачиваю данные белковых сетей из STRING database...")
print("Это эволюционный временной ряд биологических сетей\n")

# STRING — публичная база белок-белковых взаимодействий
# Берём 5 организмов разного эволюционного возраста
organisms = {
    "ecoli":  {"taxid": "511145", "name": "E.coli",     "age_mya": 3500, "age_label": "3.5 млрд лет"},
    "yeast":  {"taxid": "4932",   "name": "S.cerevisiae","age_mya": 1000, "age_label": "1 млрд лет"},
    "fly":    {"taxid": "7227",   "name": "D.melanogaster","age_mya": 600, "age_label": "600 млн лет"},
    "mouse":  {"taxid": "10090",  "name": "M.musculus",  "age_mya": 75,   "age_label": "75 млн лет"},
    "human":  {"taxid": "9606",   "name": "H.sapiens",   "age_mya": 0,    "age_label": "сейчас"},
}

BASE = "https://stringdb-downloads.org/download/protein.links.v12.0"

for key, org in organisms.items():
    cache = Path(f"string_{key}.txt.gz")
    if cache.exists():
        print(f"  {org['name']}: уже есть")
        continue

    url = f"{BASE}/{org['taxid']}.protein.links.v12.0.txt.gz"
    print(f"  {org['name']} ({org['age_label']})...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, cache)
        size = cache.stat().st_size / 1024 / 1024
        print(f" ✓ {size:.1f} MB")
    except Exception as e:
        print(f" ✗ {e}")

print("\n✓ Готово — запускай analyze_string.py")
