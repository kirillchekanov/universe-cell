import urllib.request, urllib.parse, json
from pathlib import Path

SDSS_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

slices = {
    "z_high": (0.10, 0.30),
    "z_mid":  (0.05, 0.10),
}

for name, (z_lo, z_hi) in slices.items():
    cache = Path(f"sdss_{name}.json")
    if cache.exists():
        print(f"  {name}: уже есть")
        continue

    SQL = f"""SELECT TOP 2000 p.objID, p.ra, p.dec, s.z AS redshift
    FROM PhotoObj p JOIN SpecObj s ON s.bestobjid = p.objID
    WHERE s.class='GALAXY' AND s.z BETWEEN {z_lo} AND {z_hi}
    AND s.zWarning=0 AND p.clean=1"""

    params = urllib.parse.urlencode({"cmd": SQL, "format": "json"})
    print(f"  Скачиваю {name} (z={z_lo}–{z_hi})...", end="", flush=True)
    try:
        with urllib.request.urlopen(
            f"{SDSS_URL}?{params}", timeout=45) as r:
            data = json.loads(r.read())
        rows = data[0]["Rows"]
        with open(cache, "w") as f:
            json.dump(rows, f)
        print(f" ✓ {len(rows)} галактик")
    except Exception as e:
        print(f" ✗ {e}")

print("Готово — теперь запускай time_machine2.py с новыми файлами")
