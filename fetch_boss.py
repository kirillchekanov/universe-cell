import urllib.request, urllib.parse, json, time
from pathlib import Path

SDSS_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

# 20 срезов от z=0.7 до z=2.2 с шагом 0.075
slices = []
z = 0.70
while z < 2.15:
    slices.append((round(z,3), round(z+0.075,3)))
    z += 0.075

print(f"Скачиваю {len(slices)} временных срезов (z=0.70–2.20)...")
print("Это охватит 6–10 млрд лет назад\n")

ok, skip = 0, 0
for z_lo, z_hi in slices:
    name  = f"boss_{z_lo:.3f}_{z_hi:.3f}"
    cache = Path(f"sdss_{name}.json")
    if cache.exists():
        skip += 1
        continue

    # BOSS использует спектры квазаров и галактик до z=2.2
    SQL = f"""SELECT TOP 800 p.objID, p.ra, p.dec, s.z AS redshift
    FROM PhotoObj p JOIN SpecObj s ON s.bestobjid = p.objID
    WHERE s.class='GALAXY' AND s.z BETWEEN {z_lo} AND {z_hi}
    AND s.zWarning=0 AND p.clean=1"""

    params = urllib.parse.urlencode({"cmd":SQL,"format":"json"})
    print(f"  z={z_lo}–{z_hi}...", end="", flush=True)
    try:
        with urllib.request.urlopen(
                f"{SDSS_URL}?{params}", timeout=45) as r:
            data = json.loads(r.read())
        rows = data[0]["Rows"]
        if len(rows) >= 20:
            with open(cache,"w") as f: json.dump(rows,f)
            print(f" ✓ {len(rows)}")
            ok += 1
        else:
            print(f" ~ {len(rows)} (мало, пропускаю)")
        time.sleep(0.5)  # не перегружаем сервер
    except Exception as e:
        print(f" ✗ {e}")
        time.sleep(2)

print(f"\n✓ Скачано: {ok} срезов · пропущено: {skip}")
print("Запускай: python3 full_prediction.py")
