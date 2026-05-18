import json, ephem, math, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import urllib.request, csv, io, zipfile
from pathlib import Path

print("ВАЛИДАЦИЯ — проверяем найденные корреляции на новых данных")
print("Скачиваю 2018 год (модель его не видела)...")

# Скачиваем 2018 — независимая проверка
val_events = []
for month in ["01","04","07","10"]:
    url = f"http://data.gdeltproject.org/events/2018{month}01.export.CSV.zip"
    cache = Path(f"gdelt_2018{month}.zip")
    print(f"  2018/{month}...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, cache)
        EVENT_CODES = {"14":"protest","17":"coerce","18":"assault",
                       "19":"fight","20":"mass_violence","10":"demand",
                       "11":"disapprove","12":"reject","13":"threaten"}
        with zipfile.ZipFile(cache) as zf:
            for fname in zf.namelist():
                with zf.open(fname) as ff:
                    content = ff.read().decode('latin-1', errors='ignore')
                    reader = csv.reader(io.StringIO(content), delimiter='\t')
                    for row in reader:
                        if len(row) < 30: continue
                        try:
                            date_str = row[1][:8]
                            ec = row[26][:2]
                            if ec in EVENT_CODES:
                                val_events.append({
                                    "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                                    "class": EVENT_CODES[ec]
                                })
                        except: pass
        cache.unlink()
        print(f" ✓ {len(val_events)} накоплено")
    except Exception as e:
        print(f" ✗ {e}")

print(f"\nВалидационная выборка: {len(val_events)} событий")

def get_features(date_str):
    try:
        obs = ephem.Observer()
        obs.lat = "0"; obs.lon = "0"
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        obs.date = dt.strftime("%Y/%m/%d 12:00:00")
        obs.epoch = ephem.J2000
        bodies = [ephem.Sun(), ephem.Moon(), ephem.Mercury(),
                  ephem.Venus(), ephem.Mars(), ephem.Jupiter(),
                  ephem.Saturn(), ephem.Uranus(), ephem.Neptune()]
        lons = []
        for b in bodies:
            b.compute(obs)
            ecl = ephem.Ecliptic(b, epoch=ephem.J2000)
            lons.append(math.degrees(ecl.lon) % 360)
        def asp(a,b):
            d = abs(a-b)%360
            return d if d<=180 else 360-d
        return {
            "js_aspect":   asp(lons[5],lons[6]),
            "mars_jup":    asp(lons[4],lons[5]),
            "sun_mars":    asp(lons[0],lons[4]),
            "jupiter_sign": int(lons[5]/30),
            "saturn_sign":  int(lons[6]/30),
        }
    except: return None

# Вычисляем признаки для валидационной выборки
print("Вычисляю признаки...", end="", flush=True)
val_enriched = []
for e in val_events[:5000]:
    f = get_features(e["date"])
    if f:
        f["class"] = e["class"]
        val_enriched.append(f)
print(f" ✓ {len(val_enriched)}")

val_by_class = defaultdict(list)
for e in val_enriched:
    val_by_class[e["class"]].append(e)

# Проверяем 7 найденных корреляций на новых данных
print(f"\n{'='*65}")
print(f"ВАЛИДАЦИЯ: воспроизводятся ли корреляции на 2018 году?")
print(f"{'='*65}")

# Корреляции найденные на обучающей выборке
FOUND_CORRELATIONS = [
    ("fight",   "js_aspect",    "p=0.008 на train"),
    ("fight",   "jupiter_sign", "p=0.014 на train"),
    ("demand",  "mars_jup",     "p=0.020 на train"),
    ("demand",  "sun_mars",     "p=0.022 на train"),
    ("demand",  "js_aspect",    "p=0.035 на train"),
    ("demand",  "jupiter_sign", "p=0.038 на train"),
    ("fight",   "sun_mars",     "p=0.042 на train"),
]

val_all = {fn: [e[fn] for e in val_enriched]
           for fn in ["js_aspect","mars_jup","sun_mars","jupiter_sign","saturn_sign"]}

validated = []
failed = []

for cls, fn, note in FOUND_CORRELATIONS:
    if cls not in val_by_class or len(val_by_class[cls]) < 100:
        print(f"  ~ {cls:<10} {fn:<15} недостаточно данных")
        continue
    cls_vals  = [e[fn] for e in val_by_class[cls]]
    ctrl_vals = val_all[fn]
    t, p = stats.ttest_ind(cls_vals, ctrl_vals)
    cls_mean  = np.mean(cls_vals)
    ctrl_mean = np.mean(ctrl_vals)
    diff = cls_mean - ctrl_mean

    if p < 0.05:
        status = "✓✓ ПОДТВЕРЖДЕНО"
        validated.append((cls, fn, p))
    elif p < 0.1:
        status = "~ слабо"
    else:
        status = "✗ не воспроизводится"
        failed.append((cls, fn, p))

    print(f"  {status:<20} {cls:<10} {fn:<16} "
          f"Δ={diff:>+6.1f}  p={p:.4f}  ({note})")

print(f"\n{'='*65}")
print(f"Итог валидации:")
print(f"  Подтверждено: {len(validated)}/{len(FOUND_CORRELATIONS)}")
print(f"  Не воспроизвелось: {len(failed)}/{len(FOUND_CORRELATIONS)}")

if len(validated) >= 3:
    print(f"\n  ✓ ЕСТЬ РЕАЛЬНЫЙ СИГНАЛ")
    print(f"  Планетарные конфигурации статистически связаны")
    print(f"  с классами событий в независимой выборке")
elif len(validated) >= 1:
    print(f"\n  ~ СЛАБЫЙ СИГНАЛ — нужно больше данных")
else:
    print(f"\n  ✗ ПЕРЕОБУЧЕНИЕ — корреляции случайные")
    print(f"  Нужно пересмотреть подход")

# Предсказание для следующих 30 дней
print(f"\n{'='*65}")
print(f"ПРЕДСКАЗАНИЕ: следующие 30 дней (май-июнь 2026)")
print(f"{'='*65}")

# Строим простую модель на основе подтверждённых корреляций
# Для каждого дня считаем "индекс напряжённости"

print(f"\n{'Дата':<14} {'fight':>8} {'demand':>8} {'protest':>8} {'Сигнал'}")
print(f"{'-'*55}")

for days in range(0, 30):
    future = (datetime(2026,5,17)+timedelta(days=days)).strftime("%Y-%m-%d")
    f = get_features(future)
    if not f: continue

    # Простая эвристика на основе найденных корреляций
    # fight сигнал: js_aspect < 80° (близко к соединению/квадрату)
    fight_signal = 1.0 if f["js_aspect"] < 80 else 0.3

    # demand сигнал: mars_jup около 90-120°
    demand_signal = 1.0 if 70 < f["mars_jup"] < 130 else 0.3

    # protest: случайный базовый уровень
    protest_signal = 0.5

    total = fight_signal + demand_signal
    if total > 1.5:
        signal = "⚠ ПОВЫШЕН"
    elif total > 1.0:
        signal = "~ умеренный"
    else:
        signal = "  низкий"

    if days % 7 == 0 or total > 1.3:  # печатаем еженедельно и пики
        print(f"  {future}  {fight_signal:>8.2f}  {demand_signal:>8.2f}  "
              f"{protest_signal:>8.2f}  {signal}")

# Сохраняем
results = {
    "validated": len(validated),
    "total_tested": len(FOUND_CORRELATIONS),
    "confirmed_correlations": [
        {"class":c,"feature":f,"p_val":round(p,5)}
        for c,f,p in validated
    ]
}
with open("validation_results.json","w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ validation_results.json")
