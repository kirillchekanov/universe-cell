import json, math, ephem
from datetime import datetime

print("НАТАЛЬНЫЕ МОМЕНТЫ СТРАН — персональный риск-индекс\n")

# Даты основания/провозглашения государств
COUNTRIES = {
    "USA":           "1776-07-04",
    "Russia":        "1991-12-25",
    "China":         "1949-10-01",
    "Germany":       "1990-10-03",
    "France":        "1958-10-04",
    "UK":            "1801-01-01",
    "Japan":         "1947-05-03",
    "Israel":        "1948-05-14",
    "India":         "1947-08-15",
    "Pakistan":      "1947-08-14",
    "Ukraine":       "1991-08-24",
    "Iran":          "1979-04-01",
    "Saudi Arabia":  "1932-09-23",
    "Turkey":        "1923-10-29",
    "Brazil":        "1889-11-15",
    "EU":            "1993-11-01",
    "UN":            "1945-10-24",
    "NATO":          "1949-04-04",
    "North Korea":   "1948-09-09",
    "South Korea":   "1948-08-15",
}

def get_planet_lons(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs = ephem.Observer()
    obs.date = dt.strftime("%Y/%m/%d")
    bodies = [ephem.Sun(), ephem.Moon(), ephem.Mercury(), ephem.Venus(),
              ephem.Mars(), ephem.Jupiter(), ephem.Saturn(),
              ephem.Uranus(), ephem.Neptune()]
    lons = {}
    names = ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune"]
    for name, b in zip(names, bodies):
        b.compute(obs)
        lons[name] = math.degrees(float(b.hlong)) % 360
    return lons

def angular_distance(a, b):
    """Минимальный угол между двумя позициями"""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

def natal_delta(natal_lons, current_lons):
    """Среднее отклонение текущей конфигурации от натальной"""
    deltas = []
    for planet in natal_lons:
        if planet in current_lons:
            d = angular_distance(natal_lons[planet], current_lons[planet])
            deltas.append(d)
    return sum(deltas) / len(deltas) if deltas else 180

# Вычисляем натальные конфигурации
print("Натальные конфигурации:")
natal_configs = {}
for country, date in COUNTRIES.items():
    try:
        lons = get_planet_lons(date)
        natal_configs[country] = {"date": date, "lons": lons}
        sun_lon = lons["Sun"]
        jup_lon = lons["Jupiter"]
        sat_lon = lons["Saturn"]
        print(f"  {country:<15} {date}  ☀️{sun_lon:>6.1f}° ♃{jup_lon:>6.1f}° ♄{sat_lon:>6.1f}°")
    except Exception as e:
        print(f"  {country}: ошибка {e}")

# Прогноз для каждой страны на ключевые даты
print(f"\n{'='*60}")
print("РИСК-ИНДЕКС ПО СТРАНАМ (0=максимальное совпадение с натальным)")
print("Чем МЕНЬШЕ дельта — тем БЛИЖЕ к натальному моменту")
print("="*60)

forecast_dates = [
    ("2026-08-29", "август 2026"),
    ("2027-03-21", "март 2027"),
    ("2027-09-23", "осень 2027"),
    ("2027-12-22", "декабрь 2027"),
    ("2028-12-16", "конец горизонта"),
]

country_forecasts = {}

for fdate, flabel in forecast_dates:
    current_lons = get_planet_lons(fdate)
    deltas = {}
    for country, config in natal_configs.items():
        delta = natal_delta(config["lons"], current_lons)
        deltas[country] = delta

    sorted_countries = sorted(deltas.items(), key=lambda x: x[1])

    print(f"\n{fdate} ({flabel}):")
    print(f"  {'Страна':<16} {'Δ°':>6}  Близость к натальному")
    for country, delta in sorted_countries[:10]:
        # Нормализуем: 0° = идеальное совпадение, 90° = среднее
        closeness = max(0, (90 - delta) / 90)
        bar = "▓" * int(closeness * 20)
        flag = " ⚠" if delta < 30 else ""
        print(f"  {country:<16} {delta:>5.1f}°  {bar}{flag}")

    country_forecasts[fdate] = {c: round(d, 2) for c, d in deltas.items()}

# Сохраняем
with open("natal_forecast.json", "w") as f:
    json.dump({
        "countries": COUNTRIES,
        "natal_configs": {k: {"date":v["date"],
                              "lons":{p:round(l,2) for p,l in v["lons"].items()}}
                         for k,v in natal_configs.items()},
        "forecast": country_forecasts
    }, f, indent=2, ensure_ascii=False)

print(f"\n✓ natal_forecast.json")
print("Страны с наименьшей дельтой = ближе всего к своему 'натальному' моменту")
