import urllib.request, json, numpy as np
from datetime import datetime

# Дата рождения: 1 апреля 1995, 17:45, Лондон (UTC+1 летнее время = 16:45 UTC)
BIRTH_DATE = "1995-04-01"
BIRTH_TIME = "16:45:00"  # UTC
BIRTH_LAT  = 51.5074
BIRTH_LON  = -0.1278

print(f"Астрономический контекст рождения")
print(f"Дата: {BIRTH_DATE} {BIRTH_TIME} UTC (17:45 Лондон BST)")
print(f"Координаты: {BIRTH_LAT}N {abs(BIRTH_LON)}W\n")

# NASA JPL Horizons API — точные позиции планет
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Объекты: Солнце, Луна, планеты
objects = {
    "10":  "Солнце",
    "301": "Луна",
    "199": "Меркурий",
    "299": "Венера",
    "499": "Марс",
    "599": "Юпитер",
    "699": "Сатурн",
    "799": "Уран",
    "899": "Нептун",
    "999": "Плутон",
}

positions = {}
print("Запрашиваю позиции планет у NASA JPL...")

for obj_id, name in objects.items():
    params = {
        "format":      "json",
        "COMMAND":     f"'{obj_id}'",
        "OBJ_DATA":    "NO",
        "MAKE_EPHEM":  "YES",
        "EPHEM_TYPE":  "OBSERVER",
        "CENTER":      f"coord@399",
        "COORD_TYPE":  "GEODETIC",
        "SITE_COORD":  f"{BIRTH_LON},{BIRTH_LAT},0",
        "START_TIME":  f"'{BIRTH_DATE} {BIRTH_TIME}'",
        "STOP_TIME":   f"'{BIRTH_DATE} 17:00:00'",
        "STEP_SIZE":   "1h",
        "QUANTITIES":  "1,9,20",  # RA/Dec, геоцентр. расст., угол
    }
    url = HORIZONS + "?" + urllib.parse.urlencode(params)
    try:
        import urllib.parse
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        result = data.get("result","")
        # Парсим первую строку данных
        lines = [l for l in result.split("\n") if "$$SOE" in result]
        # Ищем строку с данными
        in_data = False
        for line in result.split("\n"):
            if "$$SOE" in line: in_data = True; continue
            if "$$EOE" in line: break
            if in_data and len(line.strip()) > 20:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        ra  = float(parts[2])  # RA градусы
                        dec = float(parts[3])  # Dec градусы
                        # Переводим RA в знак зодиака
                        zodiac_names = ["Овен","Телец","Близнецы","Рак",
                                       "Лев","Дева","Весы","Скорпион",
                                       "Стрелец","Козерог","Водолей","Рыбы"]
                        # Эклиптическая долгота (приближение)
                        lon_ecl = ra  # упрощение
                        sign_idx = int(lon_ecl / 30) % 12
                        degree   = lon_ecl % 30
                        positions[name] = {
                            "ra": round(ra, 2),
                            "dec": round(dec, 2),
                            "sign": zodiac_names[sign_idx],
                            "degree": round(degree, 1),
                        }
                        print(f"  ✓ {name:<12} RA={ra:>8.2f}°  Dec={dec:>+7.2f}°  → {zodiac_names[sign_idx]} {degree:.1f}°")
                        break
                    except: pass
    except Exception as e:
        print(f"  ~ {name:<12} ошибка: {e}")

print(f"\nПозиций получено: {len(positions)}/10")
