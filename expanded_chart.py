import ephem, math, json
import numpy as np

observer = ephem.Observer()
observer.lat  = "51.5074"
observer.lon  = "-0.1278"
observer.date = "1995/04/01 16:45:00"
observer.epoch = ephem.J2000

ZODIAC = ["Овен","Телец","Близнецы","Рак","Лев","Дева",
          "Весы","Скорпион","Стрелец","Козерог","Водолей","Рыбы"]

def get_ecl_lon(body):
    body.compute(observer)
    ecl = ephem.Ecliptic(body, epoch=ephem.J2000)
    lon = math.degrees(ecl.lon) % 360
    return lon

# Классические планеты
classic = {
    "Солнце":   ephem.Sun(),
    "Луна":     ephem.Moon(),
    "Меркурий": ephem.Mercury(),
    "Венера":   ephem.Venus(),
    "Марс":     ephem.Mars(),
    "Юпитер":   ephem.Jupiter(),
    "Сатурн":   ephem.Saturn(),
    "Уран":     ephem.Uranus(),
    "Нептун":   ephem.Neptune(),
}

# Малые тела — ephem поддерживает через TLE/элементы
# Но для точности лучше NASA JPL Horizons batch API
# Используем ephem для доступных объектов

print("РАСШИРЕННАЯ АСТРОКАРТА: 1 апреля 1995 · 17:45 · Лондон")
print("="*60)
print(f"\n{'Объект':<18} {'Знак':<14} {'Долгота':>10}°")
print("-"*44)

all_positions = {}

for name, body in classic.items():
    lon = get_ecl_lon(body)
    sign = ZODIAC[int(lon/30)%12]
    deg  = lon % 30
    all_positions[name] = {"lon": round(lon,2), "sign": sign, "deg": round(deg,1)}
    print(f"  {name:<16} {sign:<14} {lon:>10.2f}°")

# Карликовые планеты и астероиды через JPL Horizons batch
# Список ключевых малых тел с их JPL ID
minor_bodies = {
    "Хирон":    "2060",   # Chiron — кентавр
    "Церера":   "1",      # Ceres — карликовая планета
    "Паллада":  "2",      # Pallas
    "Юнона":    "3",      # Juno
    "Веста":    "4",      # Vesta
    "Эрида":    "136199", # Eris
    "Седна":    "90377",  # Sedna
    "Хаумеа":  "136108", # Haumea
    "Макемаке": "136472", # Makemake
    "Квавар":   "50000",  # Quaoar
}

print(f"\n  Запрашиваю малые тела через NASA JPL Horizons...")
import urllib.request, urllib.parse, time

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

for name, obj_id in minor_bodies.items():
    try:
        params = {
            "format":     "text",
            "COMMAND":    f"'{obj_id}'",
            "OBJ_DATA":   "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "VECTORS",
            "CENTER":     "500@10",  # гелиоцентр
            "START_TIME": "1995-Apr-01",
            "STOP_TIME":  "1995-Apr-02",
            "STEP_SIZE":  "1d",
            "VEC_TABLE":  "2",
        }
        url = HORIZONS_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=10) as r:
            result = r.read().decode()

        # Парсим X,Y,Z координаты
        in_data = False
        for line in result.split("\n"):
            if "$$SOE" in line: in_data = True; continue
            if "$$EOE" in line: break
            if in_data and "X =" in line:
                # X= ... Y= ... Z= ...
                parts = line.replace("X =","").replace("Y =","").replace("Z =","").split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    # Гелиоцентрическая эклиптическая долгота
                    lon = math.degrees(math.atan2(y, x)) % 360
                    sign = ZODIAC[int(lon/30)%12]
                    deg  = lon % 30
                    all_positions[name] = {
                        "lon": round(lon,2),
                        "sign": sign,
                        "deg": round(deg,1)
                    }
                    print(f"  {name:<16} {sign:<14} {lon:>10.2f}°")
                    break
        time.sleep(0.3)
    except Exception as e:
        print(f"  {name:<16} ✗ {e}")

print(f"\n{'='*60}")
print(f"  Всего объектов: {len(all_positions)}")

# Строим вектор состояния системы
lons = [p["lon"] for p in all_positions.values()]
names_list = list(all_positions.keys())

# Спектральный анализ вектора позиций
lons_rad = np.array([math.radians(l) for l in lons])
# Синусы и косинусы — периодические признаки
sin_lons = np.sin(lons_rad)
cos_lons = np.cos(lons_rad)

print(f"\nВектор состояния солнечной системы:")
print(f"  Размерность: {len(lons)*2} признаков (sin+cos каждой планеты)")
print(f"  Среднее sin: {sin_lons.mean():.4f}")
print(f"  Среднее cos: {cos_lons.mean():.4f}")

# Индекс "напряжённости" — насколько планеты разбросаны
# vs сконцентрированы в одном секторе
resultant = math.sqrt(sin_lons.mean()**2 + cos_lons.mean()**2)
spread = 1 - resultant  # 0 = все в одной точке, 1 = равномерно разбросаны
print(f"  Индекс разброса: {spread:.4f}")
print(f"  {'Планеты равномерно распределены' if spread > 0.7 else 'Планеты сконцентрированы'}")

# Квадранты
q_count = [0,0,0,0]
for lon in lons:
    q_count[int(lon/90)] += 1
print(f"\n  Распределение по квадрантам:")
for i, q in enumerate(q_count):
    quadrant = ["I (0-90°)","II (90-180°)","III (180-270°)","IV (270-360°)"][i]
    bar = "█" * q
    print(f"    {quadrant}: {bar} ({q} объектов)")

# Сохраняем расширенную карту
with open("birth_chart_extended.json","w") as f:
    json.dump({
        "birth": {"date":"1995-04-01","time":"17:45 BST","location":"London"},
        "positions": all_positions,
        "state_vector": {
            "spread_index": round(spread,4),
            "resultant": round(resultant,4),
            "n_objects": len(all_positions),
        }
    }, f, indent=2, ensure_ascii=False)

print(f"\n✓ birth_chart_extended.json")
print(f"\nСледующий шаг: загрузить GDELT и найти")
print(f"корреляцию между векторами позиций и классами событий")
