import urllib.request, json, csv, io, gzip
import numpy as np, math
import ephem
from datetime import datetime, timedelta
from pathlib import Path

print("Загружаю GDELT — база исторических событий...")
print("(1 млн+ событий с датами, координатами и классами)\n")

# GDELT 2.0 — последние события (публичный доступ)
# Формат: каждые 15 минут новый файл
# Берём исторические данные через BigQuery export

# Альтернатива без BigQuery — GDELT Event Database v1
# Годовые файлы по одному на год
GDELT_BASE = "http://data.gdeltproject.org/events"

# Берём несколько лет для анализа
years = [2000, 2005, 2010, 2015, 2020]

# GDELT слишком большой для прямой загрузки (каждый год ~1GB)
# Используем GDELT Summary — агрегированные данные
# Или строим на основе того что уже есть

print("GDELT файлы очень большие (1GB/год).")
print("Используем умный подход: выборка ключевых дат\n")

# Вместо этого — берём известные исторические события
# и их точные даты, считаем векторы планет для каждого
# Это даст нам датасет для обучения модели

HISTORICAL_EVENTS = [
    # (дата, класс, описание)
    # Экономические кризисы
    ("2008-09-15", "economic_crisis",    "Lehman Brothers банкротство"),
    ("2000-03-10", "economic_crisis",    "Пик дотком пузыря"),
    ("1987-10-19", "economic_crisis",    "Чёрный понедельник"),
    ("2020-03-16", "economic_crisis",    "COVID обвал рынков"),
    ("1929-10-29", "economic_crisis",    "Великая депрессия"),
    ("1997-07-02", "economic_crisis",    "Азиатский кризис"),
    ("2010-05-06", "economic_crisis",    "Flash crash"),
    ("2022-01-24", "economic_crisis",    "Обвал крипто/tech"),

    # Геополитические события
    ("2001-09-11", "geopolitical",       "9/11 теракты"),
    ("1991-12-25", "geopolitical",       "Распад СССР"),
    ("1989-11-09", "geopolitical",       "Падение Берлинской стены"),
    ("2003-03-20", "geopolitical",       "Вторжение в Ирак"),
    ("2022-02-24", "geopolitical",       "Вторжение в Украину"),
    ("2011-03-11", "geopolitical",       "Фукусима/цунами Япония"),
    ("1991-08-19", "geopolitical",       "Путч в СССР"),
    ("2011-01-25", "geopolitical",       "Арабская весна Египет"),

    # Эпидемии
    ("2020-01-20", "epidemic",           "COVID-19 первые случаи вне Китая"),
    ("2009-04-17", "epidemic",           "Свиной грипп H1N1"),
    ("2003-02-26", "epidemic",           "SARS вспышка"),
    ("2014-03-23", "epidemic",           "Эбола Западная Африка"),
    ("1918-03-11", "epidemic",           "Испанский грипп"),
    ("2015-05-20", "epidemic",           "MERS Корея"),

    # Технологические прорывы
    ("1969-07-20", "tech_breakthrough",  "Луна — Аполлон 11"),
    ("2012-07-04", "tech_breakthrough",  "Бозон Хиггса открыт"),
    ("1989-03-12", "tech_breakthrough",  "WWW изобретён Бернерс-Ли"),
    ("2016-03-15", "tech_breakthrough",  "AlphaGo победил человека"),
    ("2022-11-30", "tech_breakthrough",  "Запуск ChatGPT"),
    ("1957-10-04", "tech_breakthrough",  "Спутник 1"),
    ("2023-07-05", "tech_breakthrough",  "LLaMA 2 открытый исходник"),

    # Природные катастрофы
    ("2004-12-26", "natural_disaster",   "Землетрясение Индийского океана"),
    ("2010-01-12", "natural_disaster",   "Землетрясение Гаити"),
    ("2011-03-11", "natural_disaster",   "Землетрясение Японии 9.0"),
    ("2005-08-29", "natural_disaster",   "Ураган Катрина"),
    ("1991-06-15", "natural_disaster",   "Вулкан Пинатубо"),
    ("1986-04-26", "natural_disaster",   "Чернобыль"),
]

ZODIAC = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo",
          "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]

def compute_vector(date_str):
    """Вычисляем вектор позиций планет для заданной даты"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

    obs = ephem.Observer()
    obs.lat  = "0"; obs.lon = "0"  # геоцентр
    obs.date = dt.strftime("%Y/%m/%d 12:00:00")
    obs.epoch = ephem.J2000

    bodies = [ephem.Sun(), ephem.Moon(), ephem.Mercury(),
              ephem.Venus(), ephem.Mars(), ephem.Jupiter(),
              ephem.Saturn(), ephem.Uranus(), ephem.Neptune()]

    lons = []
    for body in bodies:
        body.compute(obs)
        ecl = ephem.Ecliptic(body, epoch=ephem.J2000)
        lon = math.degrees(ecl.lon) % 360
        lons.append(lon)

    # Вектор: sin и cos каждой долготы (периодические признаки)
    vec = []
    for lon in lons:
        vec.append(math.sin(math.radians(lon)))
        vec.append(math.cos(math.radians(lon)))

    return np.array(vec), lons

print(f"Вычисляю векторы для {len(HISTORICAL_EVENTS)} исторических событий...")

dataset = []
for date_str, event_class, description in HISTORICAL_EVENTS:
    result = compute_vector(date_str)
    if result:
        vec, lons = result
        dataset.append({
            "date": date_str,
            "class": event_class,
            "description": description,
            "vector": vec,
            "lons": lons,
        })
        print(f"  ✓ {date_str}  {event_class:<20} {description[:40]}")

print(f"\n✓ Датасет: {len(dataset)} событий")

# Натальный вектор (момент рождения Кирилла)
print(f"\nВычисляю натальный вектор (1 апреля 1995)...")
natal_result = compute_vector("1995-04-01")
natal_vec, natal_lons = natal_result

# Текущий вектор (сегодня)
print(f"Вычисляю текущий вектор (май 2026)...")
now_result = compute_vector("2026-05-16")
now_vec, now_lons = now_result

# Косинусное сходство — насколько текущий момент похож на другие
def cosine_sim(a, b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\n{'='*65}")
print(f"  АНАЛИЗ СХОДСТВА ТЕКУЩЕГО МОМЕНТА С ИСТОРИЧЕСКИМИ СОБЫТИЯМИ")
print(f"  (май 2026 vs исторические конфигурации планет)")
print(f"{'='*65}")

# Сходство текущего момента с каждым событием
similarities = []
for event in dataset:
    sim = cosine_sim(now_vec, event["vector"])
    similarities.append((sim, event))

similarities.sort(key=lambda x: x[0], reverse=True)

print(f"\n  ТОП-10 наиболее похожих конфигураций на сегодня:")
print(f"  {'Сходство':>10} {'Дата':<14} {'Класс':<22} {'Событие'}")
print(f"  {'-'*75}")
for sim, event in similarities[:10]:
    print(f"  {sim:>10.4f}  {event['date']:<14} {event['class']:<22} {event['description'][:35]}")

# Анализ по классам
print(f"\n  СРЕДНЕЕ СХОДСТВО ПО КЛАССАМ СОБЫТИЙ:")
class_sims = {}
for sim, event in similarities:
    cls = event["class"]
    if cls not in class_sims: class_sims[cls] = []
    class_sims[cls].append(sim)

for cls, sims in sorted(class_sims.items(), key=lambda x: np.mean(x[1]), reverse=True):
    mean_sim = np.mean(sims)
    bar = "█" * int(mean_sim * 50)
    print(f"  {cls:<22} {mean_sim:.4f}  {bar}")

# Сходство текущего с натальным
natal_now_sim = cosine_sim(now_vec, natal_vec)
print(f"\n  СХОДСТВО ТЕКУЩЕГО МОМЕНТА С НАТАЛЬНЫМ:")
print(f"  Май 2026 vs 1 апреля 1995: {natal_now_sim:.4f}")
if natal_now_sim > 0.98:
    print(f"  → ОЧЕНЬ ВЫСОКОЕ сходство — астрологически значимый период")
elif natal_now_sim > 0.95:
    print(f"  → Высокое сходство — активный период")
else:
    print(f"  → Умеренное сходство — обычный период")

# Поиск дат максимального сходства с натальным в будущем
print(f"\n  ПОИСК ДАТ МАКСИМАЛЬНОГО СХОДСТВА С МОМЕНТОМ РОЖДЕНИЯ:")
print(f"  (следующие 2 года)")
best_dates = []
for days_ahead in range(0, 730, 7):  # каждую неделю
    future_date = (datetime(2026, 5, 16) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    result = compute_vector(future_date)
    if result:
        fvec, _ = result
        sim = cosine_sim(fvec, natal_vec)
        best_dates.append((sim, future_date))

best_dates.sort(key=lambda x: x[0], reverse=True)
print(f"\n  ТОП-5 дат максимального сходства с натальной картой:")
for sim, date in best_dates[:5]:
    print(f"  {date}  сходство={sim:.4f}")

# Сохраняем результаты
output = {
    "natal_date": "1995-04-01",
    "analysis_date": "2026-05-16",
    "natal_now_similarity": round(float(natal_now_sim), 4),
    "top_similar_events": [
        {"similarity": round(float(s),4), "date": e["date"],
         "class": e["class"], "description": e["description"]}
        for s, e in similarities[:10]
    ],
    "class_similarities": {
        cls: round(float(np.mean(sims)), 4)
        for cls, sims in class_sims.items()
    },
    "peak_similarity_dates": [
        {"date": d, "similarity": round(float(s),4)}
        for s, d in best_dates[:5]
    ]
}
with open("gdelt_analysis.json","w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ gdelt_analysis.json сохранён")
