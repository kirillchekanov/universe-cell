import urllib.request, csv, io, zipfile, os
import numpy as np
from pathlib import Path
from datetime import datetime

print("Скачиваю GDELT Event Database — реальные исторические события")
print("Формат: 57 полей включая дату, тип события, страну, координаты\n")

# GDELT хранит данные по годам — каждый файл ~100-300MB
# Скачиваем только нужные поля через умный парсинг

# Начнём с одного года для теста
YEARS = [2010, 2015, 2020]  # три года = ~50k событий

# GDELT Event Codes (CAMEO) которые нас интересуют
# https://www.gdeltproject.org/data/documentation/CAMEO.Manual.GDELT.pdf
EVENT_CODES = {
    "14": "protest",           # Протесты
    "17": "coerce",            # Принуждение
    "18": "assault",           # Нападение
    "19": "fight",             # Вооружённый конфликт
    "20": "mass_violence",     # Массовое насилие
    "10": "demand",            # Требования
    "11": "disapprove",        # Неодобрение
    "12": "reject",            # Отказ
    "13": "threaten",          # Угрозы
}

BASE_URL = "http://data.gdeltproject.org/events"

all_events = []

for year in YEARS:
    url = f"{BASE_URL}/{year}.zip"
    cache = Path(f"gdelt_{year}.zip")
    csv_cache = Path(f"gdelt_{year}_events.csv")

    if csv_cache.exists():
        print(f"  {year}: загружаю из кэша...", end="", flush=True)
        with open(csv_cache) as f:
            reader = csv.reader(f)
            events = list(reader)
        print(f" ✓ {len(events)} событий")
        all_events.extend(events)
        continue

    print(f"  Скачиваю {year} (~200MB)...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, cache)
        print(f" ✓ распаковываю...", end="", flush=True)

        # Распаковываем и парсим
        year_events = []
        with zipfile.ZipFile(cache) as zf:
            for fname in zf.namelist():
                if fname.endswith('.csv') or fname.endswith('.CSV'):
                    with zf.open(fname) as f:
                        content = f.read().decode('latin-1', errors='ignore')
                        reader = csv.reader(io.StringIO(content), delimiter='\t')
                        for row in reader:
                            if len(row) < 30: continue
                            try:
                                date_str = row[1][:8]  # YYYYMMDD
                                event_code = row[26][:2]  # первые 2 цифры CAMEO
                                if event_code in EVENT_CODES:
                                    year_events.append([
                                        date_str,
                                        EVENT_CODES[event_code],
                                        row[26],
                                        row[51] if len(row) > 51 else "0",  # Goldstein scale
                                    ])
                            except: pass

        # Сохраняем только нужные события
        with open(csv_cache, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(year_events)

        cache.unlink()  # удаляем zip
        print(f" ✓ {len(year_events)} событий")
        all_events.extend(year_events)

    except Exception as e:
        print(f" ✗ {e}")
        print(f"    Пробую альтернативный источник...")
        # Альтернатива — берём данные за один месяц
        for month in ["01","04","07","10"]:
            url2 = f"{BASE_URL}/{year}{month}01.export.CSV.zip"
            try:
                cache2 = Path(f"gdelt_{year}{month}.zip")
                urllib.request.urlretrieve(url2, cache2)
                month_events = []
                with zipfile.ZipFile(cache2) as zf:
                    for fname in zf.namelist():
                        with zf.open(fname) as ff:
                            content = ff.read().decode('latin-1', errors='ignore')
                            reader = csv.reader(io.StringIO(content), delimiter='\t')
                            for row in reader:
                                if len(row) < 30: continue
                                try:
                                    date_str = row[1][:8]
                                    event_code = row[26][:2]
                                    if event_code in EVENT_CODES:
                                        month_events.append([
                                            date_str, EVENT_CODES[event_code],
                                            row[26],
                                            row[51] if len(row) > 51 else "0",
                                        ])
                                except: pass
                cache2.unlink()
                print(f"    {year}/{month}: {len(month_events)} событий")
                all_events.extend(month_events)
            except Exception as e2:
                print(f"    {year}/{month}: ✗ {e2}")

print(f"\n✓ Всего событий: {len(all_events)}")

if len(all_events) > 100:
    # Распределение по классам
    from collections import Counter
    classes = Counter([e[1] for e in all_events])
    print(f"\nРаспределение по классам:")
    for cls, count in classes.most_common():
        print(f"  {cls:<20} {count:>8,}")

    # Сохраняем
    import json
    with open("gdelt_events.json","w") as f:
        json.dump(all_events[:10000], f)  # первые 10k
    print(f"\n✓ gdelt_events.json — {min(len(all_events),10000)} событий")
    print(f"  Теперь запускай: python3 correlation_test.py")
else:
    print(f"\n  Мало данных — GDELT сервер недоступен")
    print(f"  Используем расширенный вручную датасет")
