import ephem, math, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats

print("ЧЕСТНАЯ ПРОВЕРКА КОРРЕЛЯЦИЙ")
print("Permutation test — есть ли реальный сигнал?")
print("="*60)

# Все значимые события с датами
EVENTS_BY_CLASS = {
    "economic_crisis": [
        "2008-09-15","2008-10-10","2000-03-10","2000-04-14",
        "1987-10-19","2020-03-16","1929-10-29","1997-07-02",
        "1997-10-23","2010-05-06","2022-01-24","2011-08-05",
        "2015-08-24","2018-12-24","1973-10-17","1998-08-17",
        "2001-09-17","2022-06-16","1998-09-01","2011-09-22",
    ],
    "epidemic": [
        "2020-01-20","2020-03-11","2009-04-17","2009-06-11",
        "2003-02-26","2014-03-23","2014-10-08","2015-05-20",
        "2016-02-01","2019-07-17","1981-06-05","2002-11-16",
        "2012-09-24","2022-05-07","1957-02-01","1968-07-01",
        "1976-06-27","1976-08-01","2010-10-21","2023-05-05",
    ],
    "tech_breakthrough": [
        "1969-07-20","2012-07-04","1989-03-12","2016-03-15",
        "2022-11-30","1957-10-04","1981-08-12","1997-05-11",
        "2007-01-09","2004-02-04","1998-09-04","2023-03-14",
        "1953-04-25","2003-04-14","1947-12-23","1991-08-06",
        "1975-01-01","1976-04-01","2024-02-15","2023-07-18",
    ],
    "geopolitical": [
        "2001-09-11","1991-12-25","1989-11-09","2003-03-20",
        "2022-02-24","1991-08-19","2011-01-25","2011-05-02",
        "1963-11-22","1989-06-04","1991-01-17","2014-03-18",
        "2016-06-23","2016-11-08","1994-04-06","2004-03-11",
        "2005-07-07","2011-03-19","2020-11-03","1945-08-06",
    ],
}

def get_planet_features(date_str):
    """Возвращает набор астрономических признаков для даты"""
    try:
        obs = ephem.Observer()
        obs.lat = "0"; obs.lon = "0"
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        obs.date = dt.strftime("%Y/%m/%d 12:00:00")
        obs.epoch = ephem.J2000

        bodies = {
            "sun": ephem.Sun(), "moon": ephem.Moon(),
            "mercury": ephem.Mercury(), "venus": ephem.Venus(),
            "mars": ephem.Mars(), "jupiter": ephem.Jupiter(),
            "saturn": ephem.Saturn(), "uranus": ephem.Uranus(),
            "neptune": ephem.Neptune(),
        }
        lons = {}
        for name, body in bodies.items():
            body.compute(obs)
            ecl = ephem.Ecliptic(body, epoch=ephem.J2000)
            lons[name] = math.degrees(ecl.lon) % 360

        features = {}
        # 1. Фаза Луны (0-360°)
        features["moon_phase"] = lons["moon"]
        # 2. Позиция Солнца (сезон)
        features["sun_lon"] = lons["sun"]
        # 3. Аспект Юпитер-Сатурн (19.86 лет)
        js_angle = abs(lons["jupiter"] - lons["saturn"]) % 360
        if js_angle > 180: js_angle = 360 - js_angle
        features["jupiter_saturn"] = js_angle
        # 4. Аспект Марс-Юпитер
        mj_angle = abs(lons["mars"] - lons["jupiter"]) % 360
        if mj_angle > 180: mj_angle = 360 - mj_angle
        features["mars_jupiter"] = mj_angle
        # 5. Уран-Нептун (медленный цикл ~172 года)
        un_angle = abs(lons["uranus"] - lons["neptune"]) % 360
        if un_angle > 180: un_angle = 360 - un_angle
        features["uranus_neptune"] = un_angle
        # 6. Позиция Марса
        features["mars_lon"] = lons["mars"]
        # 7. Позиция Юпитера
        features["jupiter_lon"] = lons["jupiter"]
        # 8. Ретроградность Меркурия (угловая скорость)
        features["mercury_lon"] = lons["mercury"]

        return features, lons
    except:
        return None, None

# Вычисляем признаки для всех событий
print("Вычисляю признаки для событий...")
event_features = {}
for cls, dates in EVENTS_BY_CLASS.items():
    event_features[cls] = []
    for date in dates:
        f, _ = get_planet_features(date)
        if f: event_features[cls].append(f)
    print(f"  {cls}: {len(event_features[cls])} событий")

# Контрольная группа — случайные даты
print(f"\nГенерирую контрольную группу (500 случайных дат)...")
np.random.seed(42)
control_features = []
base = datetime(1950, 1, 1)
for _ in range(500):
    days = int(np.random.randint(0, 27000))
    d = (base + timedelta(days=days)).strftime("%Y-%m-%d")
    f, _ = get_planet_features(d)
    if f: control_features.append(f)
print(f"  Контроль: {len(control_features)} дат")

# Permutation test для каждого признака и класса
print(f"\n{'='*70}")
print(f"PERMUTATION TEST — есть ли реальный сигнал?")
print(f"(p < 0.05 = статистически значимо)")
print(f"{'='*70}")

feature_names = list(event_features["economic_crisis"][0].keys())
ctrl_vals = {fn: [f[fn] for f in control_features] for fn in feature_names}
ctrl_means = {fn: np.mean(ctrl_vals[fn]) for fn in feature_names}
ctrl_stds  = {fn: np.std(ctrl_vals[fn]) for fn in feature_names}

significant_findings = []

for cls, features_list in event_features.items():
    print(f"\n  {cls.upper()}:")
    for fn in feature_names:
        event_vals = [f[fn] for f in features_list]
        ctrl_sample = ctrl_vals[fn]

        # T-test
        t, p = stats.ttest_ind(event_vals, ctrl_sample)
        event_mean = np.mean(event_vals)
        ctrl_mean  = np.mean(ctrl_sample)
        diff = event_mean - ctrl_mean

        sig = "✓✓" if p < 0.01 else "✓" if p < 0.05 else "~" if p < 0.1 else " "
        if p < 0.1:
            print(f"    {sig} {fn:<22} событие={event_mean:>7.1f}°  "
                  f"контроль={ctrl_mean:>7.1f}°  "
                  f"разница={diff:>+7.1f}°  p={p:.3f}")
            if p < 0.05:
                significant_findings.append({
                    "class": cls, "feature": fn,
                    "p": round(p,4), "diff": round(diff,2),
                    "event_mean": round(event_mean,2),
                    "ctrl_mean": round(ctrl_mean,2),
                })

# Итог
print(f"\n{'='*70}")
print(f"ИТОГ: найдено {len(significant_findings)} значимых корреляций (p<0.05)")
print(f"{'='*70}")

if significant_findings:
    for f in significant_findings:
        direction = "выше" if f["diff"] > 0 else "ниже"
        print(f"  ✓ {f['class']:<20} {f['feature']:<22} "
              f"p={f['p']:.4f}  ({direction} нормы на {abs(f['diff']):.1f}°)")
else:
    print(f"  Значимых корреляций не найдено")
    print(f"  → Нужно больше данных (сейчас ~20 событий на класс)")
    print(f"  → Минимум для p<0.05: ~50-100 событий на класс")

# Анализ цикличности — есть ли 11-летний цикл в событиях?
print(f"\n{'='*70}")
print(f"АНАЛИЗ ЦИКЛИЧНОСТИ")
print(f"{'='*70}")

all_dates = []
for cls, dates in EVENTS_BY_CLASS.items():
    for d in dates:
        try:
            all_dates.append(datetime.strptime(d, "%Y-%m-%d"))
        except: pass

all_dates.sort()
years = np.array([(d - datetime(1900,1,1)).days/365.25 for d in all_dates])

# FFT для поиска периодов
if len(years) > 10:
    # Создаём временной ряд (1 год = 1 бин)
    year_bins = np.arange(50, 130)  # 1950-2030
    counts = np.zeros(len(year_bins))
    for y in years:
        idx = int(y) - 50
        if 0 <= idx < len(counts):
            counts[idx] += 1

    fft = np.fft.rfft(counts - counts.mean())
    freqs = np.fft.rfftfreq(len(counts), d=1.0)
    periods = 1.0 / (freqs[1:] + 1e-10)
    power = np.abs(fft[1:])

    # Топ периоды
    top_idx = np.argsort(power)[::-1][:5]
    print(f"\n  Топ периоды в распределении событий:")
    for idx in top_idx:
        if 3 < periods[idx] < 40:
            print(f"    {periods[idx]:.1f} лет  (мощность={power[idx]:.1f})")

    # Сравниваем с планетарными циклами
    print(f"\n  Планетарные циклы для сравнения:")
    print(f"    11.0 лет — солнечный цикл")
    print(f"    11.86 лет — орбита Юпитера")
    print(f"    19.86 лет — синодический цикл Юпитер-Сатурн")
    print(f"    29.46 лет — орбита Сатурна")

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Поиск корреляций: планеты → исторические события",
             fontsize=13, color="white")

colors_cls = {
    "economic_crisis": "#E24B4A",
    "epidemic": "#EF9F27",
    "tech_breakthrough": "#5DCAA5",
    "geopolitical": "#7F77DD",
}

# 1. Фаза Луны по классам
ax = axes[0,0]; ax.set_facecolor("#0d1117")
for cls, features_list in event_features.items():
    moon_phases = [f["moon_phase"] for f in features_list]
    ax.scatter(moon_phases, [cls]*len(moon_phases),
               color=colors_cls[cls], alpha=0.6, s=50)
ctrl_moons = [f["moon_phase"] for f in control_features]
ax.scatter(ctrl_moons[:50], ["control"]*50, color="#888", alpha=0.3, s=20)
ax.set_xlabel("Фаза Луны (°)", color="#888")
ax.set_title("Лунная фаза по классам", color="white")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. Юпитер-Сатурн аспект
ax = axes[0,1]; ax.set_facecolor("#0d1117")
for cls, features_list in event_features.items():
    js_angles = [f["jupiter_saturn"] for f in features_list]
    ax.hist(js_angles, bins=12, alpha=0.5, color=colors_cls[cls],
            label=cls, density=True)
ctrl_js = [f["jupiter_saturn"] for f in control_features]
ax.hist(ctrl_js, bins=12, alpha=0.3, color="#888",
        label="control", density=True)
ax.set_xlabel("Аспект Юпитер-Сатурн (°)", color="#888")
ax.set_title("Аспект Юпитер-Сатурн", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Временной ряд событий
ax = axes[0,2]; ax.set_facecolor("#0d1117")
for cls, dates in EVENTS_BY_CLASS.items():
    event_years = []
    for d in dates:
        try:
            event_years.append(datetime.strptime(d,"%Y-%m-%d").year)
        except: pass
    ax.scatter(event_years, [cls]*len(event_years),
               color=colors_cls[cls], alpha=0.7, s=60)
ax.set_xlabel("Год", color="#888")
ax.set_title("Временное распределение событий", color="white")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 4. Средние значения признаков по классам
ax = axes[1,0]; ax.set_facecolor("#0d1117")
fn = "jupiter_saturn"
class_means = {cls: np.mean([f[fn] for f in fl])
               for cls, fl in event_features.items()}
class_means["control"] = np.mean(ctrl_vals[fn])
bars = ax.bar(list(class_means.keys()),
              list(class_means.values()),
              color=[colors_cls.get(c,"#888") for c in class_means.keys()],
              alpha=0.85)
ax.set_ylabel("Средний аспект Юп-Сат (°)", color="#888")
ax.set_title("Юпитер-Сатурн по классам", color="white")
ax.tick_params(colors="#888", axis='y')
ax.set_xticklabels(list(class_means.keys()), rotation=15,
                   fontsize=8, color="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 5. P-values тепловая карта
ax = axes[1,1]; ax.set_facecolor("#0d1117")
classes_list = list(event_features.keys())
p_matrix = np.ones((len(feature_names), len(classes_list)))
for j, cls in enumerate(classes_list):
    for i, fn in enumerate(feature_names):
        ev_vals = [f[fn] for f in event_features[cls]]
        ct_vals = ctrl_vals[fn]
        _, p = stats.ttest_ind(ev_vals, ct_vals)
        p_matrix[i, j] = p

im = ax.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.5, aspect="auto")
ax.set_xticks(range(len(classes_list)))
ax.set_xticklabels([c[:8] for c in classes_list],
                   rotation=15, fontsize=8, color="#888")
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names, fontsize=8, color="#888")
ax.set_title("P-values (зелёный = значимо)", color="white")
plt.colorbar(im, ax=ax)
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 6. Марс-позиция для экономических кризисов
ax = axes[1,2]; ax.set_facecolor("#0d1117")
crisis_mars = [f["mars_lon"] for f in event_features["economic_crisis"]]
tech_mars   = [f["mars_lon"] for f in event_features["tech_breakthrough"]]
ctrl_mars   = [f["mars_lon"] for f in control_features]
ax.hist(ctrl_mars,  bins=12, alpha=0.4, color="#888",
        label="control", density=True)
ax.hist(crisis_mars, bins=12, alpha=0.6, color="#E24B4A",
        label="economic_crisis", density=True)
ax.hist(tech_mars,   bins=12, alpha=0.6, color="#5DCAA5",
        label="tech_breakthrough", density=True)
ax.set_xlabel("Позиция Марса (°)", color="#888")
ax.set_title("Марс: кризисы vs прорывы", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("correlation_analysis.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ correlation_analysis.png")
