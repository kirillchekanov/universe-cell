import urllib.request, json, csv, io
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from pathlib import Path

print("Скачиваю данные солнечной активности...")

# 1. Числа Вольфа — солнечные пятна, 1700-сейчас
# SILSO (Royal Observatory of Belgium) — официальный источник
url_sunspot = "https://www.sidc.be/silso/DATA/SN_y_tot_V2.0.txt"
cache_ss = Path("sunspots_yearly.txt")
if not cache_ss.exists():
    print("  Скачиваю числа Вольфа (солнечные пятна 1700-сейчас)...", end="")
    try:
        urllib.request.urlretrieve(url_sunspot, cache_ss)
        print(f" ✓")
    except Exception as e:
        print(f" ✗ {e}")

# 2. Глобальная температура — NASA GISS
url_temp = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
cache_temp = Path("global_temp.csv")
if not cache_temp.exists():
    print("  Скачиваю глобальную температуру NASA GISS...", end="")
    try:
        urllib.request.urlretrieve(url_temp, cache_temp)
        print(f" ✓")
    except Exception as e:
        print(f" ✗ {e}")

# Читаем солнечные пятна
print("\nЧитаю данные...")
ss_years, ss_vals = [], []
with open(cache_ss) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                year = int(float(parts[0]))
                val  = float(parts[1])
                if val >= 0:
                    ss_years.append(year)
                    ss_vals.append(val)
            except: pass

ss_years = np.array(ss_years)
ss_vals  = np.array(ss_vals)
print(f"  Солнечные пятна: {len(ss_years)} лет ({ss_years[0]}–{ss_years[-1]})")

# Читаем температуру
temp_years, temp_vals = [], []
with open(cache_temp) as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            year = int(row[0])
            # Годовое среднее — последний столбец "J-D"
            val  = float(row[13])
            if 1880 <= year <= 2030:
                temp_years.append(year)
                temp_vals.append(val)
        except: pass

temp_years = np.array(temp_years)
temp_vals  = np.array(temp_vals)
print(f"  Температура:     {len(temp_years)} лет ({temp_years[0]}–{temp_years[-1]})")

# Совмещаем по общему периоду
common_years = np.intersect1d(ss_years, temp_years)
ss_common   = np.array([ss_vals[ss_years==y][0] for y in common_years])
temp_common = np.array([temp_vals[temp_years==y][0] for y in common_years])

print(f"  Общий период:    {common_years[0]}–{common_years[-1]} ({len(common_years)} лет)")

# Корреляция
r, p = stats.pearsonr(ss_common, temp_common)
print(f"\n  Корреляция солн. пятна ↔ температура: r={r:.3f}  p={p:.4f}")

# 11-летний цикл — спектральный анализ
fft_ss   = np.fft.rfft(ss_vals - ss_vals.mean())
freqs    = np.fft.rfftfreq(len(ss_vals), d=1.0)
periods  = 1.0 / (freqs + 1e-10)
dominant = periods[np.argmax(np.abs(fft_ss)[1:])+1]
print(f"  Доминирующий период: {dominant:.1f} лет")

# Задержанная корреляция (лаг до 15 лет)
print(f"\n  Корреляция с задержкой (солнце → температура):")
best_r, best_lag = 0, 0
for lag in range(0, 16):
    if lag < len(ss_common):
        r_lag, _ = stats.pearsonr(ss_common[:-lag or None],
                                   temp_common[lag:])
        if abs(r_lag) > abs(best_r):
            best_r, best_lag = r_lag, lag
        print(f"    лаг {lag:>2} лет: r={r_lag:+.3f}")

print(f"\n  Лучшая корреляция: r={best_r:.3f} при лаге {best_lag} лет")
print(f"  → Солнечная активность влияет на температуру с задержкой {best_lag} лет")

# Прогноз на основе солнечного цикла
# Текущий цикл 25 начался в 2019, максимум ~2025
print(f"\n  ПРОГНОЗ на основе солнечного цикла:")
print(f"  Цикл 25: максимум ~2025, спад 2026–2030")
print(f"  При лаге {best_lag} лет → эффект на температуру: 202{5+best_lag}–203{0+best_lag}")
recent_max = ss_vals[ss_years >= 2000].max()
recent_mean = ss_vals[ss_years >= 1950].mean()
if recent_max > recent_mean * 1.2:
    print(f"  Текущий цикл сильный ({recent_max:.0f} vs среднее {recent_mean:.0f})")
    print(f"  → Ожидаем повышенную геомагнитную активность в 2025–2026")
else:
    print(f"  Текущий цикл умеренный ({recent_max:.0f} vs среднее {recent_mean:.0f})")

# График
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Солнечная активность → Земные события\nМакро→Микро предсказательная цепочка",
             fontsize=14, color="white")

# 1. Солнечные пятна
ax = axes[0]; ax.set_facecolor("#0d1117")
ax.fill_between(ss_years, ss_vals, color="#EF9F27", alpha=0.4)
ax.plot(ss_years, ss_vals, color="#EF9F27", linewidth=0.8)
# Отмечаем циклы
for cycle_max in [1750,1761,1769,1778,1788,1805,1816,1830,1837,1848,
                   1860,1870,1883,1894,1906,1917,1928,1937,1947,1958,
                   1968,1979,1989,2000,2014,2025]:
    if ss_years[0] <= cycle_max <= ss_years[-1]:
        ax.axvline(cycle_max, color="#E24B4A", linewidth=0.5, alpha=0.4)
ax.axvline(2025, color="#E24B4A", linewidth=2, linestyle="--",
           label="пик цикла 25 (~2025)")
ax.set_title("Солнечные пятна (числа Вольфа) 1700–2024", color="white")
ax.set_ylabel("Число пятен", color="#888")
ax.tick_params(colors="#888")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. Глобальная температура
ax = axes[1]; ax.set_facecolor("#0d1117")
colors_temp = ["#5DCAA5" if v < 0 else "#E24B4A" for v in temp_vals]
ax.bar(temp_years, temp_vals, color=colors_temp, alpha=0.7, width=1)
ax.axhline(0, color="#888", linewidth=0.5)
ax.set_title("Глобальная температура — аномалия от базового уровня (NASA GISS)",
             color="white")
ax.set_ylabel("°C отклонение", color="#888")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Корреляция с лагом
ax = axes[2]; ax.set_facecolor("#0d1117")
lags_range = range(0, 16)
r_vals = []
for lag in lags_range:
    r_lag, _ = stats.pearsonr(ss_common[:-lag or None], temp_common[lag:])
    r_vals.append(r_lag)
bars = ax.bar(list(lags_range), r_vals,
              color=["#5DCAA5" if r > 0 else "#E24B4A" for r in r_vals],
              alpha=0.85)
ax.axhline(0, color="#888", linewidth=0.5)
ax.axvline(best_lag, color="#EF9F27", linewidth=2, linestyle="--",
           label=f"лучший лаг = {best_lag} лет (r={best_r:.3f})")
ax.set_title("Задержанная корреляция: солнечная активность → температура Земли",
             color="white")
ax.set_xlabel("Задержка (лет)", color="#888")
ax.set_ylabel("Коэффициент корреляции r", color="#888")
ax.tick_params(colors="#888")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("solar_earth.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print("\n✓ solar_earth.png — первый уровень предсказательной цепочки")
print("\nСледующий шаг: добавить урожайность, эпидемии, экономику")
