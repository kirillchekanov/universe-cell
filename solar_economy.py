import urllib.request, numpy as np, matplotlib.pyplot as plt
from scipy import stats, signal
from pathlib import Path
import csv

print("Скачиваю экономические данные...")

# S&P 500 годовые данные — через Yahoo Finance CSV
url_sp = "https://stooq.com/q/d/l/?s=%5Espx&i=y"
cache_sp = Path("sp500_yearly.csv")
if not cache_sp.exists():
    print("  S&P 500 (1928-сейчас)...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url_sp, cache_sp)
        print(" ✓")
    except Exception as e:
        print(f" ✗ {e}")

# Читаем солнечные пятна (уже есть)
ss_years, ss_vals = [], []
with open("sunspots_yearly.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                y, v = int(float(parts[0])), float(parts[1])
                if v >= 0:
                    ss_years.append(y); ss_vals.append(v)
            except: pass
ss_years = np.array(ss_years)
ss_vals  = np.array(ss_vals)

# Читаем S&P 500
sp_years, sp_returns = [], []
if cache_sp.exists():
    prices = {}
    with open(cache_sp) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                from datetime import datetime
                dt   = datetime.strptime(row[0], "%Y-%m-%d")
                year = dt.year
                close = float(row[4])
                if year not in prices:
                    prices[year] = []
                prices[year].append(close)
            except: pass

    sorted_years = sorted(prices.keys())
    for i in range(1, len(sorted_years)):
        y    = sorted_years[i]
        y_prev = sorted_years[i-1]
        ret  = (np.mean(prices[y]) - np.mean(prices[y_prev])) / np.mean(prices[y_prev]) * 100
        sp_years.append(y)
        sp_returns.append(ret)

sp_years   = np.array(sp_years)
sp_returns = np.array(sp_returns)
print(f"  S&P 500: {len(sp_years)} лет ({sp_years[0] if len(sp_years) else '?'}–{sp_years[-1] if len(sp_years) else '?'})")

if len(sp_years) < 10:
    # Используем известные исторические данные S&P 500
    print("  Использую исторические данные S&P 500...")
    # Годовая доходность S&P 500 1950-2023 (процент)
    sp_data = {
        1950:21.8, 1951:16.5, 1952:11.8, 1953:-6.6, 1954:45.0,
        1955:26.4, 1956:2.6,  1957:-14.3,1958:38.1, 1959:8.5,
        1960:-3.0, 1961:23.1, 1962:-11.8,1963:18.9, 1964:13.0,
        1965:9.1,  1966:-13.1,1967:20.1, 1968:7.7,  1969:-11.4,
        1970:0.1,  1971:10.8, 1972:15.6, 1973:-17.4,1974:-29.7,
        1975:31.6, 1976:19.1, 1977:-11.5,1978:1.1,  1979:12.3,
        1980:25.8, 1981:-9.7, 1982:14.8, 1983:17.3, 1984:1.4,
        1985:26.3, 1986:14.6, 1987:2.0,  1988:12.4, 1989:27.3,
        1990:-6.6, 1991:26.3, 1992:4.5,  1993:7.1,  1994:-1.5,
        1995:34.1, 1996:20.3, 1997:31.0, 1998:26.7, 1999:19.5,
        2000:-10.1,2001:-13.0,2002:-23.4,2003:26.4, 2004:9.0,
        2005:3.0,  2006:13.6, 2007:3.5,  2008:-38.5,2009:23.5,
        2010:12.8, 2011:0.0,  2012:13.4, 2013:29.6, 2014:11.4,
        2015:-0.7, 2016:9.5,  2017:19.4, 2018:-6.2, 2019:28.9,
        2020:16.3, 2021:26.9, 2022:-19.4,2023:24.2,
    }
    sp_years   = np.array(sorted(sp_data.keys()))
    sp_returns = np.array([sp_data[y] for y in sp_years])

# Совмещаем с солнечными данными
common = np.intersect1d(ss_years, sp_years)
ss_c = np.array([ss_vals[ss_years==y][0]   for y in common])
sp_c = np.array([sp_returns[sp_years==y][0] for y in common])

# Корреляция прямая
r_direct, p_direct = stats.pearsonr(ss_c, sp_c)
print(f"\n  Прямая корреляция пятна ↔ S&P: r={r_direct:.3f}  p={p_direct:.3f}")

# Корреляция с задержками
print(f"\n  Задержанная корреляция (солнце → рынок):")
best_r, best_lag = 0, 0
lag_results = []
for lag in range(0, 12):
    ss_lag = ss_c[:len(ss_c)-lag] if lag > 0 else ss_c
    sp_lag = sp_c[lag:]
    if len(ss_lag) > 10:
        r_l, p_l = stats.pearsonr(ss_lag, sp_lag)
        lag_results.append((lag, r_l, p_l))
        if abs(r_l) > abs(best_r):
            best_r, best_lag = r_l, lag
        sig = "✓" if p_l < 0.05 else " "
        print(f"    {sig} лаг {lag:>2} лет: r={r_l:+.3f}  p={p_l:.3f}")

print(f"\n  Лучшая корреляция: r={best_r:.3f} при лаге {best_lag} лет")

# Фазовая корреляция — растущий vs падающий цикл
print(f"\n  ФАЗОВАЯ КОРРЕЛЯЦИЯ (важнее чем число пятен):")
# Определяем фазу цикла
ss_diff = np.diff(ss_c)
ss_diff = np.append(ss_diff, 0)
rising  = ss_diff > 0  # растущая фаза

sp_rising  = sp_c[rising]
sp_falling = sp_c[~rising]
t, p_phase = stats.ttest_ind(sp_rising, sp_falling)
print(f"  Доходность S&P в растущей фазе:  {sp_rising.mean():+.1f}%  (n={rising.sum()})")
print(f"  Доходность S&P в падающей фазе:  {sp_falling.mean():+.1f}%  (n={(~rising).sum()})")
print(f"  Разница значима: {'✓ да' if p_phase < 0.05 else '~ нет'}  (p={p_phase:.3f})")

# Прогноз 2025-2030
print(f"\n  ПРОГНОЗ РЫНКА на основе солнечного цикла:")
cycle25 = {2024:150, 2025:170, 2026:130, 2027:90, 2028:60, 2029:40, 2030:25}
prev_ss = 150
print(f"  {'Год':>6} {'Фаза':>10} {'Историч. средняя':>18} {'Сигнал':>10}")
print(f"  {'-'*48}")
for year, ss_est in cycle25.items():
    phase   = "растущая" if ss_est >= prev_ss else "падающая"
    avg_ret = sp_rising.mean() if ss_est >= prev_ss else sp_falling.mean()
    signal  = "↑ бычий" if avg_ret > 5 else "↓ медвежий" if avg_ret < -2 else "→ нейтральный"
    print(f"  {year:>6} {phase:>10} {avg_ret:>+17.1f}% {signal:>10}")
    prev_ss = ss_est

# График
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Солнечная активность → Экономика (S&P 500)\nТретий уровень предсказательной цепочки",
             fontsize=13, color="white")

# 1. Солнечные пятна + S&P на одном графике
ax = axes[0,0]; ax.set_facecolor("#0d1117")
ax2 = ax.twinx()
ax.fill_between(common, ss_c, color="#EF9F27", alpha=0.3, label="Солн. пятна")
ax.plot(common, ss_c, color="#EF9F27", linewidth=1)
ax2.bar(common, sp_c,
        color=["#5DCAA5" if r > 0 else "#E24B4A" for r in sp_c],
        alpha=0.6, width=0.8)
ax.set_ylabel("Солнечные пятна", color="#EF9F27")
ax2.set_ylabel("S&P 500 доходность %", color="#5DCAA5")
ax.set_title("Солнечная активность и S&P 500", color="white")
ax.tick_params(colors="#888"); ax2.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. Scatter корреляция
ax = axes[0,1]; ax.set_facecolor("#0d1117")
colors_s = ["#5DCAA5" if r > 0 else "#E24B4A" for r in sp_c]
ax.scatter(ss_c, sp_c, c=colors_s, alpha=0.7, s=40)
# Тренд
s, i, *_ = stats.linregress(ss_c, sp_c)
xl = np.linspace(ss_c.min(), ss_c.max(), 100)
ax.plot(xl, s*xl+i, color="#EF9F27", linewidth=2,
        label=f"r={r_direct:.3f}")
ax.axhline(0, color="#888", linewidth=0.5)
ax.set_xlabel("Число солнечных пятен", color="#888")
ax.set_ylabel("S&P 500 доходность %", color="#888")
ax.set_title("Корреляция: пятна vs рынок", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Фазовая доходность
ax = axes[1,0]; ax.set_facecolor("#0d1117")
phases = ["Растущая\nфаза", "Падающая\nфаза"]
means  = [sp_rising.mean(), sp_falling.mean()]
stds   = [sp_rising.std(),  sp_falling.std()]
colors_p = ["#5DCAA5" if m > 0 else "#E24B4A" for m in means]
bars = ax.bar(phases, means, color=colors_p, alpha=0.85,
              yerr=stds, capsize=8)
ax.axhline(sp_c.mean(), color="#EF9F27", linewidth=1.5,
           linestyle="--", label=f"общее среднее {sp_c.mean():+.1f}%")
for bar, m in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2,
            m + (2 if m > 0 else -4),
            f"{m:+.1f}%", ha="center", fontsize=12,
            color="white", fontweight="bold")
ax.set_title(f"Доходность S&P по фазе солнечного цикла\np={p_phase:.3f}",
             color="white")
ax.set_ylabel("Средняя годовая доходность %", color="#888")
ax.tick_params(colors="#888")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 4. Прогноз 2025-2030
ax = axes[1,1]; ax.set_facecolor("#0d1117")
pred_years = list(cycle25.keys())
pred_ss    = list(cycle25.values())
prev = 150
pred_returns = []
pred_colors  = []
for ss_e in pred_ss:
    phase   = ss_e >= prev
    avg_ret = sp_rising.mean() if phase else sp_falling.mean()
    pred_returns.append(avg_ret)
    pred_colors.append("#5DCAA5" if avg_ret > 0 else "#E24B4A")
    prev = ss_e

bars = ax.bar(pred_years, pred_returns,
              color=pred_colors, alpha=0.85)
for bar, y, r in zip(bars, pred_years, pred_returns):
    ax.text(bar.get_x()+bar.get_width()/2,
            r + (0.5 if r > 0 else -1.5),
            f"{r:+.1f}%", ha="center", fontsize=9,
            color="white", fontweight="bold")
ax.axhline(0, color="#888", linewidth=0.5)
ax.set_title("Прогноз S&P 500 по фазе солнечного цикла",
             color="white")
ax.set_ylabel("Ожидаемая доходность %", color="#888")
ax.tick_params(colors="#888")
ax.text(0.05, 0.05,
        "⚠ Статистический прогноз.\nНе является инвестиционной рекомендацией.",
        transform=ax.transAxes, fontsize=8,
        color="#888", verticalalignment="bottom")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("solar_economy.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print("\n✓ solar_economy.png")
print("\nСледующий уровень: Солнце → здоровье → эпидемии")
