import json, ephem, math, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from collections import defaultdict

print("Загружаю GDELT события...")
with open("gdelt_events.json") as f:
    raw_events = json.load(f)

print(f"  Загружено: {len(raw_events)} событий")

# Парсим даты и классы
events = []
for row in raw_events:
    try:
        date_str = str(row[0])
        if len(date_str) == 8:
            date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            cls  = row[1]
            events.append({"date": date, "class": cls})
    except: pass

print(f"  Распарсено: {len(events)} событий")

# Вычисляем планетарные признаки
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

        # Ключевые аспекты
        def asp(a, b): 
            d = abs(a-b) % 360
            return d if d <= 180 else 360-d

        return {
            "sun":            lons[0],
            "moon":           lons[1],
            "mars":           lons[3+1],  # mars
            "jupiter":        lons[5],
            "saturn":         lons[6],
            "js_aspect":      asp(lons[5], lons[6]),   # Юп-Сат
            "mars_jup":       asp(lons[4], lons[5]),   # Марс-Юп
            "sun_mars":       asp(lons[0], lons[4]),   # Солнце-Марс
            "moon_phase_sin": math.sin(math.radians(lons[1])),
            "jupiter_sign":   int(lons[5]/30),         # знак Юпитера
            "saturn_sign":    int(lons[6]/30),         # знак Сатурна
        }
    except:
        return None

print(f"Вычисляю планетарные признаки для 10 000 событий...")
print(f"(займёт 2-3 минуты)", flush=True)

enriched = []
for i, event in enumerate(events[:10000]):
    f = get_features(event["date"])
    if f:
        f["class"] = event["class"]
        f["date"]  = event["date"]
        enriched.append(f)
    if i % 1000 == 0:
        print(f"  {i}/10000...", end="\r", flush=True)

print(f"  ✓ {len(enriched)} событий с признаками")

# Группируем по классам
by_class = defaultdict(list)
for e in enriched:
    by_class[e["class"]].append(e)

print(f"\nРаспределение:")
for cls, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
    print(f"  {cls:<20} {len(items):>8,}")

# PERMUTATION TEST на реальных данных
print(f"\n{'='*65}")
print(f"PERMUTATION TEST — реальные данные GDELT")
print(f"{'='*65}")

feature_names = ["js_aspect","mars_jup","sun_mars","jupiter_sign",
                 "saturn_sign","moon_phase_sin"]

# Контрольная группа — все события вместе
all_features = {fn: [e[fn] for e in enriched] for fn in feature_names}

significant = []
for cls, items in by_class.items():
    if len(items) < 200: continue  # нужно достаточно данных
    print(f"\n  {cls.upper()} (n={len(items)}):")
    for fn in feature_names:
        cls_vals  = [e[fn] for e in items]
        ctrl_vals = all_features[fn]
        t, p = stats.ttest_ind(cls_vals, ctrl_vals)
        cls_mean  = np.mean(cls_vals)
        ctrl_mean = np.mean(ctrl_vals)
        diff = cls_mean - ctrl_mean
        sig = "✓✓" if p<0.001 else "✓" if p<0.01 else "~" if p<0.05 else " "
        if p < 0.05:
            print(f"    {sig} {fn:<20} "
                  f"cls={cls_mean:>7.2f}  ctrl={ctrl_mean:>7.2f}  "
                  f"Δ={diff:>+7.2f}  p={p:.4f}")
            significant.append({"class":cls,"feature":fn,
                                "p":round(p,5),"diff":round(diff,3)})

print(f"\n{'='*65}")
print(f"Значимых корреляций (p<0.05): {len(significant)}")
if significant:
    print(f"\nТОП находки:")
    for s in sorted(significant, key=lambda x: x["p"])[:10]:
        print(f"  p={s['p']:.5f}  {s['class']:<18} {s['feature']}")

# Предсказание для натального момента
print(f"\n{'='*65}")
print(f"ТВОЙ НАТАЛЬНЫЙ МОМЕНТ vs GDELT КЛАССЫ")
print(f"{'='*65}")

natal = get_features("1995-04-01")
now   = get_features("2026-05-16")

# Сходство с каждым классом (косинус на ключевых признаках)
fn_for_sim = ["js_aspect","mars_jup","sun_mars","moon_phase_sin"]

print(f"\nСходство СЕЙЧАС (май 2026) с классами GDELT:")
for cls, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
    if len(items) < 100: continue
    # Среднее евклидово расстояние
    dists = []
    for e in items[:500]:
        dist = sum((now[fn]-e[fn])**2 for fn in fn_for_sim)**0.5
        dists.append(dist)
    mean_dist = np.mean(dists)
    # Нормируем: меньше расстояние = больше сходство
    sim = 1/(1+mean_dist/100)
    print(f"  {cls:<20} сходство={sim:.4f}  (дист={mean_dist:.1f})")

# Пики по времени для натального момента
print(f"\nДаты максимального сходства с натальной картой (2026-2027):")
from datetime import timedelta
peak_dates = []
for days in range(0, 365, 7):
    fd = (datetime(2026,5,17)+timedelta(days=days)).strftime("%Y-%m-%d")
    fv = get_features(fd)
    if fv:
        dist = sum((fv[fn]-natal[fn])**2 for fn in fn_for_sim)**0.5
        peak_dates.append((dist, fd))

peak_dates.sort()
print(f"{'Дата':<14} {'Расстояние':>12} {'Интерпретация'}")
print(f"{'-'*50}")
for dist, date in peak_dates[:5]:
    sim = 1/(1+dist/100)
    interp = "очень близко к рождению" if sim>0.6 else "умеренно близко"
    print(f"  {date}  {dist:>10.1f}  {interp}")

# График
fig, axes = plt.subplots(2,2,figsize=(15,10))
fig.patch.set_facecolor("#07080F")
fig.suptitle(f"GDELT {len(enriched):,} событий — планетарные корреляции",
             fontsize=13, color="white")

colors = {"fight":"#E24B4A","disapprove":"#EF9F27",
          "coerce":"#7F77DD","protest":"#5DCAA5",
          "threaten":"#BA7517","assault":"#E07070",
          "demand":"#378ADD","reject":"#639922"}

# 1. Юп-Сат аспект по классам
ax = axes[0,0]; ax.set_facecolor("#0d1117")
cls_means_js = {}
for cls, items in by_class.items():
    if len(items) > 200:
        cls_means_js[cls] = np.mean([e["js_aspect"] for e in items])
ctrl_js_mean = np.mean(all_features["js_aspect"])
sorted_cls = sorted(cls_means_js.items(), key=lambda x: x[1])
names = [c for c,_ in sorted_cls]
vals  = [v for _,v in sorted_cls]
bar_colors = [colors.get(n,"#888") for n in names]
ax.barh(names, vals, color=bar_colors, alpha=0.85)
ax.axvline(ctrl_js_mean, color="white", linewidth=2,
           linestyle="--", label=f"среднее={ctrl_js_mean:.1f}°")
ax.set_xlabel("Аспект Юпитер-Сатурн (°)", color="#888")
ax.set_title("Юп-Сат аспект по классам событий", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. Марс-Юпитер аспект
ax = axes[0,1]; ax.set_facecolor("#0d1117")
for cls in ["fight","protest","coerce","disapprove"]:
    if cls in by_class and len(by_class[cls])>200:
        vals = [e["mars_jup"] for e in by_class[cls]]
        ax.hist(vals, bins=18, alpha=0.5, color=colors.get(cls,"#888"),
                label=f"{cls} (n={len(vals)})", density=True)
ax.hist(all_features["mars_jup"], bins=18, alpha=0.25,
        color="white", label="все события", density=True)
ax.set_xlabel("Аспект Марс-Юпитер (°)", color="#888")
ax.set_title("Марс-Юпитер по классам", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Знак Юпитера по классам
ax = axes[1,0]; ax.set_facecolor("#0d1117")
ZODIAC = ["♈","♉","♊","♋","♌","♍","♎","♏","♐","♑","♒","♓"]
for ci, (cls, items) in enumerate(list(by_class.items())[:4]):
    if len(items) < 200: continue
    sign_counts = np.zeros(12)
    for e in items:
        sign_counts[int(e["jupiter_sign"])] += 1
    sign_counts /= sign_counts.sum()
    ax.plot(range(12), sign_counts, "o-", color=list(colors.values())[ci],
            linewidth=1.5, markersize=5, label=cls, alpha=0.8)
ax.set_xticks(range(12))
ax.set_xticklabels(ZODIAC, color="#888", fontsize=11)
ax.set_ylabel("Доля событий", color="#888")
ax.set_title("Знак Юпитера по классам событий", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 4. P-values реальные
ax = axes[1,1]; ax.set_facecolor("#0d1117")
if significant:
    sig_cls  = [f"{s['class'][:8]}\n{s['feature'][:10]}" for s in significant[:12]]
    sig_pvals = [-math.log10(s['p']) for s in significant[:12]]
    sig_colors = [colors.get(s['class'],"#888") for s in significant[:12]]
    bars = ax.barh(sig_cls, sig_pvals, color=sig_colors, alpha=0.85)
    ax.axvline(1.3, color="#EF9F27", linewidth=1.5,
               linestyle="--", label="p=0.05")
    ax.axvline(2.0, color="#E24B4A", linewidth=1.5,
               linestyle="--", label="p=0.01")
    ax.set_xlabel("-log₁₀(p-value)", color="#888")
    ax.set_title("Значимость корреляций (GDELT)", color="white")
    ax.tick_params(colors="#888")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
else:
    ax.text(0.5,0.5,"Значимых корреляций\nне найдено\n(нужно больше данных)",
            ha="center",va="center",color="#888",fontsize=12,
            transform=ax.transAxes)
    ax.set_title("P-values", color="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("gdelt_correlation.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ gdelt_correlation.png")
