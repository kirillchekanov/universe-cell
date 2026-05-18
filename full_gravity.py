import ephem, math, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats, signal
from pathlib import Path
import json, urllib.request, urllib.parse

print("ПОЛНАЯ ГРАВИТАЦИОННАЯ МОДЕЛЬ")
print("Все доступные тела NASA JPL + правильный масштаб времени")
print("="*60)

G  = 6.674e-11
AU = 1.496e11

# Расширенный список масс (кг)
MASSES = {
    "sun":      1.989e30,
    "mercury":  3.301e23,
    "venus":    4.867e24,
    "moon":     7.342e22,
    "mars":     6.417e23,
    "jupiter":  1.898e27,
    "saturn":   5.683e26,
    "uranus":   8.681e25,
    "neptune":  1.024e26,
    "ceres":    9.383e20,
    "pluto":    1.303e22,
    "eris":     1.660e22,
    "haumea":   4.006e21,
    "makemake": 3.100e21,
    "sedna":    1.000e21,
    "quaoar":   1.400e21,
    "chiron":   2.700e18,
    "pallas":   2.110e20,
    "vesta":    2.590e20,
    "juno":     2.670e19,
}

def get_positions(date_str):
    obs = ephem.Observer()
    obs.lat = "0"; obs.lon = "0"
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs.date = dt.strftime("%Y/%m/%d 12:00:00")
    obs.epoch = ephem.J2000

    classic = {
        "mercury": ephem.Mercury(),
        "venus":   ephem.Venus(),
        "moon":    ephem.Moon(),
        "mars":    ephem.Mars(),
        "jupiter": ephem.Jupiter(),
        "saturn":  ephem.Saturn(),
        "uranus":  ephem.Uranus(),
        "neptune": ephem.Neptune(),
    }

    positions = {"sun": np.zeros(3)}

    for name, body in classic.items():
        body.compute(obs)
        ecl = ephem.Ecliptic(body, epoch=ephem.J2000)
        lon = float(ecl.lon)
        lat = float(ecl.lat)
        r   = float(body.sun_distance)  # расстояние от Солнца в AU
        x = r * math.cos(lat) * math.cos(lon)
        y = r * math.cos(lat) * math.sin(lon)
        z = r * math.sin(lat)
        positions[name] = np.array([x, y, z])

    # Добавляем малые тела через JPL Horizons API
    minor_ids = {
        "ceres":    "1",
        "pallas":   "2",
        "vesta":    "4",
        "eris":     "136199",
        "haumea":   "136108",
        "makemake": "136472",
    }

    for name, obj_id in minor_ids.items():
        try:
            # JPL Horizons векторный запрос
            params = {
                "format":     "json",
                "COMMAND":    f"'{obj_id}'",
                "OBJ_DATA":   "NO",
                "MAKE_EPHEM": "YES",
                "EPHEM_TYPE": "VECTORS",
                "CENTER":     "500@10",
                "START_TIME": f"'{dt.strftime('%Y-%b-%d')}'",
                "STOP_TIME":  f"'{dt.strftime('%Y-%b-%d')}'",
                "STEP_SIZE":  "1d",
                "VEC_TABLE":  "2",
                "CSV_FORMAT": "YES",
            }
            url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + \
                  urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())

            result = data.get("result","")
            in_soe = False
            for line in result.split("\n"):
                if "$$SOE" in line: in_soe = True; continue
                if "$$EOE" in line: break
                if in_soe and "," in line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        x = float(parts[2]) / AU  # перевод из км в AU
                        y = float(parts[3]) / AU
                        z = float(parts[4]) / AU if len(parts) > 4 else 0
                        positions[name] = np.array([x, y, z])
                        break
        except:
            pass  # если не получили — пропускаем

    return positions

def compute_full_physics(date_str):
    """Полный гравитационный анализ системы"""
    try:
        positions = get_positions(date_str)

        # Земля примерно в 1 AU вдоль оси X
        earth_pos = positions.get("moon",
                    np.array([1.0, 0.0, 0.0]))
        # Используем более точное положение
        if "moon" not in positions:
            earth_pos = np.array([1.0, 0.0, 0.0])

        metrics = {
            "grav_potential": 0.0,
            "grav_force_x":   0.0,
            "grav_force_y":   0.0,
            "grav_force_z":   0.0,
            "tidal_force":    0.0,
            "n_bodies":       0,
        }

        for name, pos in positions.items():
            mass = MASSES.get(name, 1e20)
            if name in ["sun"]:
                r_vec = pos - earth_pos
                r = np.linalg.norm(r_vec)
            else:
                r_vec = pos - earth_pos
                r = np.linalg.norm(r_vec)

            if r < 0.0001: continue
            r_m = r * AU

            metrics["grav_potential"] += -G * mass / r_m
            force_mag = G * mass / r_m**2
            metrics["grav_force_x"] += force_mag * r_vec[0]/r
            metrics["grav_force_y"] += force_mag * r_vec[1]/r
            metrics["grav_force_z"] += force_mag * r_vec[2]/r
            metrics["tidal_force"]  += 2 * G * mass / r_m**3
            metrics["n_bodies"] += 1

        metrics["grav_force_mag"] = math.sqrt(
            metrics["grav_force_x"]**2 +
            metrics["grav_force_y"]**2 +
            metrics["grav_force_z"]**2
        )
        metrics["grav_potential_log"] = math.log10(
            abs(metrics["grav_potential"])
        ) if metrics["grav_potential"] != 0 else 0

        # Угловой момент системы
        L = 0.0
        for name, pos in positions.items():
            mass = MASSES.get(name, 1e20)
            r = np.linalg.norm(pos)
            if r > 0.001:
                v = math.sqrt(G * MASSES["sun"] / (r * AU))
                L += mass * r * AU * v
        metrics["angular_momentum_log"] = math.log10(L) if L > 0 else 0

        return metrics
    except Exception as e:
        return None

# Строим временной ряд на ПРАВИЛЬНОМ масштабе
# Для 11.86-летнего цикла Юпитера нужны данные за 50+ лет
# с разрешением 1-3 месяца
print("\nСтрою временной ряд 1950-2026 (квартальные данные)...")
print("Разрешение: 3 месяца — правильный масштаб для планетарных циклов")
print()

quarterly_data = []
year = 1950
quarter_months = ["01-15", "04-15", "07-15", "10-15"]

while year <= 2026:
    for month in quarter_months:
        date_str = f"{year}-{month}"
        if date_str > "2026-05-17": break
        f = compute_full_physics(date_str)
        if f:
            quarterly_data.append({
                "date": date_str,
                "year_frac": year + quarter_months.index(month)*0.25,
                **f
            })
    print(f"  {year}...", end="", flush=True)
    if year % 10 == 9: print()
    year += 1

print(f"\n✓ {len(quarterly_data)} точек")

# Сохраняем
with open("quarterly_gravity.json","w") as f:
    json.dump(quarterly_data, f, indent=2, default=str)

years_q  = np.array([d["year_frac"] for d in quarterly_data])
grav_q   = np.array([d["grav_potential_log"] for d in quarterly_data])
tidal_q  = np.array([d["tidal_force"] for d in quarterly_data])
force_q  = np.array([d["grav_force_mag"] for d in quarterly_data])
angmom_q = np.array([d["angular_momentum_log"] for d in quarterly_data])

# Спектральный анализ на квартальных данных
print(f"\nСпектральный анализ (квартальное разрешение):")
for metric_name, metric_vals in [
    ("grav_potential", grav_q),
    ("grav_force",     force_q),
    ("angular_momentum", angmom_q),
]:
    # Welch PSD для более точного спектра
    freqs, psd = signal.welch(metric_vals - metric_vals.mean(),
                               fs=4.0,  # 4 измерения в год
                               nperseg=min(64, len(metric_vals)//4))
    periods = 1.0/(freqs[1:]+1e-10)
    top_idx = np.argsort(psd[1:])[::-1][:5]
    print(f"\n  {metric_name}:")
    for idx in top_idx:
        p = periods[idx]
        if 2 < p < 40:
            print(f"    {p:>8.2f} лет  PSD={psd[idx+1]:.4f}")

# Корреляция с кризисными годами
crisis_years_set = {1929,1937,1973,1980,1987,1990,1997,
                    1998,2000,2001,2008,2011,2015,2018,2020,2022}

def is_crisis(year_frac):
    return int(year_frac) in crisis_years_set or \
           int(year_frac+0.5) in crisis_years_set

crisis_mask = np.array([is_crisis(y) for y in years_q])
normal_mask = ~crisis_mask

print(f"\n{'='*60}")
print(f"КОРРЕЛЯЦИЯ С КРИЗИСАМИ (квартальные данные)")
print(f"Кризисные кварталы: {crisis_mask.sum()}")
print(f"Нормальные кварталы: {normal_mask.sum()}")
print(f"{'='*60}")

for name, vals in [("grav_potential_log", grav_q),
                   ("grav_force_mag",     force_q),
                   ("angular_momentum",   angmom_q),
                   ("tidal_force",        tidal_q)]:
    if crisis_mask.sum() < 5: continue
    cv = vals[crisis_mask]
    nv = vals[normal_mask]
    t, p = stats.ttest_ind(cv, nv)
    r, p_r = stats.pearsonr(
        crisis_mask.astype(float),
        vals
    )
    sig = "✓✓" if p<0.01 else "✓" if p<0.05 else "~" if p<0.1 else " "
    diff = cv.mean() - nv.mean()
    print(f"  {sig} {name:<24} "
          f"Δ={diff:+.4f}  t-test p={p:.4f}  r={r:+.3f}")

# Натальный момент рождения
natal = compute_full_physics("1995-04-01")
now   = compute_full_physics("2026-05-17")

print(f"\n{'='*60}")
print(f"ТВОЙ НАТАЛЬНЫЙ МОМЕНТ (1 апр 1995)")
print(f"{'='*60}")
if natal and now:
    for key in ["grav_potential_log","grav_force_mag",
                "tidal_force","angular_momentum_log"]:
        n_val = natal.get(key, 0)
        c_val = now.get(key, 0)
        diff  = c_val - n_val
        print(f"  {key:<26} рожд={n_val:.5f}  "
              f"сейчас={c_val:.5f}  Δ={diff:+.5f}")

# Ищем периоды максимального сходства с рождением
print(f"\nПоиск пиков сходства с натальным моментом (2026-2028):")
from datetime import timedelta

natal_vec = np.array([natal.get(k,0) for k in
    ["grav_potential_log","grav_force_mag","tidal_force"]])

peaks = []
for days in range(0, 730, 14):  # каждые 2 недели
    fd = (datetime(2026,5,17)+timedelta(days=days)).strftime("%Y-%m-%d")
    f  = compute_full_physics(fd)
    if f:
        fv = np.array([f.get(k,0) for k in
            ["grav_potential_log","grav_force_mag","tidal_force"]])
        # Нормализованное расстояние
        if natal_vec.std() > 0 and fv.std() > 0:
            dist = np.linalg.norm(
                (fv - natal_vec) / (abs(natal_vec)+1e-10)
            )
            peaks.append((dist, fd))

peaks.sort()
print(f"{'Дата':<14} {'Расстояние':>12}")
for dist, date in peaks[:5]:
    print(f"  {date}   {dist:.6f}")

# Финальный график
fig, axes = plt.subplots(2,2,figsize=(16,10))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Полная гравитационная модель — квартальные данные 1950-2026\n"
             f"Тел в модели: {quarterly_data[-1]['n_bodies'] if quarterly_data else '?'}",
             fontsize=13, color="white")

# 1. Гравитационный потенциал
ax = axes[0,0]; ax.set_facecolor("#0d1117")
ax.plot(years_q, grav_q, color="#5DCAA5", linewidth=1, alpha=0.9)
ax.fill_between(years_q, grav_q, grav_q.min(),
                where=crisis_mask, color="#E24B4A", alpha=0.3,
                label="кризисные периоды")
ax.axvline(1995.25, color="#EF9F27", linewidth=2,
           linestyle="--", label="рождение (апр 1995)")
ax.set_title("Гравитационный потенциал Φ", color="white")
ax.set_ylabel("log₁₀|Φ|", color="#888")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. Гравитационная сила
ax = axes[0,1]; ax.set_facecolor("#0d1117")
ax.plot(years_q, force_q, color="#EF9F27", linewidth=0.8, alpha=0.8)
ax.fill_between(years_q, force_q, 0,
                where=crisis_mask, color="#E24B4A", alpha=0.3)
ax.set_title("Результирующая гравитационная сила на Земле", color="white")
ax.set_ylabel("Сила (м/с²)", color="#888")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Спектр мощности (Welch)
ax = axes[1,0]; ax.set_facecolor("#0d1117")
freqs_w, psd_w = signal.welch(grav_q - grav_q.mean(),
                               fs=4.0, nperseg=32)
mask_p = freqs_w > 0
periods_w = 1.0/freqs_w[mask_p]
ax.semilogy(periods_w, psd_w[mask_p],
            color="#7F77DD", linewidth=1.5)
for p, lbl, col in [(11.86,"Юпитер","#5DCAA5"),
                     (29.46,"Сатурн","#EF9F27"),
                     (19.86,"Юп+Сат","#E24B4A"),
                     (11.00,"☉ цикл", "#BA7517")]:
    ax.axvline(p, color=col, linewidth=1.2,
               linestyle="--", label=lbl, alpha=0.8)
ax.set_xlabel("Период (лет)", color="#888")
ax.set_ylabel("Спектральная плотность (log)", color="#888")
ax.set_title("Спектр Φ — планетарные циклы",  color="white")
ax.set_xlim(2, 40)
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", ncol=2)
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 4. Угловой момент
ax = axes[1,1]; ax.set_facecolor("#0d1117")
ax.plot(years_q, angmom_q, color="#534AB7", linewidth=1, alpha=0.9)
ax.fill_between(years_q, angmom_q, angmom_q.min(),
                where=crisis_mask, color="#E24B4A", alpha=0.3,
                label="кризисы")
ax.axvline(1995.25, color="#EF9F27", linewidth=2,
           linestyle="--", label="рождение")
ax.set_title("Угловой момент системы", color="white")
ax.set_ylabel("log₁₀(L)", color="#888")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("full_gravity.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ full_gravity.png")
print(f"✓ quarterly_gravity.json")
