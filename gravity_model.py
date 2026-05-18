import ephem, math, numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from pathlib import Path
import json

print("ГРАВИТАЦИОННАЯ МОДЕЛЬ — физически обоснованный подход")
print("="*60)

# Физические константы
G  = 6.674e-11  # гравитационная постоянная
AU = 1.496e11   # астрономическая единица в метрах

# Массы тел (кг)
MASSES = {
    "sun":     1.989e30,
    "mercury": 3.301e23,
    "venus":   4.867e24,
    "earth":   5.972e24,
    "moon":    7.342e22,
    "mars":    6.417e23,
    "jupiter": 1.898e27,
    "saturn":  5.683e26,
    "uranus":  8.681e25,
    "neptune": 1.024e26,
    "ceres":   9.383e20,
    "pluto":   1.303e22,
}

def get_heliocentric_xyz(date_str):
    """Гелиоцентрические координаты всех тел в AU"""
    obs = ephem.Observer()
    obs.lat = "0"; obs.lon = "0"
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    obs.date = dt.strftime("%Y/%m/%d 12:00:00")
    obs.epoch = ephem.J2000

    bodies_ephem = {
        "sun":     ephem.Sun(),
        "mercury": ephem.Mercury(),
        "venus":   ephem.Venus(),
        "moon":    ephem.Moon(),
        "mars":    ephem.Mars(),
        "jupiter": ephem.Jupiter(),
        "saturn":  ephem.Saturn(),
        "uranus":  ephem.Uranus(),
        "neptune": ephem.Neptune(),
    }

    coords = {}
    for name, body in bodies_ephem.items():
        body.compute(obs)
        # Гелиоцентрические координаты
        if name == "sun":
            coords[name] = np.array([0.0, 0.0, 0.0])
        else:
            # Расстояние от Земли в AU
            dist_earth = float(body.earth_distance)
            ecl = ephem.Ecliptic(body, epoch=ephem.J2000)
            lon = float(ecl.lon)
            lat = float(ecl.lat)
            # Приближённые гелиоцентрические координаты
            x = dist_earth * math.cos(lat) * math.cos(lon)
            y = dist_earth * math.cos(lat) * math.sin(lon)
            z = dist_earth * math.sin(lat)
            coords[name] = np.array([x, y, z])

    return coords

def compute_physics(date_str):
    """Вычисляем физические характеристики системы"""
    try:
        coords = get_heliocentric_xyz(date_str)

        # 1. Гравитационный потенциал на Земле от каждого тела
        earth_pos = coords.get("moon", np.array([1.0, 0.0, 0.0]))
        # Земля примерно в 1 AU от Солнца
        earth_pos = np.array([1.0, 0.0, 0.0])

        grav_potential = 0.0
        grav_force = np.zeros(3)

        for name, pos in coords.items():
            if name == "sun": continue
            mass = MASSES.get(name, 1e22)
            r_vec = pos - earth_pos
            r = np.linalg.norm(r_vec)
            if r < 0.001: continue  # избегаем деления на 0
            r_m = r * AU  # в метрах

            # Потенциал φ = -GM/r
            grav_potential += -G * mass / r_m

            # Сила F = GM/r² (направление к телу)
            grav_force += G * mass / r_m**2 * r_vec/r

        # 2. Угловой момент системы (упрощённый)
        L_total = 0.0
        for name, pos in coords.items():
            mass = MASSES.get(name, 1e22)
            r = np.linalg.norm(pos)
            # v ≈ sqrt(GM_sun/r) для круговых орбит
            if r > 0.001:
                v = math.sqrt(G * MASSES["sun"] / (r * AU))
                L_total += mass * r * AU * v

        # 3. Суммарная кинетическая энергия конфигурации
        ke_total = 0.0
        for name, pos in coords.items():
            mass = MASSES.get(name, 1e22)
            r = np.linalg.norm(pos)
            if r > 0.001:
                v = math.sqrt(G * MASSES["sun"] / (r * AU))
                ke_total += 0.5 * mass * v**2

        # 4. Планетарные квадрупольные моменты
        # (мера асимметрии распределения масс)
        quadrupole = 0.0
        for name, pos in coords.items():
            mass = MASSES.get(name, 1e22)
            r = np.linalg.norm(pos)
            if r > 0.001:
                cos_theta = pos[2]/r if r > 0 else 0
                quadrupole += mass * r**2 * AU**2 * (3*cos_theta**2 - 1)

        # 5. Приливная сила Луны + Солнца
        moon_pos = coords.get("moon", np.zeros(3))
        r_moon = np.linalg.norm(moon_pos - earth_pos)
        tidal_moon = 2 * G * MASSES["moon"] / (r_moon * AU)**3 if r_moon > 0 else 0
        tidal_sun  = 2 * G * MASSES["sun"]  / (1.0 * AU)**3

        return {
            "grav_potential":  grav_potential,
            "grav_force_mag":  np.linalg.norm(grav_force),
            "angular_momentum": L_total,
            "kinetic_energy":  ke_total,
            "quadrupole":      quadrupole,
            "tidal_total":     tidal_moon + tidal_sun,
            "grav_potential_log": math.log10(abs(grav_potential)) if grav_potential != 0 else 0,
        }
    except Exception as e:
        return None

# Тест на нескольких датах
print("\nТест физических вычислений:")
test_dates = ["1929-10-29","2008-09-15","1969-07-20",
              "2020-03-16","1995-04-01","2026-05-17"]
for d in test_dates:
    f = compute_physics(d)
    if f:
        print(f"  {d}  φ={f['grav_potential_log']:.4f}  "
              f"F={f['grav_force_mag']:.3e}  "
              f"тиды={f['tidal_total']:.3e}")

# Временной ряд гравитационного потенциала 1920-2026
print(f"\nСтрою временной ряд 1920-2026 (годовые данные)...")
years = range(1920, 2027)
ts_data = []
for year in years:
    date = f"{year}-07-01"  # середина года
    f = compute_physics(date)
    if f:
        ts_data.append({"year": year, **f})
    if year % 10 == 0:
        print(f"  {year}...", end="", flush=True)
print(f" ✓ {len(ts_data)} точек")

ts_years = np.array([d["year"] for d in ts_data])
ts_grav  = np.array([d["grav_potential_log"] for d in ts_data])
ts_tidal = np.array([d["tidal_total"] for d in ts_data])
ts_quad  = np.array([d["quadrupole"] for d in ts_data])

# Спектральный анализ — ищем периоды
print(f"\nСпектральный анализ гравитационного потенциала:")
fft = np.fft.rfft(ts_grav - ts_grav.mean())
freqs = np.fft.rfftfreq(len(ts_grav), d=1.0)
periods = 1.0/(freqs[1:]+1e-10)
power = np.abs(fft[1:])
top_idx = np.argsort(power)[::-1][:8]
print(f"  Доминирующие периоды:")
for idx in top_idx:
    if 3 < periods[idx] < 50:
        print(f"    {periods[idx]:>8.2f} лет  мощность={power[idx]:.4f}")

print(f"\n  Планетарные циклы для сравнения:")
print(f"    11.00 лет — солнечный")
print(f"    11.86 лет — Юпитер")
print(f"    19.86 лет — Юп+Сат")
print(f"    29.46 лет — Сатурн")

# Корреляция с историческими кризисами
crisis_years = [1929, 1937, 1973, 1980, 1987, 1990,
                1997, 1998, 2000, 2001, 2008, 2011,
                2015, 2018, 2020, 2022]

crisis_mask = np.isin(ts_years, crisis_years)
normal_mask = ~crisis_mask

for metric_name, ts_metric in [
    ("grav_potential_log", ts_grav),
    ("tidal_total",        ts_tidal),
    ("quadrupole",         ts_quad),
]:
    if crisis_mask.sum() > 5 and normal_mask.sum() > 5:
        crisis_vals = ts_metric[crisis_mask]
        normal_vals = ts_metric[normal_mask]
        t, p = stats.ttest_ind(crisis_vals, normal_vals)
        sig = "✓" if p < 0.05 else "~" if p < 0.1 else " "
        diff = crisis_vals.mean() - normal_vals.mean()
        print(f"\n  {sig} {metric_name:<22} кризис={crisis_vals.mean():.4f}  "
              f"норма={normal_vals.mean():.4f}  Δ={diff:+.4f}  p={p:.4f}")

# Прогноз
print(f"\nГравитационный потенциал 2026-2030:")
for year in range(2026, 2031):
    f = compute_physics(f"{year}-07-01")
    if f:
        # Сравниваем с историческими кризисами
        hist_crisis_grav = np.mean(ts_grav[crisis_mask])
        diff = f["grav_potential_log"] - hist_crisis_grav
        signal = "⚠ близко к кризисным" if abs(diff) < 0.001 else "норма"
        print(f"  {year}: φ={f['grav_potential_log']:.6f}  "
              f"Δ от кризисов={diff:+.6f}  {signal}")

# Сохраняем временной ряд
with open("gravity_timeseries.json","w") as f:
    json.dump([{**d, "is_crisis": int(d["year"] in crisis_years)}
               for d in ts_data], f, indent=2)

# График
fig, axes = plt.subplots(3,1,figsize=(15,12))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Гравитационный потенциал системы 1920-2026\n"
             "Физически обоснованный подход к предсказанию",
             fontsize=13, color="white")

# 1. Гравитационный потенциал
ax = axes[0]; ax.set_facecolor("#0d1117")
ax.plot(ts_years, ts_grav, color="#5DCAA5", linewidth=1.5, label="φ (log)")
for cy in crisis_years:
    if cy in ts_years:
        ax.axvline(cy, color="#E24B4A", linewidth=0.8, alpha=0.5)
ax.axvline(1995, color="#EF9F27", linewidth=2,
           linestyle="--", label="рождение Кирилла (1995)")
ax.set_title("Гравитационный потенциал на Земле от планет",
             color="white")
ax.set_ylabel("log₁₀|φ| (Дж/кг)", color="#888")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
ax.text(0.02, 0.05, "красные линии = экономические кризисы",
        transform=ax.transAxes, fontsize=8, color="#E24B4A")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. Приливные силы
ax = axes[1]; ax.set_facecolor("#0d1117")
ax.plot(ts_years, ts_tidal, color="#EF9F27", linewidth=1, alpha=0.8)
for cy in crisis_years:
    if cy in ts_years:
        ax.axvline(cy, color="#E24B4A", linewidth=0.8, alpha=0.4)
ax.set_title("Суммарная приливная сила (Луна + Солнце + планеты)",
             color="white")
ax.set_ylabel("Приливная сила (м/с²)", color="#888")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Спектр мощности
ax = axes[2]; ax.set_facecolor("#0d1117")
mask_periods = (3 < periods) & (periods < 60)
ax.plot(periods[mask_periods], power[mask_periods],
        color="#7F77DD", linewidth=1.5)
ax.fill_between(periods[mask_periods], power[mask_periods],
                color="#7F77DD", alpha=0.2)
# Отмечаем планетарные периоды
for period, label, color in [
    (11.0,  "☉ 11 лет", "#EF9F27"),
    (11.86, "♃ Юпитер", "#5DCAA5"),
    (19.86, "♃♄ синод.", "#E24B4A"),
    (29.46, "♄ Сатурн", "#7F77DD"),
]:
    ax.axvline(period, color=color, linewidth=1.5,
               linestyle="--", label=label, alpha=0.8)
ax.set_xlabel("Период (лет)", color="#888")
ax.set_ylabel("Спектральная мощность", color="#888")
ax.set_title("Спектр гравитационного потенциала — поиск планетарных циклов",
             color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
          ncol=4)
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("gravity_model.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print(f"\n✓ gravity_model.png")
print(f"✓ gravity_timeseries.json")
