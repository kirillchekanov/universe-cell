import json, numpy as np, matplotlib.pyplot as plt
from scipy import stats

# Наши три точки во времени
# Время в млрд лет от Большого взрыва (возраст вселенной = 13.8 млрд)
data = {
    "z=0.200 (2 млрд лет назад)": {"age": 11.8, "sm": 0.96600, "cl": 0.7212},
    "z=0.075 (700 млн лет назад)": {"age": 13.1, "sm": 0.96400, "cl": 0.7066},
    "z=0.025 (сейчас)":            {"age": 13.8, "sm": 0.98780, "cl": 0.7185},
}

ages   = np.array([d["age"] for d in data.values()])
sm_vals = np.array([d["sm"] for d in data.values()])
cl_vals = np.array([d["cl"] for d in data.values()])

# Линейная экстраполяция вперёд
future_ages = np.array([14.3, 15.0, 16.0, 18.0, 20.0])
future_labels = ["500 млн лет", "1.2 млрд", "2.2 млрд", "4.2 млрд", "6.2 млрд"]

def extrapolate(x, y, x_future):
    slope, intercept, r, p, se = stats.linregress(x, y)
    y_future = slope * x_future + intercept
    # Доверительный интервал
    n = len(x)
    t = 2.0  # ~95% для малой выборки
    ci = t * se * np.sqrt(1 + 1/n + (x_future - x.mean())**2 / ((x - x.mean())**2).sum())
    return y_future, ci, slope, r**2

sm_future, sm_ci, sm_slope, sm_r2 = extrapolate(ages, sm_vals, future_ages)
cl_future, cl_ci, cl_slope, cl_r2 = extrapolate(ages, cl_vals, future_ages)

# Таблица предсказаний
print("\n" + "="*70)
print("  ПРЕДСКАЗАНИЕ ТОПОЛОГИИ ВСЕЛЕННОЙ")
print("="*70)
print(f"\n  Тренд spectral_mean: {sm_slope:+.5f} за млрд лет  (R²={sm_r2:.3f})")
print(f"  Тренд clustering:   {cl_slope:+.5f} за млрд лет  (R²={cl_r2:.3f})")
print()
print(f"{'Через':<16} {'spectral_mean':>14} {'±':>6} {'clustering':>12} {'±':>6}")
print("-"*56)
for i, (lbl, sm, sm_e, cl, cl_e) in enumerate(zip(
    future_labels, sm_future, sm_ci, cl_future, cl_ci)):
    print(f"  {lbl:<14} {sm:>14.5f} {sm_e:>6.4f} {cl:>12.4f} {cl_e:>6.4f}")
print("="*70)

# Что это означает физически
print(f"\n  Интерпретация:")
if sm_slope > 0:
    print(f"  → spectral_mean растёт — вселенная становится БОЛЕЕ однородной")
else:
    print(f"  → spectral_mean падает — вселенная становится МЕНЕЕ однородной")
if cl_slope > 0:
    print(f"  → clustering растёт — галактики будут БОЛЕЕ кластеризованы")
else:
    print(f"  → clustering падает — структура будет БОЛЕЕ разреженной")

print(f"\n  Предостережение: 3 точки → R²={sm_r2:.2f}")
print(f"  Нужно минимум 10 точек для надёжного прогноза")
print(f"  Это направление, не точечное предсказание")

# Через F(s) — что это означает для клетки
print(f"\n  Через F(s) — прогноз для биологических сетей:")
print(f"  Если spectral_mean вселенной → {sm_future[0]:.4f}")
print(f"  То ожидаемый spectral_mean клеточных сетей будущего → ~{sm_future[0]:.4f}")
print(f"  (при условии что F(s) подтверждена)")

# График
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Прогноз топологии вселенной — экстраполяция F(s) в будущее",
             fontsize=13, color="white")

all_ages = np.concatenate([ages, future_ages])
x_line   = np.linspace(ages.min(), future_ages.max(), 100)

for ax, (obs, fut, ci, ylabel, color, cell_ref, slope, r2) in zip(axes, [
    (sm_vals, sm_future, sm_ci, "spectral_mean λ", "#5DCAA5", 0.9815, sm_slope, sm_r2),
    (cl_vals, cl_future, cl_ci, "clustering coeff", "#EF9F27", 0.9751, cl_slope, cl_r2),
]):
    ax.set_facecolor("#0d1117")

    # Тренд линия
    s, i, *_ = stats.linregress(ages, obs)
    ax.plot(x_line, s*x_line+i, color=color, linewidth=1,
            linestyle="--", alpha=0.4)

    # Наблюдаемые точки
    ax.scatter(ages, obs, color=color, s=150, zorder=5,
               edgecolors="white", linewidth=0.8, label="наблюдения SDSS")

    # Предсказанные точки
    ax.scatter(future_ages, fut, color=color, s=100, zorder=5,
               marker="D", edgecolors="white", linewidth=0.8,
               alpha=0.7, label="прогноз")

    # Доверительный интервал
    ax.fill_between(future_ages, fut-ci, fut+ci,
                    color=color, alpha=0.15, label="95% интервал")

    # Разделитель прошлое/будущее
    ax.axvline(13.8, color="#E24B4A", linewidth=1.5,
               linestyle=":", label="сейчас (13.8 млрд лет)")

    # Клетка как референс
    ax.axhline(cell_ref, color="#7F77DD", linewidth=1,
               linestyle="--", alpha=0.6, label=f"клетка = {cell_ref}")

    # Подписи точек
    for age, v, lbl in zip(ages, obs, ["2 млрд назад","700 млн назад","сейчас"]):
        ax.annotate(lbl, (age, v), textcoords="offset points",
                    xytext=(-5, 12), ha="center", fontsize=7, color=color)
    for age, v, lbl in zip(future_ages[:3], fut[:3], future_labels[:3]):
        ax.annotate(f"+{lbl}\n{v:.4f}", (age, v),
                    textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=7, color=color, alpha=0.8)

    ax.set_xlabel("Возраст вселенной (млрд лет)", color="#888")
    ax.set_ylabel(ylabel, color="#888")
    ax.set_title(f"{ylabel}  (тренд {slope:+.5f}/млрд лет, R²={r2:.2f})",
                 color="white", fontsize=10)
    ax.tick_params(colors="#888")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    # Зона будущего
    ax.axvspan(13.8, future_ages.max(), color="#1a1a2e", alpha=0.3)
    ax.text(16, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
            "будущее →", color="#444466", fontsize=9)

plt.tight_layout()
plt.savefig("prediction.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print("\n✓ prediction.png — прогноз топологии")
