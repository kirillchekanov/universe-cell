import gzip, networkx as nx, numpy as np
import matplotlib.pyplot as plt
from scipy import stats

organisms = [
    ("ecoli",  "E.coli",          3500, "511145"),
    ("yeast",  "S.cerevisiae",    1000, "4932"),
    ("fly",    "D.melanogaster",   600, "7227"),
    ("mouse",  "M.musculus",        75, "10090"),
    ("human",  "H.sapiens",          0, "9606"),
]

# Возраст вселенной = 13.8 млрд — переводим эволюционный возраст
# в абсолютное время (когда появился организм)
def to_cosmic_age(mya):
    return 13800 - mya  # млн лет от Большого взрыва

results = []

for key, name, age_mya, taxid in organisms:
    fname = f"string_{key}.txt.gz"
    print(f"\nАнализирую {name} ({age_mya} млн лет эволюции)...")

    # Читаем только высококачественные взаимодействия (score >= 700)
    G = nx.Graph()
    n_edges = 0
    with gzip.open(fname, "rt") as f:
        next(f)  # пропускаем заголовок
        for i, line in enumerate(f):
            if i > 500000: break  # берём первые 500k строк
            parts = line.strip().split()
            if len(parts) < 3: continue
            score = int(parts[2])
            if score >= 700:  # высокая достоверность
                G.add_edge(parts[0], parts[1], weight=score)
                n_edges += 1

    if G.number_of_nodes() < 10:
        print(f"  Слишком мало узлов, пропускаю")
        continue

    print(f"  Узлов: {G.number_of_nodes():,}  Рёбер: {G.number_of_edges():,}")

    # Берём наибольшую компоненту
    largest = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(largest).copy()
    print(f"  Главная компонента: {Gc.number_of_nodes():,} узлов")

    # Считаем метрики (на подвыборке для скорости)
    sample_size = min(500, Gc.number_of_nodes())
    nodes_sample = list(Gc.nodes())[:sample_size]
    Gs = Gc.subgraph(nodes_sample)

    # Спектральный анализ
    L   = nx.normalized_laplacian_matrix(Gs).toarray().astype(float)
    eig = np.linalg.eigvalsh(L)

    degrees = [d for _, d in Gs.degree()]
    cosmic_age = to_cosmic_age(age_mya)

    m = {
        "name":          name,
        "age_mya":       age_mya,
        "cosmic_age_mya": cosmic_age,
        "nodes":         Gc.number_of_nodes(),
        "edges":         Gc.number_of_edges(),
        "spectral_mean": round(float(eig.mean()), 5),
        "spectral_std":  round(float(eig.std()),  5),
        "clustering":    round(nx.average_clustering(Gs), 4),
        "mean_degree":   round(float(np.mean(degrees)), 2),
    }
    results.append(m)
    print(f"  spectral_mean={m['spectral_mean']}  clustering={m['clustering']}")

# Таблица
print("\n" + "="*75)
print(f"  ЭВОЛЮЦИЯ БИОЛОГИЧЕСКИХ СЕТЕЙ — временной ряд")
print("="*75)
print(f"{'Организм':<20} {'Возраст':>12} {'spectral_mean':>14} {'clustering':>12} {'узлов':>8}")
print("-"*75)
for m in results:
    age_str = f"{m['age_mya']} млн лет"
    print(f"  {m['name']:<18} {age_str:>12} {m['spectral_mean']:>14.5f} {m['clustering']:>12.4f} {m['nodes']:>8,}")

# Тренд
if len(results) >= 3:
    ages_bio = np.array([m['age_mya'] for m in results])
    sm_bio   = np.array([m['spectral_mean'] for m in results])
    cl_bio   = np.array([m['clustering'] for m in results])

    sm_slope, sm_int, sm_r, _, _ = stats.linregress(ages_bio, sm_bio)
    cl_slope, cl_int, cl_r, _, _ = stats.linregress(ages_bio, cl_bio)

    print(f"\n  Тренд spectral_mean: {sm_slope:+.6f} за млн лет  R²={sm_r**2:.3f}")
    print(f"  Тренд clustering:   {cl_slope:+.6f} за млн лет  R²={cl_r**2:.3f}")

    # Прогноз на будущее (через 500 млн, 1 млрд лет)
    print(f"\n  МИКРОПРОГНОЗ через F(s):")
    print(f"  {'Через':<16} {'spectral_mean':>14} {'clustering':>12}")
    print(f"  {'-'*44}")
    for future_mya, lbl in [(-500,"500 млн лет"), (-1000,"1 млрд лет"), (-2000,"2 млрд лет")]:
        sm_f = sm_slope * future_mya + sm_int
        cl_f = cl_slope * future_mya + cl_int
        print(f"  {lbl:<16} {sm_f:>14.5f} {cl_f:>12.4f}")

    # Сравнение трендов: космос vs биология
    print(f"\n  СРАВНЕНИЕ ТРЕНДОВ:")
    print(f"  Космос (SDSS):    spectral_mean растёт +{abs(0.003):.4f}/млрд лет")
    print(f"  Биология (STRING): spectral_mean {'растёт' if sm_slope < 0 else 'падает'} {abs(sm_slope*1000):.4f}/млрд лет")
    ratio = abs(sm_slope*1000) / 0.003
    print(f"  Соотношение скоростей: {ratio:.2f}×")
    if 0.1 < ratio < 10:
        print(f"  ✓ Тренды СОПОСТАВИМЫ — F(s) работает во времени")
    else:
        print(f"  ~ Тренды разные по скорости — нужна нормировка")

# График
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Эволюция биологических сетей STRING — временной ряд\nE.coli → Дрожжи → Муха → Мышь → Человек",
             fontsize=13, color="white")

colors_bio = ["#E24B4A","#EF9F27","#5DCAA5","#7F77DD","#B4B2A9"]

# 1. spectral_mean по времени
ax = axes[0,0]; ax.set_facecolor("#0d1117")
ages_plot = [m['age_mya'] for m in results]
sm_plot   = [m['spectral_mean'] for m in results]
ax.plot(ages_plot, sm_plot, "o-", color="#5DCAA5", linewidth=2, markersize=10)
for age, sm, m in zip(ages_plot, sm_plot, results):
    ax.annotate(f"{m['name']}\n{sm:.4f}", (age, sm),
                textcoords="offset points", xytext=(0,12),
                ha="center", fontsize=8, color="#5DCAA5")
ax.axhline(0.9815, color="#7F77DD", linestyle="--",
           linewidth=1, label="HeLa клетка")
ax.set_xlabel("Эволюционный возраст (млн лет назад)", color="#888")
ax.set_ylabel("spectral_mean", color="#888")
ax.set_title("spectral_mean через эволюцию", color="white")
ax.tick_params(colors="#888"); ax.invert_xaxis()
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 2. clustering по времени
ax = axes[0,1]; ax.set_facecolor("#0d1117")
cl_plot = [m['clustering'] for m in results]
ax.plot(ages_plot, cl_plot, "o-", color="#EF9F27", linewidth=2, markersize=10)
for age, cl, m in zip(ages_plot, cl_plot, results):
    ax.annotate(f"{m['name']}\n{cl:.4f}", (age, cl),
                textcoords="offset points", xytext=(0,12),
                ha="center", fontsize=8, color="#EF9F27")
ax.set_xlabel("Эволюционный возраст (млн лет назад)", color="#888")
ax.set_ylabel("clustering", color="#888")
ax.set_title("Кластеризация через эволюцию", color="white")
ax.tick_params(colors="#888"); ax.invert_xaxis()
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 3. Размер сетей
ax = axes[1,0]; ax.set_facecolor("#0d1117")
nodes_plot = [m['nodes'] for m in results]
bars = ax.bar([m['name'] for m in results], nodes_plot,
              color=colors_bio[:len(results)], alpha=0.85)
for bar, v in zip(bars, nodes_plot):
    ax.text(bar.get_x()+bar.get_width()/2, v+50,
            f"{v:,}", ha="center", fontsize=9, color="white")
ax.set_ylabel("Узлов в сети (белков)", color="#888")
ax.set_title("Размер белковых сетей", color="white")
ax.tick_params(colors="#888")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

# 4. Космос vs Биология — оба тренда
ax = axes[1,1]; ax.set_facecolor("#0d1117")
# Нормируем оба к [0,1] для сравнения
if len(results) >= 2:
    sm_norm_bio  = (np.array(sm_plot) - min(sm_plot)) / (max(sm_plot) - min(sm_plot) + 1e-9)
    ages_norm_bio = np.array(ages_plot) / max(ages_plot)
    ax.plot(ages_norm_bio, sm_norm_bio, "o-", color="#5DCAA5",
            linewidth=2, markersize=8, label="Биология (STRING)")

# Космические данные (нормированные)
cosmos_ages = np.array([13800-6500, 13800-5500, 13800-4500,
                         13800-3500, 13800-2000, 13800-700, 13800])
cosmos_sm   = np.array([0.964, 0.966, 0.968, 0.970, 0.966, 0.964, 0.988])
if len(cosmos_sm) > 1:
    sm_norm_cos = (cosmos_sm - cosmos_sm.min()) / (cosmos_sm.max() - cosmos_sm.min() + 1e-9)
    ages_norm_cos = (cosmos_ages - cosmos_ages.min()) / (cosmos_ages.max() - cosmos_ages.min() + 1e-9)
    ax.plot(ages_norm_cos, sm_norm_cos, "s--", color="#EF9F27",
            linewidth=2, markersize=8, label="Космос (SDSS)")

ax.set_xlabel("Нормированное время (0=прошлое, 1=сейчас)", color="#888")
ax.set_ylabel("spectral_mean (нормированный)", color="#888")
ax.set_title("Биология vs Космос — сравнение трендов", color="white")
ax.tick_params(colors="#888")
ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("string_analysis.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print("\n✓ string_analysis.png")
