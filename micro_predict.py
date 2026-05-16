import gzip, networkx as nx, numpy as np
import matplotlib.pyplot as plt
from scipy import stats

organisms = [
    ("ecoli",  "E.coli",           3500),
    ("yeast",  "S.cerevisiae",     1000),
    ("fly",    "D.melanogaster",    600),
    ("mouse",  "M.musculus",         75),
    ("human",  "H.sapiens",           0),
]

print("Считаю метрики чувствительные к масштабу сети...")
results = []

for key, name, age_mya in organisms:
    fname = f"string_{key}.txt.gz"
    print(f"\n  {name} ({age_mya} млн лет)...")

    G = nx.Graph()
    with gzip.open(fname, "rt") as f:
        next(f)
        for i, line in enumerate(f):
            if i > 200000: break
            parts = line.strip().split()
            if len(parts) < 3: continue
            if int(parts[2]) >= 700:
                G.add_edge(parts[0], parts[1], weight=int(parts[2]))

    largest = max(nx.connected_components(G), key=len)
    Gc = G.subgraph(largest).copy()

    # Берём случайную подвыборку 200 узлов — чтобы графы были сопоставимы
    np.random.seed(42)
    sample = list(np.random.choice(list(Gc.nodes()), 
                  min(200, len(Gc)), replace=False))
    Gs = Gc.subgraph(sample)

    degrees = [d for _,d in Gs.degree()]
    L   = nx.normalized_laplacian_matrix(Gs).toarray().astype(float)
    eig = np.sort(np.linalg.eigvalsh(L))

    # Метрики чувствительные к структуре
    # Спектральный зазор — расстояние между λ1 и λ2
    spectral_gap  = float(eig[1]) if len(eig) > 1 else 0
    # Алгебраическая связность
    algebraic_con = float(eig[1]) if len(eig) > 1 else 0
    # Энтропия спектра
    eig_pos = eig[eig > 1e-10]
    spectral_entropy = float(-np.sum(eig_pos/eig_pos.sum() * 
                             np.log(eig_pos/eig_pos.sum()))) if len(eig_pos) > 0 else 0

    # Модулярность — насколько чётко выражены кластеры
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(Gs))
        modularity  = nx.community.modularity(Gs, communities)
    except:
        modularity = 0

    m = {
        "name":              name,
        "age_mya":           age_mya,
        "nodes":             Gs.number_of_nodes(),
        "spectral_gap":      round(spectral_gap, 5),
        "algebraic_con":     round(algebraic_con, 5),
        "spectral_entropy":  round(spectral_entropy, 4),
        "clustering":        round(nx.average_clustering(Gs), 4),
        "modularity":        round(modularity, 4),
        "mean_degree":       round(float(np.mean(degrees)), 2),
    }
    results.append(m)
    print(f"    gap={m['spectral_gap']:.5f}  entropy={m['spectral_entropy']:.4f}  "
          f"modularity={m['modularity']:.4f}  clustering={m['clustering']:.4f}")

# Таблица
print("\n" + "="*80)
print(f"  БИОЛОГИЧЕСКИЕ СЕТИ — метрики чувствительные к структуре")
print("="*80)
print(f"{'Организм':<18} {'Возраст':>10} {'gap':>8} {'entropy':>9} "
      f"{'modularity':>11} {'clustering':>11}")
print("-"*80)
for m in results:
    print(f"  {m['name']:<16} {m['age_mya']:>8} млн  {m['spectral_gap']:>8.5f} "
          f"{m['spectral_entropy']:>9.4f} {m['modularity']:>11.4f} {m['clustering']:>11.4f}")

# Тренды и прогнозы
print(f"\n  ТРЕНДЫ (от прошлого к настоящему):")
ages  = np.array([m['age_mya'] for m in results])
metrics_to_analyze = [
    ("spectral_entropy", "Энтропия спектра"),
    ("modularity",       "Модулярность"),
    ("clustering",       "Кластеризация"),
    ("mean_degree",      "Средняя степень"),
]

trends = {}
for key, label in metrics_to_analyze:
    vals = np.array([m[key] for m in results])
    slope, intercept, r, _, se = stats.linregress(ages, vals)
    trends[key] = {"slope": slope, "intercept": intercept, "r2": r**2, "se": se}
    direction = "растёт" if slope < 0 else "падает"
    print(f"  {label:<22} {direction} {abs(slope*1000):.4f}/млрд лет  R²={r**2:.3f}")

# Прогноз
print(f"\n  МИКРОПРОГНОЗ — биологические сети будущего:")
print(f"  {'Через':<16}", end="")
for key, label in metrics_to_analyze[:3]:
    print(f"  {label[:10]:>12}", end="")
print()
print(f"  {'-'*60}")

for future_mya, lbl in [(-500,"+500 млн"), (-1000,"+1 млрд"), (-2000,"+2 млрд")]:
    print(f"  {lbl:<16}", end="")
    for key, _ in metrics_to_analyze[:3]:
        t = trends[key]
        pred = t['slope'] * future_mya + t['intercept']
        print(f"  {pred:>12.4f}", end="")
    print()

# Ключевой вопрос — куда идёт модулярность?
mod_vals = [m['modularity'] for m in results]
print(f"\n  КЛЮЧЕВОЙ ВЫВОД — модулярность:")
if mod_vals[-1] > mod_vals[0]:
    print(f"  Модулярность РАСТЁТ со временем: {mod_vals[0]:.4f} → {mod_vals[-1]:.4f}")
    print(f"  → Биологические сети становятся БОЛЕЕ специализированными")
    print(f"  → Кластеры становятся чётче — больше функциональных модулей")
else:
    print(f"  Модулярность ПАДАЕТ со временем: {mod_vals[0]:.4f} → {mod_vals[-1]:.4f}")
    print(f"  → Биологические сети становятся БОЛЕЕ интегрированными")
    print(f"  → Граница между модулями размывается")

# Сравнение с космосом
print(f"\n  СРАВНЕНИЕ БИОЛОГИЯ vs КОСМОС:")
print(f"  Биология: clustering {'растёт' if trends['clustering']['slope'] < 0 else 'падает'}"
      f"  R²={trends['clustering']['r2']:.3f}")
print(f"  Космос:   clustering падает  (0.7212 → 0.6945 за 6.5 млрд лет)")
print(f"  → Оба тренда в одном направлении: {'✓ совпадают' if trends['clustering']['slope'] > 0 else '~ расходятся'}")

# График
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Биологические сети STRING — эволюция через 3.5 млрд лет",
             fontsize=13, color="white")

plot_configs = [
    ("spectral_entropy", "Энтропия спектра",  "#5DCAA5"),
    ("modularity",       "Модулярность",       "#EF9F27"),
    ("clustering",       "Кластеризация",       "#7F77DD"),
    ("mean_degree",      "Средняя степень",    "#E24B4A"),
]

for ax, (key, title, color) in zip(axes.flat, plot_configs):
    ax.set_facecolor("#0d1117")
    ages_plot = [m['age_mya'] for m in results]
    vals_plot  = [m[key] for m in results]
    names_plot = [m['name'] for m in results]

    ax.plot(ages_plot, vals_plot, "o-", color=color,
            linewidth=2, markersize=10,
            markeredgecolor="white", markeredgewidth=0.8)

    for age, v, name in zip(ages_plot, vals_plot, names_plot):
        ax.annotate(f"{name}\n{v:.4f}", (age, v),
                    textcoords="offset points", xytext=(0,14),
                    ha="center", fontsize=8, color=color)

    # Тренд линия
    t = trends[key]
    x_line = np.linspace(min(ages_plot), 500, 100)
    ax.plot(x_line, t['slope']*x_line + t['intercept'],
            color=color, linewidth=1, linestyle="--", alpha=0.4)

    # Прогноз
    for fut_mya, lbl in [(-500,"+500M"), (-1000,"+1B")]:
        pred = t['slope']*fut_mya + t['intercept']
        ax.scatter([fut_mya], [pred], color=color, s=80,
                   marker="D", edgecolors="white", linewidth=0.8, alpha=0.7)

    ax.axvline(0, color="#E24B4A", linewidth=1,
               linestyle=":", alpha=0.7, label="сейчас")
    ax.set_xlabel("Эволюционный возраст (млн лет назад)", color="#888")
    ax.set_ylabel(title, color="#888")
    ax.set_title(f"{title}  R²={t['r2']:.3f}", color="white", fontsize=11)
    ax.tick_params(colors="#888")
    ax.invert_xaxis()
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("micro_prediction.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")
print("\n✓ micro_prediction.png — микропрогноз биологических сетей")
