import json, networkx as nx, numpy as np, matplotlib.pyplot as plt
from scipy import stats

print("Загружаю графы...")
G_mito   = nx.read_graphml("cell_graph.graphml")
G_hela   = nx.read_graphml("hela_graph.graphml")
G_neuron = nx.read_graphml("neuron_graph.graphml")
G_galaxy = nx.read_graphml("galaxy_graph.graphml")

graphs = {
    "Митохондрия": G_mito,
    "Нейрон":      G_neuron,
    "HeLa":        G_hela,
    "Галактики":   G_galaxy,
}

results = {}
for name, G in graphs.items():
    degrees = [d for _,d in G.degree()]
    n = G.number_of_nodes()
    
    # 1. ДОЛЯ ПУСТОТ (войды / вакуоли)
    # пустоты = узлы с нулевой степенью или изолированные кластеры
    isolates = len(list(nx.isolates(G)))
    void_fraction = isolates / n if n else 0
    
    # 2. POWER LAW — степенной закон распределения степеней
    if len(degrees) > 10 and max(degrees) > 2:
        deg_count = np.bincount(degrees)
        deg_vals  = np.where(deg_count > 0)[0]
        deg_freq  = deg_count[deg_vals]
        # логарифмическая регрессия — если R² > 0.7 то power law
        try:
            log_x = np.log(deg_vals[deg_vals > 0])
            log_y = np.log(deg_freq[deg_vals > 0])
            slope, intercept, r, p, _ = stats.linregress(log_x, log_y)
            power_law_r2 = round(r**2, 3)
            power_law_exp = round(abs(slope), 3)
        except:
            power_law_r2 = 0
            power_law_exp = 0
    else:
        power_law_r2, power_law_exp = 0, 0

    # 3. МАЛЫЙ МИР (small world)
    # σ = (C/C_rand) / (L/L_rand) > 1 → малый мир
    C = nx.average_clustering(G)
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        Gc = G.subgraph(max(nx.connected_components(G), key=len))
        L  = nx.average_shortest_path_length(Gc)
    # случайный граф: C_rand ≈ k/n, L_rand ≈ ln(n)/ln(k)
    k = np.mean(degrees) if degrees else 1
    C_rand = k / n if n > 1 else 1
    L_rand = np.log(n) / np.log(k) if k > 1 else 1
    sigma  = (C / C_rand) / (L / L_rand) if L_rand and C_rand else 0
    small_world = sigma > 1

    # 4. ФРАКТАЛЬНАЯ РАЗМЕРНОСТЬ — box-counting приближение
    # через связь между размером компонент и их числом
    components = [len(c) for c in nx.connected_components(G)]
    if len(components) > 2:
        sizes = np.array(sorted(components, reverse=True))
        ranks = np.arange(1, len(sizes)+1)
        try:
            slope2, _, r2, _, _ = stats.linregress(np.log(ranks), np.log(sizes))
            fractal_dim = round(abs(slope2), 3)
        except:
            fractal_dim = 0
    else:
        fractal_dim = 0

    results[name] = {
        "void_fraction":   round(void_fraction, 4),
        "power_law_r2":    power_law_r2,
        "power_law_exp":   power_law_exp,
        "small_world_sigma": round(sigma, 3),
        "is_small_world":  small_world,
        "fractal_dim":     fractal_dim,
        "clustering":      round(C, 4),
        "nodes":           n,
    }

# ВЫВОД ТАБЛИЦЫ
print("\n" + "="*75)
print(f"  ЧЕТЫРЕ ГИПОТЕЗЫ — ЧЕТЫРЕ ДАТАСЕТА")
print("="*75)

# 1. 70% пустот
print(f"\n{'1. ДОЛЯ ПУСТОТ (войды / вакуоли)'}")
print(f"{'Датасет':<16} {'Изолятов':>10} {'% пустот':>10} {'~70%?':>8}")
print("-"*46)
for name, r in results.items():
    flag = "✓" if 0.5 < r['void_fraction'] < 0.9 else "~"
    print(f"{flag} {name:<14} {int(r['void_fraction']*r['nodes']):>10} {r['void_fraction']*100:>9.1f}% {'да' if flag=='✓' else 'нет':>8}")

# 2. Power law
print(f"\n{'2. СТЕПЕННОЙ ЗАКОН (power law в распределении степеней)'}")
print(f"{'Датасет':<16} {'R²':>8} {'показатель':>12} {'подтверждён?':>14}")
print("-"*52)
for name, r in results.items():
    flag = "✓" if r['power_law_r2'] > 0.7 else "~"
    print(f"{flag} {name:<14} {r['power_law_r2']:>8.3f} {r['power_law_exp']:>12.3f} {'да' if flag=='✓' else 'нет':>14}")

# 3. Малый мир
print(f"\n{'3. МАЛЫЙ МИР (small world, σ > 1)'}")
print(f"{'Датасет':<16} {'σ':>8} {'малый мир?':>12}")
print("-"*38)
for name, r in results.items():
    flag = "✓" if r['is_small_world'] else "✗"
    print(f"{flag} {name:<14} {r['small_world_sigma']:>8.3f} {'да' if r['is_small_world'] else 'нет':>12}")

# 4. Фрактальная размерность
print(f"\n{'4. ФРАКТАЛЬНАЯ РАЗМЕРНОСТЬ'}")
print(f"{'Датасет':<16} {'D_f':>8} {'похожи?':>10}")
print("-"*36)
dims = [r['fractal_dim'] for r in results.values()]
mean_dim = np.mean([d for d in dims if d > 0])
for name, r in results.items():
    flag = "✓" if r['fractal_dim'] > 0 and abs(r['fractal_dim']-mean_dim) < 0.5 else "~"
    print(f"{flag} {name:<14} {r['fractal_dim']:>8.3f}")

# ИТОГ
print("\n" + "="*75)
confirmed = []
if all(0.3 < r['void_fraction'] < 0.95 for r in results.values()):
    confirmed.append("доля пустот")
if sum(1 for r in results.values() if r['power_law_r2'] > 0.5) >= 2:
    confirmed.append("power law")
if sum(1 for r in results.values() if r['is_small_world']) >= 2:
    confirmed.append("малый мир")

print(f"  Подтверждено гипотез: {len(confirmed)}/4")
for c in confirmed:
    print(f"  ✓ {c}")
print("="*75)

with open("all_hypotheses.json","w") as f:
    json.dump({k: {kk: int(vv) if isinstance(vv, bool) else vv for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, ensure_ascii=False)
print("\n✓ all_hypotheses.json сохранён")

# График
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Четыре гипотезы — четыре датасета", fontsize=14, color="white")

colors = ["#7F77DD","#5DCAA5","#5DCAA5","#EF9F27"]
names  = list(results.keys())

# 1. Доля пустот
ax = axes[0,0]; ax.set_facecolor("#0d1117")
vals = [r['void_fraction']*100 for r in results.values()]
bars = ax.bar(names, vals, color=colors, alpha=0.85)
ax.axhline(70, color="#E24B4A", linestyle="--", linewidth=1, label="70% цель")
ax.set_title("Доля пустот %", color="white"); ax.tick_params(colors="#888")
ax.set_ylabel("%", color="#888"); ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.0f}%",
            ha="center", fontsize=9, color="white")

# 2. Power law R²
ax = axes[0,1]; ax.set_facecolor("#0d1117")
vals = [r['power_law_r2'] for r in results.values()]
bars = ax.bar(names, vals, color=colors, alpha=0.85)
ax.axhline(0.7, color="#E24B4A", linestyle="--", linewidth=1, label="порог R²=0.7")
ax.set_title("Power law R²", color="white"); ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
            ha="center", fontsize=9, color="white")

# 3. Small world σ
ax = axes[1,0]; ax.set_facecolor("#0d1117")
vals = [min(r['small_world_sigma'], 20) for r in results.values()]
bars = ax.bar(names, vals, color=colors, alpha=0.85)
ax.axhline(1, color="#E24B4A", linestyle="--", linewidth=1, label="порог σ=1")
ax.set_title("Малый мир σ", color="white"); ax.tick_params(colors="#888")
ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for bar, v, name in zip(bars, vals, names):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{results[name]['small_world_sigma']:.1f}",
            ha="center", fontsize=9, color="white")

# 4. Фрактальная размерность
ax = axes[1,1]; ax.set_facecolor("#0d1117")
vals = [r['fractal_dim'] for r in results.values()]
bars = ax.bar(names, vals, color=colors, alpha=0.85)
ax.set_title("Фрактальная размерность D_f", color="white"); ax.tick_params(colors="#888")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
            ha="center", fontsize=9, color="white")

for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("all_hypotheses.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print("✓ all_hypotheses.png")
