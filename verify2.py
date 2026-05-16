import networkx as nx, numpy as np, matplotlib.pyplot as plt
from scipy import stats

G_mito   = nx.read_graphml("cell_graph.graphml")
G_hela   = nx.read_graphml("hela_graph.graphml")
G_neuron = nx.read_graphml("neuron_graph.graphml")
G_galaxy = nx.read_graphml("galaxy_graph.graphml")

graphs = {"Митохондрия":G_mito, "Нейрон":G_neuron, "HeLa":G_hela, "Галактики":G_galaxy}
results = {}

for name, G in graphs.items():
    degrees = [d for _,d in G.degree()]
    n = G.number_of_nodes()
    k = float(np.mean(degrees)) if degrees else 1.0

    # 1. Доля пустот
    isolates = len(list(nx.isolates(G)))
    void_frac = isolates / n if n else 0.0

    # 2. Power law R²
    if len(degrees) > 5 and max(degrees) > 1:
        bc = np.bincount(degrees)
        dv = np.where(bc>0)[0]
        df = bc[dv]
        mask = dv > 0
        try:
            _, _, r, _, _ = stats.linregress(np.log(dv[mask]), np.log(df[mask]))
            pl_r2 = float(r**2)
        except: pl_r2 = 0.0
    else: pl_r2 = 0.0

    # 3. Small world σ
    C = nx.average_clustering(G)
    Gc = G.subgraph(max(nx.connected_components(G), key=len))
    L = nx.average_shortest_path_length(Gc) if len(Gc) > 1 else 1.0
    C_r = k/n if n>1 else 1.0
    L_r = float(np.log(n)/np.log(k)) if k>1 else 1.0
    sigma = float((C/C_r)/(L/L_r)) if C_r and L_r else 0.0

    # 4. Фрактальная размерность
    comps = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    if len(comps) > 2:
        try:
            s,_,_,_,_ = stats.linregress(np.log(range(1,len(comps)+1)), np.log(comps))
            fd = float(abs(s))
        except: fd = 0.0
    else: fd = 0.0

    results[name] = {"void":round(void_frac,4), "pl_r2":round(pl_r2,3),
                     "sigma":round(sigma,3), "fd":round(fd,3), "n":n}

# Таблица
print("\n"+"="*65)
print(f"{'Датасет':<14} {'Пустоты%':>9} {'PowerLaw R²':>12} {'SmallWorld σ':>13} {'Fractal D':>10}")
print("-"*65)
for name, r in results.items():
    p = "✓" if r['pl_r2']>0.5 else "~"
    s = "✓" if r['sigma']>1 else "✗"
    print(f"  {name:<12} {r['void']*100:>8.1f}% {p}{r['pl_r2']:>10.3f} {s}{r['sigma']:>11.2f} {r['fd']:>10.3f}")
print("="*65)

# График
colors = ["#7F77DD","#5DCAA5","#5DCAA5","#EF9F27"]
names  = list(results.keys())
fig, axes = plt.subplots(2,2,figsize=(14,9))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Четыре гипотезы · четыре датасета", fontsize=14, color="white", y=1.01)

for ax, (key, title, threshold, thresh_label) in zip(axes.flat, [
    ("void",  "Доля пустот %",          0.7,  "70% цель"),
    ("pl_r2", "Power Law R²",           0.7,  "R²=0.7"),
    ("sigma", "Малый мир σ",            1.0,  "σ=1"),
    ("fd",    "Фрактальная размерность",None,  None),
]):
    ax.set_facecolor("#0d1117")
    vals = [results[n][key]*(100 if key=="void" else 1) for n in names]
    bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="none")
    if threshold:
        ax.axhline(threshold*(100 if key=="void" else 1),
                   color="#E24B4A", linestyle="--", linewidth=1.2, label=thresh_label)
        ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="#888888")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, color="#888888")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.02,
                f"{v:.2f}", ha="center", fontsize=9, color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("all_hypotheses.png", dpi=150, bbox_inches="tight", facecolor="#07080F")
print("\n✓ all_hypotheses.png сохранён")
