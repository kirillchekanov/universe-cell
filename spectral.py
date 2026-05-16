import json, networkx as nx, numpy as np, matplotlib.pyplot as plt
from scipy import linalg

print("Загружаю графы...")
G_mito   = nx.read_graphml("cell_graph.graphml")
G_hela   = nx.read_graphml("hela_graph.graphml")
G_galaxy = nx.read_graphml("galaxy_graph.graphml")

def spectral_density(G, name):
    print(f"  {name}: считаю спектр...")
    # Нормализованный лапласиан — именно его использовал Vazza 2020
    L = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    return {
        "name": name,
        "eigenvalues": eigenvalues.tolist(),
        "spectral_gap":  round(float(eigenvalues[1]), 6),
        "spectral_mean": round(float(eigenvalues.mean()), 6),
        "spectral_std":  round(float(eigenvalues.std()), 6),
        "spectral_max":  round(float(eigenvalues.max()), 6),
    }

results = []
for G, name in [
    (G_mito,   "Митохондрия"),
    (G_hela,   "HeLa"),
    (G_galaxy, "Галактики SDSS"),
]:
    r = spectral_density(G, name)
    results.append(r)

print("\n" + "="*65)
print(f"{'Метрика':<22} {'Митохондрия':>14} {'HeLa':>14} {'Галактики':>14}")
print("-"*65)
for k in ["spectral_gap","spectral_mean","spectral_std","spectral_max"]:
    vals = [r[k] for r in results]
    # Отношение HeLa / Галактики
    ratio = vals[1]/vals[2] if vals[2] else 0
    flag = "✓" if 0.5 < ratio < 2 else "~"
    print(f"{flag} {k:<20} {vals[0]:>14.6f} {vals[1]:>14.6f} {vals[2]:>14.6f}")
print("="*65)

# График спектральных плотностей
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#07080F")
fig.suptitle("Спектральная плотность графов — метод Vazza & Feletti 2020",
             fontsize=13, color="white", y=1.02)

colors = ["#7F77DD", "#5DCAA5", "#EF9F27"]
for ax, r, color in zip(axes, results, colors):
    ax.set_facecolor("#0d1117")
    eig = np.array(r["eigenvalues"])
    # Гистограмма собственных значений = спектральная плотность
    ax.hist(eig, bins=min(30, len(eig)), color=color, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(r["spectral_mean"], color="white", linestyle="--",
               linewidth=1, label=f"mean={r['spectral_mean']:.3f}")
    ax.set_title(r["name"], color="white", fontsize=11)
    ax.set_xlabel("Собственное значение λ", color="#888888", fontsize=9)
    ax.set_ylabel("Частота", color="#888888", fontsize=9)
    ax.tick_params(colors="#888888")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

plt.tight_layout()
plt.savefig("spectral_analysis.png", dpi=150,
            bbox_inches="tight", facecolor="#07080F")

# Сохранить метрики
spectral_metrics = {r["name"]: {k:v for k,v in r.items() if k!="eigenvalues"}
                    for r in results}
with open("spectral_metrics.json","w") as f:
    json.dump(spectral_metrics, f, indent=2)

print("\n✓ spectral_analysis.png")
print("✓ spectral_metrics.json")
print("\nЭти числа напрямую сравнимы со статьёй Vazza 2020!")
