import mrcfile, networkx as nx, numpy as np, json
from pathlib import Path
from scipy import ndimage

print("Читаю HeLa...")
with mrcfile.open("emd_11756.map", permissive=True) as mrc:
    data = mrc.data.copy()
    vox  = float(mrc.voxel_size.x)
print(f"  ✓ {data.shape}, воксель {vox:.2f} Å")

norm = (data - data.min()) / (data.max() - data.min())
zones = {
    "outer_membrane": (0.82, 1.00),
    "inner_membrane": (0.68, 0.82),
    "organelle_body": (0.55, 0.68),
    "ribosomes":      (0.45, 0.55),
    "cytoplasm":      (0.25, 0.45),
    "void":           (0.00, 0.25),
}

print("Строю граф...")
G = nx.Graph()
node_id = 0
for zone_name, (lo, hi) in zones.items():
    mask = (norm >= lo) & (norm < hi)
    labeled, n_comp = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, min(n_comp+1, 51)))
    top_labels = np.argsort(sizes)[::-1][:50] + 1
    for lbl in top_labels:
        comp = labeled == lbl
        coords = np.argwhere(comp)
        if len(coords) < 10:
            continue
        c = coords.mean(axis=0).astype(int)
        # Сохраняем координаты как отдельные числа, не tuple
        G.add_node(node_id, zone=zone_name, size=int(comp.sum()),
                   cx=int(c[0]), cy=int(c[1]), cz=int(c[2]))
        node_id += 1

print(f"  ✓ узлов: {G.number_of_nodes()}")

nodes = list(G.nodes(data=True))
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        c1 = np.array([nodes[i][1]["cx"], nodes[i][1]["cy"], nodes[i][1]["cz"]])
        c2 = np.array([nodes[j][1]["cx"], nodes[j][1]["cy"], nodes[j][1]["cz"]])
        dist = float(np.linalg.norm(c1-c2))
        if dist < 80:
            G.add_edge(nodes[i][0], nodes[j][0], weight=round(1/(dist+1),4))

print(f"  ✓ рёбер: {G.number_of_edges()}")

nx.write_graphml(G, "hela_graph.graphml")

degrees = [d for _, d in G.degree()]
hela_metrics = {
    "nodes":       G.number_of_nodes(),
    "edges":       G.number_of_edges(),
    "clustering":  round(nx.average_clustering(G), 4),
    "density":     round(nx.density(G), 4),
    "mean_degree": round(float(np.mean(degrees)), 2),
}
with open("hela_metrics.json","w") as f:
    json.dump(hela_metrics, f, indent=2)

with open("galaxy_metrics.json") as f: gm = json.load(f)
with open("cell_metrics.json")   as f: cm = json.load(f)

print("\n" + "="*65)
print(f"{'Метрика':<22} {'Митохондрия':>12} {'HeLa':>12} {'Галактики':>12}")
print("-"*65)
for k in ["clustering","density","mean_degree"]:
    a = cm.get(k,0)
    b = hela_metrics.get(k,0)
    g = gm.get(k,0)
    flag = "✓" if g and 0.1<b/g<10 else "~"
    print(f"  {k:<20} {a:>12.4f} {flag}{b:>11.4f} {g:>12.4f}")
print("="*65)
print(f"\n  Митохондрия: {cm['nodes']} узлов")
print(f"  HeLa:        {hela_metrics['nodes']} узлов")
print(f"  Галактики:   {gm['nodes']} узлов")
print("\n✓ Готово! hela_metrics.json и hela_graph.graphml сохранены")
