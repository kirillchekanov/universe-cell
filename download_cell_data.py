import urllib.request, gzip, shutil, mrcfile, networkx as nx, numpy as np, json
from pathlib import Path

DATA_DIR = Path(".")
URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-3805/map/emd_3805.map.gz"
GZ  = DATA_DIR / "emd_3805.map.gz"
MAP = DATA_DIR / "emd_3805.map"

print("Шаг 1: скачиваю данные митохондрии (~50 MB)...")
def progress(n, bs, total):
    mb = n*bs/1024/1024
    print(f"\r  {mb:.1f} / {total/1024/1024:.1f} MB", end="", flush=True)
urllib.request.urlretrieve(URL, GZ, reporthook=progress)
print("\n  ✓ скачано")

print("Шаг 2: распаковываю...")
with gzip.open(GZ,"rb") as f_in, open(MAP,"wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
GZ.unlink()
print("  ✓ распаковано")

print("Шаг 3: читаю 3D карту плотности...")
with mrcfile.open(MAP, permissive=True) as mrc:
    data = mrc.data.copy()
    vox  = float(mrc.voxel_size.x)
print(f"  ✓ размер: {data.shape}, воксель: {vox:.2f} Å")

print("Шаг 4: сегментирую органеллы...")
norm = (data - data.min()) / (data.max() - data.min())
zones = {"membrane":(0.55,0.70),"organelle":(0.70,0.85),"dense":(0.85,1.0),"cytoplasm":(0.30,0.55),"void":(0.0,0.30)}
organelles = {}
for name,(lo,hi) in zones.items():
    mask = (norm>=lo)&(norm<hi)
    cnt  = int(mask.sum())
    coords = np.argwhere(mask)
    centroid = tuple(coords.mean(axis=0).astype(int)) if len(coords) else (0,0,0)
    organelles[name] = {"count":cnt,"fraction":cnt/data.size,"centroid":centroid}
    print(f"  {name:<15} {cnt:>10} вокселей  ({cnt/data.size*100:.1f}%)")

print("Шаг 5: строю граф...")
G = nx.Graph()
for name,props in organelles.items():
    G.add_node(name, **{k:v for k,v in props.items() if k!="centroid"})
names = list(organelles.keys())
for i in range(len(names)):
    for j in range(i+1,len(names)):
        c1 = np.array(organelles[names[i]]["centroid"])
        c2 = np.array(organelles[names[j]]["centroid"])
        dist = float(np.linalg.norm(c1-c2))
        G.add_edge(names[i],names[j],weight=round(1/(dist+1),4))

nx.write_graphml(G,"cell_graph.graphml")
metrics = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "clustering": round(nx.average_clustering(G),4),
    "density":    round(nx.density(G),4),
    "mean_degree":round(np.mean([d for _,d in G.degree()]),2),
}
with open("cell_metrics.json","w") as f:
    json.dump(metrics,f,indent=2)

print("\n✓ ГОТОВО!")
print(f"  Узлов в графе:       {metrics['nodes']}")
print(f"  Рёбер:               {metrics['edges']}")
print(f"  Кластеризация:       {metrics['clustering']}")
print(f"  Плотность:           {metrics['density']}")
print("\n  Файлы сохранены:")
print("  → cell_graph.graphml  (граф для GNN)")
print("  → cell_metrics.json   (метрики для сравнения с космосом)")
