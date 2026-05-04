# DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are densely packed together and labels sparse points as noise. Unlike K-Means it requires no k and can find arbitrarily-shaped clusters.

---

## Algorithm

**Three point types:**

- **Core point:** has at least `min_samples` neighbors within radius `eps`
- **Border point:** within `eps` of a core point, but not itself a core point
- **Noise point:** not reachable from any core point, labeled $-1$

**Steps:**

1. For each unvisited point, check if it is a core point
2. If yes, expand a new cluster via BFS over all density-reachable points
3. If no, mark as noise (may be reassigned later as a border point)

---

## Dataset

**Food Nutrition** (`FOOD-DATA-GROUP1.csv`) - 551 food items, 37 nutritional features. Noise points represent foods with unusual nutritional profiles that do not fit any cluster.

---

## Notebook Covers

- Core/border/noise point diagram with eps-circles and annotations
- EDA: PCA scatter highlighting potential outliers
- DBSCAN fit with cluster and noise point visualization
- Effect of `eps` on clustering (4-panel comparison)
- DBSCAN vs K-Means comparison table
