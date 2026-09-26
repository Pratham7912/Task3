# Unsupervised Learning & Clustering Analysis — Wine Dataset

Segmenting the UCI Wine dataset using **K-Means** and **Agglomerative Hierarchical Clustering**, with full preprocessing, PCA visualization, data-driven selection of *k*, multi-metric evaluation, and business-oriented cluster interpretation.

## 📊 Overview

This project applies unsupervised learning to a real-world chemistry dataset (178 wine samples, 13 physicochemical features, 3 underlying cultivars) to discover natural groupings without using any label information during training. The true cultivar labels are used only *after* clustering, purely as an external validation check.

**Key results:**
- Optimal clusters selected via elbow method + silhouette analysis → **k = 3**
- K-Means silhouette score: **0.285**
- Adjusted Rand Index vs. true cultivar labels: **0.897**
- Agreement between K-Means and Hierarchical clustering: **ARI = 0.853**

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `Week3_Clustering_Analysis.py` | Full analysis pipeline: EDA → preprocessing → PCA → optimal-k selection → K-Means & Hierarchical clustering → evaluation → cluster profiling |
| `Week3_Clustering_Analysis_Report.docx` | Complete written report with methodology, results, figures, and business/research implications |
| `figs/` *(generated on run)* | All output visualizations (correlation heatmap, elbow/silhouette plots, PCA scatterplots, dendrogram, cluster profile heatmaps) |

## 🛠️ Tech Stack

- Python 3
- scikit-learn (`KMeans`, `AgglomerativeClustering`, `PCA`, `StandardScaler`, metrics)
- pandas, numpy
- matplotlib, seaborn
- scipy (hierarchical linkage / dendrograms)

## 🚀 Getting Started

```bash
# clone the repo
git clone https://github.com/<your-username>/wine-clustering-analysis.git
cd wine-clustering-analysis

# install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn scipy

# run the analysis
python Week3_Clustering_Analysis.py
```

All figures are saved to `figs/`, and summary metrics are printed to the console and saved to `summary.json`.

## 🔍 Methodology

1. **EDA** — feature distributions, correlation analysis
2. **Preprocessing** — `StandardScaler` (critical, since features like `proline` and `hue` are on very different scales)
3. **Dimensionality reduction** — PCA for 2-D visualization only (clustering is done on full 13-D standardized space)
4. **Optimal k selection** — elbow method (inertia) + silhouette score across k = 2–10
5. **Clustering** — K-Means (primary) and Ward-linkage Agglomerative Clustering (cross-validation)
6. **Evaluation** — Silhouette, Calinski-Harabasz, Davies-Bouldin, and Adjusted Rand Index (vs. true cultivar, post-hoc only)
7. **Cluster profiling** — standardized/raw feature means, boxplots of top differentiating features

## 📈 Sample Output

- PCA scatter plots showing well-separated clusters
- Dendrogram confirming a natural 3-cluster cut
- Per-sample silhouette plot
- Cluster profile heatmap identifying each cluster's chemical "fingerprint"

## 💡 Business/Research Applications

- Quality-control screening for atypical batches
- Cultivar / origin verification from chemistry alone
- Data-driven blending strategy
- Reusable template for other product-segmentation problems

## 📄 License

This project uses the publicly available UCI Wine Recognition dataset (bundled with scikit-learn). Code is provided for educational purposes.
