"""
Week 3 Task: Unsupervised Learning and Clustering Analysis
Dataset: UCI Wine Dataset (via scikit-learn), 178 samples, 13 chemical features,
originating from three different cultivars grown in the same region of Italy.

This script performs the full pipeline: EDA -> preprocessing -> optimal-k selection
-> K-Means clustering -> Hierarchical clustering -> evaluation -> cluster profiling.
All figures are saved to /home/claude/wk3/figs/ for embedding in the Word report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, silhouette_samples,
                              calinski_harabasz_score, davies_bouldin_score,
                              adjusted_rand_score, confusion_matrix)
from scipy.cluster.hierarchy import dendrogram, linkage

import os
FIG = "/home/claude/wk3/figs"
os.makedirs(FIG, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 140

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
data = load_wine(as_frame=True)
df = data.frame.copy()
feature_cols = data.feature_names
true_labels = df["target"]  # cultivar labels — withheld from clustering, used only for validation

print("Shape:", df.shape)
print(df.head())
print("\nClass distribution (cultivars):\n", true_labels.value_counts())
print("\nMissing values:\n", df.isnull().sum().sum())

df[feature_cols].describe().T.to_csv("/home/claude/wk3/describe.csv")

# ---------------------------------------------------------------------------
# 2. EDA — correlation heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
corr = df[feature_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True,
            cbar_kws={"shrink": .8})
plt.title("Figure 1. Feature Correlation Heatmap (Wine Dataset)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_correlation_heatmap.png")
plt.close()

# Distribution of a few key features
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, col in zip(axes, ["alcohol", "flavanoids", "color_intensity"]):
    sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
    ax.set_title(col)
plt.suptitle("Figure 2. Distributions of Selected Chemical Properties")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_distributions.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. PREPROCESSING — standardization (crucial: features are on very different scales,
#    e.g. proline ~ hundreds vs. hue ~ 0-1.5, which would dominate a Euclidean-distance
#    clustering algorithm if left unscaled)
# ---------------------------------------------------------------------------
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 4. DIMENSIONALITY REDUCTION FOR VISUALIZATION (PCA)
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("\nExplained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
print("Cumulative variance explained by 2 PCs: %.1f%%" % (pca.explained_variance_ratio_.sum() * 100))

pca_full = PCA(random_state=42).fit(X_scaled)
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), marker="o")
plt.axhline(0.8, color="grey", linestyle="--", linewidth=1)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Figure 3. PCA Cumulative Explained Variance")
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_pca_variance.png")
plt.close()

# ---------------------------------------------------------------------------
# 5. DETERMINE OPTIMAL K — Elbow method + Silhouette analysis
# ---------------------------------------------------------------------------
k_range = range(2, 11)
inertias, silhouettes, ch_scores, db_scores = [], [], [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels_k))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels_k))
    db_scores.append(davies_bouldin_score(X_scaled, labels_k))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[1].plot(list(k_range), silhouettes, marker="o", color="#DD8452")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Average Silhouette Score")
axes[1].set_title("Silhouette Analysis")
plt.suptitle("Figure 4. Selecting the Optimal Number of Clusters")
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_elbow_silhouette.png")
plt.close()

print("\nk | inertia | silhouette | calinski-harabasz | davies-bouldin")
for k, i, s, c, d in zip(k_range, inertias, silhouettes, ch_scores, db_scores):
    print(f"{k} | {i:.1f} | {s:.3f} | {c:.1f} | {d:.3f}")

best_k = list(k_range)[int(np.argmax(silhouettes))]
print("\nBest k by silhouette score:", best_k)

# ---------------------------------------------------------------------------
# 6. FINAL K-MEANS MODEL (k=3, chosen via silhouette + domain knowledge of 3 cultivars)
# ---------------------------------------------------------------------------
K_FINAL = 3
kmeans = KMeans(n_clusters=K_FINAL, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)
df["kmeans_cluster"] = km_labels

sil_final = silhouette_score(X_scaled, km_labels)
ch_final = calinski_harabasz_score(X_scaled, km_labels)
db_final = davies_bouldin_score(X_scaled, km_labels)
ari_final = adjusted_rand_score(true_labels, km_labels)
print(f"\nFinal K-Means (k={K_FINAL}): silhouette={sil_final:.3f}, "
      f"calinski-harabasz={ch_final:.1f}, davies-bouldin={db_final:.3f}, "
      f"ARI vs true cultivar={ari_final:.3f}")

# PCA scatter colored by cluster
plt.figure(figsize=(6.5, 5.5))
palette = sns.color_palette("Set2", K_FINAL)
for c in range(K_FINAL):
    mask = km_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=45, alpha=0.8,
                color=palette[c], label=f"Cluster {c}")
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=250,
            color="black", label="Centroids", edgecolor="white", linewidth=1.2)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Figure 5. K-Means Clusters in PCA Space (k=3)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_kmeans_pca.png")
plt.close()

# Silhouette plot (per-sample)
sample_sil = silhouette_samples(X_scaled, km_labels)
fig, ax = plt.subplots(figsize=(6.5, 5))
y_lower = 10
for c in range(K_FINAL):
    vals = sample_sil[km_labels == c]
    vals.sort()
    size = vals.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=palette[c],
                      edgecolor=palette[c], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size, str(c))
    y_lower = y_upper + 10
ax.axvline(sil_final, color="red", linestyle="--", label=f"Mean = {sil_final:.2f}")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title("Figure 6. Silhouette Plot per Sample (K-Means, k=3)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_silhouette_plot.png")
plt.close()

# ---------------------------------------------------------------------------
# 7. HIERARCHICAL CLUSTERING (Agglomerative, Ward linkage) — comparison model
# ---------------------------------------------------------------------------
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90., leaf_font_size=8,
           show_contracted=True)
plt.axhline(y=18, color="grey", linestyle="--", label="Cut for k=3")
plt.title("Figure 7. Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Sample index / (cluster size)")
plt.ylabel("Ward Distance")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_dendrogram.png")
plt.close()

agglo = AgglomerativeClustering(n_clusters=K_FINAL, linkage="ward")
agglo_labels = agglo.fit_predict(X_scaled)
df["hier_cluster"] = agglo_labels
sil_h = silhouette_score(X_scaled, agglo_labels)
ari_h = adjusted_rand_score(true_labels, agglo_labels)
ari_km_hier = adjusted_rand_score(km_labels, agglo_labels)
print(f"\nHierarchical (Ward, k=3): silhouette={sil_h:.3f}, ARI vs true cultivar={ari_h:.3f}, "
      f"ARI vs K-Means labels={ari_km_hier:.3f}")

plt.figure(figsize=(6.5, 5.5))
for c in range(K_FINAL):
    mask = agglo_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=45, alpha=0.8, color=palette[c],
                label=f"Cluster {c}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("Figure 8. Hierarchical (Agglomerative) Clusters in PCA Space (k=3)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig8_hier_pca.png")
plt.close()

# ---------------------------------------------------------------------------
# 8. CLUSTER PROFILING — mean feature values per K-Means cluster (z-scored + raw)
# ---------------------------------------------------------------------------
profile_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
profile_scaled["cluster"] = km_labels
cluster_means_z = profile_scaled.groupby("cluster")[feature_cols].mean()

plt.figure(figsize=(11, 4.5))
sns.heatmap(cluster_means_z, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            cbar_kws={"label": "Mean (standardized)"})
plt.title("Figure 9. Cluster Feature Profile Heatmap (Standardized Means)")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(f"{FIG}/fig9_cluster_profile_heatmap.png")
plt.close()

cluster_means_raw = df.groupby("kmeans_cluster")[feature_cols].mean()
cluster_means_raw["n_samples"] = df.groupby("kmeans_cluster").size()
cluster_means_raw.to_csv("/home/claude/wk3/cluster_profiles_raw.csv")

# Boxplots of the most differentiating features across clusters
top_feats = cluster_means_z.abs().max().sort_values(ascending=False).index[:4]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, feat in zip(axes, top_feats):
    sns.boxplot(x="kmeans_cluster", y=feat, data=df, palette=palette, ax=ax)
    ax.set_xlabel("Cluster")
plt.suptitle("Figure 10. Distribution of Top Differentiating Features by Cluster")
plt.tight_layout()
plt.savefig(f"{FIG}/fig10_boxplots_top_features.png")
plt.close()

# Cross-tab: K-Means cluster vs. true cultivar (validation only, not used in training)
ct = pd.crosstab(df["kmeans_cluster"], true_labels.map({0: "cultivar_0", 1: "cultivar_1", 2: "cultivar_2"}))
ct.to_csv("/home/claude/wk3/cluster_vs_cultivar.csv")
print("\nCross-tab of K-Means cluster vs. true cultivar (validation):\n", ct)

# Save summary numbers for the report
summary = {
    "n_samples": len(df),
    "n_features": len(feature_cols),
    "pca_var_2pc": float(pca.explained_variance_ratio_.sum()),
    "best_k_silhouette": int(best_k),
    "k_final": K_FINAL,
    "kmeans_silhouette": float(sil_final),
    "kmeans_calinski_harabasz": float(ch_final),
    "kmeans_davies_bouldin": float(db_final),
    "kmeans_ari_vs_cultivar": float(ari_final),
    "hier_silhouette": float(sil_h),
    "hier_ari_vs_cultivar": float(ari_h),
    "ari_kmeans_vs_hier": float(ari_km_hier),
}
import json
with open("/home/claude/wk3/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSummary:", json.dumps(summary, indent=2))

df.to_csv("/home/claude/wk3/wine_with_clusters.csv", index=False)
cluster_means_z.to_csv("/home/claude/wk3/cluster_means_zscore.csv")
print("\nDone. Figures saved to", FIG)
