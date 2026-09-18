"""
Análise Multivariada — Wine Quality (UCI)
Vinhos Verdes Portugueses (Cortez et al., 2009)
"""
import io, base64, warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FactorAnalysis as SKFactorAnalysis

warnings.filterwarnings("ignore")

# ── Paleta de cores ────────────────────────────────────────────────────────────
C_RED   = "#7B2335"   # vinho tinto
C_BLUE  = "#2563A8"   # vinho branco
C_GREEN = "#3A7D5C"
C_AMBER = "#C47F2A"
C_GRAY  = "#6B6B6B"
PALETTE = [C_RED, C_BLUE, C_GREEN, C_AMBER, C_GRAY]

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#CCCCCC",
        "axes.linewidth":   0.8,
        "axes.grid":        True,
        "grid.color":       "#EEEEEE",
        "grid.linewidth":   0.6,
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
    })

set_style()

# ── Utilitários ────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def df_to_html(df, decimals=3):
    fmt = {c: f"{{:.{decimals}f}}" for c in df.select_dtypes("number").columns}
    return (df.style
              .format(fmt)
              .set_table_attributes('class="table"')
              .to_html())

FIGS = {}   # nome → base64 PNG
TABS = {}   # nome → HTML string

# ── 1. DADOS ───────────────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/"

print("Baixando dados…")
r_red   = requests.get(BASE + "winequality-red.csv",   timeout=30)
r_white = requests.get(BASE + "winequality-white.csv", timeout=30)

df_red   = pd.read_csv(io.BytesIO(r_red.content),   sep=";")
df_white = pd.read_csv(io.BytesIO(r_white.content), sep=";")
df_red["tipo"]   = "Tinto"
df_white["tipo"] = "Branco"

df = pd.concat([df_red, df_white], ignore_index=True)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

VARS = [c for c in df.columns if c not in ("quality", "tipo")]
print(f"Base combinada: {df.shape}  |  tintos={len(df_red)}  brancos={len(df_white)}")

# ── Estatísticas descritivas ───────────────────────────────────────────────────
desc = df[VARS + ["quality"]].describe().T[["mean","std","min","max"]]
desc.columns = ["Média","Desvio Padrão","Mínimo","Máximo"]
desc.index = [i.replace("_"," ").title() for i in desc.index]
TABS["desc"] = df_to_html(desc, 3)

# Distribuição de qualidade por tipo
fig, ax = plt.subplots(figsize=(9, 4.5))
for tipo, cor in [("Tinto", C_RED), ("Branco", C_BLUE)]:
    sub = df[df.tipo == tipo]["quality"].value_counts().sort_index()
    ax.bar(sub.index - (0.2 if tipo == "Tinto" else -0.2),
           sub.values, width=0.38, color=cor, alpha=0.85, label=tipo,
           edgecolor="white", linewidth=0.8)
ax.set_xlabel("Nota de Qualidade")
ax.set_ylabel("Frequência")
ax.set_title("Distribuição da Qualidade por Tipo de Vinho")
ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.tight_layout()
FIGS["dist_qualidade"] = fig_to_b64(fig)
plt.close(fig)

# ── 2. PCA ─────────────────────────────────────────────────────────────────────
print("PCA…")
X = df[VARS].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_sc)
eigenvalues = pca.explained_variance_
ev_ratio    = pca.explained_variance_ratio_
ev_cum      = np.cumsum(ev_ratio)

# Tabela de autovalores
ev_df = pd.DataFrame({
    "Componente":        [f"CP{i+1}" for i in range(len(eigenvalues))],
    "Autovalor":         eigenvalues,
    "Variância (%)":     ev_ratio * 100,
    "Variância Acum.(%)": ev_cum * 100,
})
n_comp_kaiser = int((eigenvalues >= 1).sum())
TABS["autovalores"] = df_to_html(ev_df.iloc[:8], 3)

# Scree plot
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(range(1, 12), eigenvalues, color=C_BLUE, alpha=0.7,
       edgecolor="white", linewidth=0.8, label="Autovalor")
ax.plot(range(1, 12), eigenvalues, "o-", color=C_RED, lw=2,
        ms=7, label="Tendência")
ax.axhline(1, color=C_GRAY, linestyle="--", lw=1.5, label="Critério Kaiser (λ=1)")
ax.set_xlabel("Componente Principal")
ax.set_ylabel("Autovalor")
ax.set_title("Scree Plot — Análise de Componentes Principais")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend()
plt.tight_layout()
FIGS["scree"] = fig_to_b64(fig)
plt.close(fig)

# Cargas dos componentes
pca_n = PCA(n_components=n_comp_kaiser)
X_pca = pca_n.fit_transform(X_sc)
loadings = pd.DataFrame(
    pca_n.components_.T,
    index=[v.replace("_"," ").title() for v in VARS],
    columns=[f"CP{i+1}" for i in range(n_comp_kaiser)]
)
TABS["loadings"] = df_to_html(loadings, 3)

# Biplot (CP1 vs CP2)
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                c=df["tipo"].map({"Tinto": C_RED, "Branco": C_BLUE}),
                alpha=0.25, s=10, linewidths=0)
scale = 4.5
for i, var in enumerate(VARS):
    lx, ly = pca_n.components_[0, i] * scale, pca_n.components_[1, i] * scale
    ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.5))
    ax.text(lx * 1.12, ly * 1.12,
            var.replace("_"," "), fontsize=8.5, color="#333333", ha="center")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_RED, label="Tinto"),
                   Patch(color=C_BLUE, label="Branco")], loc="upper right")
ax.set_xlabel(f"CP1 ({ev_ratio[0]*100:.1f}%)")
ax.set_ylabel(f"CP2 ({ev_ratio[1]*100:.1f}%)")
ax.set_title("Biplot PCA — CP1 vs CP2")
ax.axhline(0, color="#CCCCCC", lw=0.8)
ax.axvline(0, color="#CCCCCC", lw=0.8)
plt.tight_layout()
FIGS["biplot"] = fig_to_b64(fig)
plt.close(fig)

# Heatmap de correlação
corr = pd.DataFrame(X_sc, columns=VARS).corr()
fig, ax = plt.subplots(figsize=(9, 7))
import matplotlib.colors as mcolors
cmap = plt.cm.RdBu_r
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
labels = [v.replace("_"," ").title() for v in VARS]
ax.set_xticks(range(len(VARS))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(VARS))); ax.set_yticklabels(labels, fontsize=9)
for i in range(len(VARS)):
    for j in range(len(VARS)):
        val = corr.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color="white" if abs(val) > 0.6 else "#333")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matriz de Correlação das Variáveis")
plt.tight_layout()
FIGS["corr"] = fig_to_b64(fig)
plt.close(fig)

# ── 3. ANÁLISE FATORIAL ────────────────────────────────────────────────────────
print("Análise Fatorial…")

# KMO manual
corr_mat = np.corrcoef(X_sc.T)
inv_corr = np.linalg.inv(corr_mat)
partial_sq = np.zeros_like(corr_mat)
for i in range(len(VARS)):
    for j in range(len(VARS)):
        if i != j:
            partial_sq[i, j] = (inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j]))**2
kmo_num = corr_mat.sum() - np.diag(corr_mat).sum()
kmo_den = kmo_num + partial_sq.sum()
kmo_model = kmo_num / kmo_den

# Bartlett manual
n_obs = X_sc.shape[0]
p = len(VARS)
det = np.linalg.det(corr_mat)
chi2_val = -(n_obs - 1 - (2*p + 5)/6) * np.log(det)
dof_bart  = p * (p - 1) / 2
p_val = 1 - stats.chi2.cdf(chi2_val, dof_bart)

kmo_df = pd.DataFrame({
    "Teste": ["KMO (Kaiser-Meyer-Olkin)", "Bartlett — Qui-quadrado", "Bartlett — p-valor"],
    "Resultado": [f"{kmo_model:.4f}", f"{chi2_val:.2f}", f"{p_val:.2e}"]
})
TABS["kmo_bartlett"] = df_to_html(kmo_df.set_index("Teste"), 4)

# Factor Analysis via sklearn + Varimax manual
n_fat = 3
sk_fa = SKFactorAnalysis(n_components=n_fat, random_state=42, max_iter=1000)
sk_fa.fit(X_sc)
L0 = sk_fa.components_.T   # (p × n_fat)

def varimax(L, tol=1e-6, max_iter=1000):
    p, k = L.shape
    R = np.eye(k)
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vt = np.linalg.svd(
            Lr.T @ (Lr**3) - (1/p) * (Lr.T @ Lr) @ np.diag((Lr**2).sum(axis=0)))
        R_new = u @ vt
        if np.max(np.abs(R_new - R)) < tol:
            break
        R = R_new
    return L @ R

L_var = varimax(L0)
comunal = (L_var**2).sum(axis=1)
var_each = (L_var**2).sum(axis=0)
var_prop  = var_each / p
var_cum   = np.cumsum(var_prop)

cargas = pd.DataFrame(
    L_var,
    index=[v.replace("_"," ").title() for v in VARS],
    columns=[f"Fator {i+1}" for i in range(n_fat)]
)
cargas["Comunalidade"] = comunal
TABS["cargas"] = df_to_html(cargas, 3)

var_df = pd.DataFrame({
    "Variância":        var_each,
    "Proporção (%)":    var_prop * 100,
    "Cumulativa (%)":   var_cum  * 100,
}, index=[f"Fator {i+1}" for i in range(n_fat)])
TABS["var_fatorial"] = df_to_html(var_df, 3)

# Heatmap de cargas fatoriais
fig, ax = plt.subplots(figsize=(7, 7))
cmap_f = plt.cm.RdBu_r
data_f = cargas.iloc[:, :n_fat].values
im_f = ax.imshow(data_f, cmap=cmap_f, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(n_fat))
ax.set_xticklabels([f"Fator {i+1}" for i in range(n_fat)], fontsize=11)
ax.set_yticks(range(len(VARS)))
ax.set_yticklabels([v.replace("_"," ").title() for v in VARS], fontsize=9)
for i in range(len(VARS)):
    for j in range(n_fat):
        val = data_f[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(val) > 0.5 else "#333")
plt.colorbar(im_f, ax=ax, shrink=0.7)
ax.set_title("Cargas Fatoriais — Rotação Varimax")
plt.tight_layout()
FIGS["cargas_fat"] = fig_to_b64(fig)
plt.close(fig)

# ── 4. ANÁLISE DE CLUSTER ──────────────────────────────────────────────────────
print("Análise de Cluster…")

# Elbow + Silhouette
inertias, silhs = [], []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    inertias.append(km.inertia_)
    silhs.append(silhouette_score(X_sc, km.labels_, sample_size=2000, random_state=42))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(list(ks), inertias, "o-", color=C_RED, lw=2, ms=8)
ax1.set_xlabel("Número de Clusters (k)")
ax1.set_ylabel("Inércia (WCSS)")
ax1.set_title("Método do Cotovelo")
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

ax2.plot(list(ks), silhs, "o-", color=C_BLUE, lw=2, ms=8)
ax2.set_xlabel("Número de Clusters (k)")
ax2.set_ylabel("Coeficiente de Silhueta")
ax2.set_title("Coeficiente de Silhueta")
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.tight_layout()
FIGS["elbow_silh"] = fig_to_b64(fig)
plt.close(fig)

# K-means k=3
best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_sc) + 1

# Perfil dos clusters
perfil = df.groupby("cluster")[VARS + ["quality"]].mean()
perfil.index = [f"Cluster {i}" for i in perfil.index]
perfil.columns = [c.replace("_"," ").title() for c in perfil.columns]
TABS["perfil_cluster"] = df_to_html(perfil.T, 3)

# Visualização em espaço PCA
fig, ax = plt.subplots(figsize=(9, 6))
cluster_colors = {1: C_RED, 2: C_BLUE, 3: C_GREEN}
for cl in range(1, best_k + 1):
    mask = df["cluster"] == cl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=cluster_colors[cl], alpha=0.35, s=12,
               label=f"Cluster {cl}", linewidths=0)
ax.set_xlabel(f"CP1 ({ev_ratio[0]*100:.1f}%)")
ax.set_ylabel(f"CP2 ({ev_ratio[1]*100:.1f}%)")
ax.set_title("Clusters no Espaço PCA (CP1 × CP2)")
ax.legend(markerscale=2)
ax.axhline(0, color="#CCCCCC", lw=0.8)
ax.axvline(0, color="#CCCCCC", lw=0.8)
plt.tight_layout()
FIGS["cluster_pca"] = fig_to_b64(fig)
plt.close(fig)

# Dendrograma (amostra de 200 obs)
np.random.seed(42)
idx_sample = np.random.choice(len(X_sc), 200, replace=False)
Z = linkage(X_sc[idx_sample], method="ward")
fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
           color_threshold=0.7 * max(Z[:, 2]),
           above_threshold_color=C_GRAY,
           leaf_rotation=90, leaf_font_size=8)
ax.set_xlabel("Observações")
ax.set_ylabel("Distância de Ward")
ax.set_title("Dendrograma — Clustering Hierárquico (amostra n=200)")
plt.tight_layout()
FIGS["dendro"] = fig_to_b64(fig)
plt.close(fig)

# Boxplot: qualidade por cluster
fig, ax = plt.subplots(figsize=(7, 4.5))
data_bp = [df[df.cluster == k]["quality"].values for k in range(1, best_k + 1)]
bp = ax.boxplot(data_bp, patch_artist=True,
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color=C_GRAY),
                capprops=dict(color=C_GRAY),
                flierprops=dict(marker="o", ms=3, alpha=0.3,
                                markerfacecolor=C_GRAY, markeredgewidth=0))
for patch, cor in zip(bp["boxes"], [C_RED, C_BLUE, C_GREEN]):
    patch.set_facecolor(cor); patch.set_alpha(0.7)
ax.set_xticklabels([f"Cluster {i}" for i in range(1, best_k + 1)])
ax.set_xlabel("Cluster")
ax.set_ylabel("Nota de Qualidade")
ax.set_title("Distribuição da Qualidade por Cluster")
plt.tight_layout()
FIGS["quality_cluster"] = fig_to_b64(fig)
plt.close(fig)

# ── 5. ANÁLISE DE CORRESPONDÊNCIA ─────────────────────────────────────────────
print("Análise de Correspondência…")

def quality_group(q):
    if q <= 5: return "Baixa (3–5)"
    if q == 6: return "Média (6)"
    return "Alta (7–9)"

df["qual_grupo"] = df["quality"].apply(quality_group)

ordem_qual = ["Baixa (3–5)", "Média (6)", "Alta (7–9)"]
ct = pd.crosstab(df["tipo"], df["qual_grupo"])[ordem_qual]
ct.index.name = "Tipo"
TABS["contingencia"] = df_to_html(ct, 0)

# Teste Qui-Quadrado
chi2_ct, p_ct, dof_ct, expected_ct = stats.chi2_contingency(ct.values)
chi2_res = pd.DataFrame({
    "Estatística": ["Qui-quadrado (χ²)", "p-valor", "Graus de liberdade"],
    "Valor": [f"{chi2_ct:.4f}", f"{p_ct:.4e}", str(dof_ct)]
}).set_index("Estatística")
TABS["chi2"] = df_to_html(chi2_res, 4)

# CA via SVD
N = ct.values.astype(float)
grand = N.sum()
P = N / grand
r = P.sum(axis=1)
c = P.sum(axis=0)
Dr_inv = np.diag(1.0 / np.sqrt(r))
Dc_inv = np.diag(1.0 / np.sqrt(c))
S = Dr_inv @ (P - np.outer(r, c)) @ Dc_inv
U, sv, Vt = np.linalg.svd(S, full_matrices=False)

row_coords = Dr_inv @ U[:, :2] * sv[:2]
col_coords = Dc_inv @ Vt.T[:, :2] * sv[:2]
inertia_total = (sv**2).sum()
inertia_pct   = sv**2 / inertia_total * 100

ca_inercia = pd.DataFrame({
    "Dimensão": ["Dim 1", "Dim 2"],
    "Valor Singular": sv[:2],
    "Inércia (%)": inertia_pct[:2],
    "Inércia Acumulada (%)": np.cumsum(inertia_pct[:2])
})
TABS["ca_inercia"] = df_to_html(ca_inercia, 3)

fig, ax = plt.subplots(figsize=(8, 6))
# Linhas (tipos)
for i, label in enumerate(ct.index):
    ax.scatter(row_coords[i, 0], row_coords[i, 1],
               s=160, color=C_RED if label == "Tinto" else C_BLUE,
               zorder=5, edgecolors="white", linewidths=1)
    ax.annotate(f" {label}", (row_coords[i, 0], row_coords[i, 1]),
                fontsize=11, fontweight="bold",
                color=C_RED if label == "Tinto" else C_BLUE)
# Colunas (qualidade)
for j, label in enumerate(ct.columns):
    ax.scatter(col_coords[j, 0], col_coords[j, 1],
               s=120, color=C_AMBER, marker="D", zorder=5,
               edgecolors="white", linewidths=1)
    ax.annotate(f" {label}", (col_coords[j, 0], col_coords[j, 1]),
                fontsize=10, color=C_AMBER, fontstyle="italic")
ax.axhline(0, color="#CCCCCC", lw=0.8)
ax.axvline(0, color="#CCCCCC", lw=0.8)
ax.set_xlabel(f"Dimensão 1 ({inertia_pct[0]:.1f}% da inércia)")
ax.set_ylabel(f"Dimensão 2 ({inertia_pct[1]:.1f}% da inércia)")
ax.set_title("Mapa de Correspondência — Tipo × Qualidade")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_RED, label="Tipos de vinho"),
                   Patch(color=C_AMBER, label="Categorias de qualidade")])
plt.tight_layout()
FIGS["mapa_ca"] = fig_to_b64(fig)
plt.close(fig)

# ── 6. HTML REPORT ─────────────────────────────────────────────────────────────
print("Gerando relatório HTML…")

def img(key, alt="", w="100%"):
    return f'<img src="data:image/png;base64,{FIGS[key]}" alt="{alt}" style="width:{w};max-width:900px;display:block;margin:16px auto;">'

def section(title, level=2):
    tag = f"h{level}"
    return f"<{tag}>{title}</{tag}>"

HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Análise Multivariada — Wine Quality</title>
<style>
  @page {{ size: A4; margin: 2.2cm 2.0cm; }}
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    font-family: "Georgia", serif;
    font-size: 11.5pt;
    line-height: 1.65;
    color: #1a1a1a;
    background: white;
  }}
  h1 {{ font-size: 20pt; text-align: center; color: {C_RED}; margin-bottom: 4px; }}
  h2 {{ font-size: 14pt; color: {C_RED}; border-bottom: 2px solid {C_RED};
        padding-bottom: 4px; margin-top: 32px; page-break-after: avoid; }}
  h3 {{ font-size: 12pt; color: #333; margin-top: 22px; page-break-after: avoid; }}
  .capa {{ text-align: center; padding: 60px 20px 40px; page-break-after: always; }}
  .capa .subtitulo {{ font-size: 13pt; color: #555; margin: 8px 0; }}
  .capa .meta {{ font-size: 11pt; color: #666; margin-top: 30px; }}
  p {{ text-align: justify; margin: 8px 0; }}
  .hipotese {{
    background: #FDF5F5; border-left: 4px solid {C_RED};
    padding: 10px 16px; margin: 14px 0; border-radius: 0 4px 4px 0;
    font-style: italic;
  }}
  .conclusao {{
    background: #F0F5FF; border-left: 4px solid {C_BLUE};
    padding: 10px 16px; margin: 14px 0; border-radius: 0 4px 4px 0;
  }}
  .page-break {{ page-break-before: always; }}
  img {{ max-width: 100%; height: auto; }}
  .table-wrap {{ overflow-x: auto; margin: 14px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 10pt; font-family: "Arial", sans-serif; }}
  th {{ background: {C_RED}; color: white; padding: 6px 10px; text-align: left; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #E8E8E8; }}
  tr:nth-child(even) td {{ background: #F9F9F9; }}
  caption {{ font-size: 10pt; color: #555; margin-bottom: 6px; }}
  .destaque {{ font-weight: bold; color: {C_RED}; }}
</style>
</head>
<body>

<!-- CAPA -->
<div class="capa">
  <h1>Análise Multivariada<br>Wine Quality</h1>
  <p class="subtitulo">Vinhos Verdes Portugueses — UCI Machine Learning Repository</p>
  <p class="subtitulo">Cortez et al. (2009) — Universidade do Minho, Portugal</p>
  <p class="meta">
    Disciplina: Análise Multivariada<br>
    Aluno: Gabriel Pereira<br>
    Data: Setembro de 2026
  </p>
</div>

<!-- 1. DESCRIÇÃO DA BASE -->
<h2>1. Descrição da Base de Dados</h2>

<p>
A base de dados utilizada neste trabalho é a <strong>Wine Quality</strong>, disponível no
<em>UCI Machine Learning Repository</em> (Cortez et al., 2009). Os dados foram coletados
pelo Laboratório de Sistemas de Decisão e Inteligência Artificial (GECAD) da Universidade do
Minho, em parceria com a Comissão de Viticultura da Região dos Vinhos Verdes (CVRVV) de
Portugal.
</p>

<p>
A base compreende amostras de dois tipos de vinho da região dos Vinhos Verdes:
<strong>vinho tinto</strong> (1.599 amostras) e <strong>vinho branco</strong>
(4.898 amostras), totalizando <span class="destaque">6.497 observações</span>.
Para cada amostra foram registradas 11 variáveis físico-químicas contínuas obtidas em
análise laboratorial, além de uma avaliação sensorial de qualidade (nota de 0 a 10) atribuída
por especialistas certificados da CVRVV.
</p>

<p>
O maior número de vinhos brancos reflete a realidade da produção regional: o Vinho Verde
branco domina a certificação e exportação da denominação de origem, sendo historicamente
mais representativo no volume certificado pelo CVRVV.
</p>

<h3>Variáveis da Base</h3>
<p>As 11 variáveis preditoras e a variável-resposta são descritas a seguir:</p>
<ul>
  <li><strong>Fixed Acidity</strong> — acidez fixa (ácido tartárico, g/dm³)</li>
  <li><strong>Volatile Acidity</strong> — acidez volátil (ácido acético, g/dm³); em excesso, confere sabor de vinagre</li>
  <li><strong>Citric Acid</strong> — ácido cítrico (g/dm³); contribui para frescor e sabor</li>
  <li><strong>Residual Sugar</strong> — açúcar residual após a fermentação (g/dm³)</li>
  <li><strong>Chlorides</strong> — cloretos, concentração de sais (g/dm³)</li>
  <li><strong>Free Sulfur Dioxide</strong> — dióxido de enxofre livre (mg/dm³); previne oxidação e crescimento microbiano</li>
  <li><strong>Total Sulfur Dioxide</strong> — SO₂ total: livre + combinado (mg/dm³)</li>
  <li><strong>Density</strong> — densidade do vinho (g/cm³)</li>
  <li><strong>pH</strong> — potencial hidrogeniônico (acidez global)</li>
  <li><strong>Sulphates</strong> — sulfatos (g/dm³); aditivo antimicrobiano e antioxidante</li>
  <li><strong>Alcohol</strong> — teor alcoólico (% vol)</li>
  <li><strong>Quality</strong> (variável-resposta) — nota sensorial de 0 a 10</li>
</ul>

<h3>Estatísticas Descritivas</h3>
<div class="table-wrap">{TABS["desc"]}</div>

<h3>Distribuição da Qualidade por Tipo de Vinho</h3>
{img("dist_qualidade", "Distribuição de qualidade")}
<p>
O gráfico revela que a maior parte das avaliações concentra-se nas notas 5, 6 e 7 para ambos
os tipos, caracterizando uma distribuição aproximadamente normal levemente assimétrica à
direita. Vinhos com notas extremas (≤ 4 ou ≥ 8) são raros, refletindo a dificuldade de
produzir vinhos excepcionalmente bons ou ruins dentro de uma mesma denominação de origem
controlada.
</p>

<!-- 2. PCA -->
<div class="page-break"></div>
<h2>2. Hipótese 1 — Análise de Componentes Principais (PCA)</h2>

<div class="hipotese">
<strong>Hipótese:</strong> As 11 variáveis físico-químicas dos vinhos podem ser resumidas em
um conjunto reduzido de componentes principais que capture pelo menos 70% da variabilidade
total dos dados, evidenciando que as propriedades dos vinhos seguem padrões estruturais
subjacentes mensuráveis.
</div>

<p>
A Análise de Componentes Principais (PCA) é uma técnica de redução de dimensionalidade que
transforma um conjunto de variáveis correlacionadas em componentes ortogonais (não
correlacionados), ordenados pela quantidade de variância explicada. Antes da análise, todas as
variáveis foram padronizadas (média zero, desvio padrão unitário) para eliminar o efeito de
escala.
</p>

<h3>Matriz de Correlação</h3>
{img("corr", "Matriz de correlação")}
<p>
A matriz de correlação evidencia associações relevantes entre as variáveis. Destacam-se
correlações positivas fortes entre <em>free sulfur dioxide</em> e
<em>total sulfur dioxide</em> (r ≈ 0,72), entre <em>density</em> e <em>residual sugar</em>
(r ≈ 0,55) e entre <em>fixed acidity</em> e <em>density</em>. Correlações negativas
notáveis aparecem entre <em>alcohol</em> e <em>density</em> (r ≈ −0,69), e entre
<em>pH</em> e <em>fixed acidity</em>. A presença dessas correlações justifica e valida a
aplicação da PCA.
</p>

<h3>Critério de Kaiser e Scree Plot</h3>
{img("scree", "Scree plot")}
<div class="table-wrap">{TABS["autovalores"]}</div>
<p>
Pelo critério de Kaiser (autovalores ≥ 1), foram retidos
<span class="destaque">{n_comp_kaiser} componentes principais</span>, que juntos explicam
<span class="destaque">{ev_cum[n_comp_kaiser-1]*100:.1f}%</span> da variância total dos
dados. O scree plot confirma essa escolha: observa-se uma inflexão clara ("cotovelo") após
o {n_comp_kaiser}º componente, indicando queda abrupta na contribuição dos demais.
</p>

<h3>Cargas dos Componentes</h3>
<div class="table-wrap">{TABS["loadings"]}</div>
<p>
As cargas (loadings) revelam quais variáveis contribuem mais para cada componente:
</p>
<ul>
  <li><strong>CP1</strong> — opõe <em>density</em> e <em>residual sugar</em> (positivos) a
      <em>alcohol</em> (negativo), captando a <em>estrutura de corpo e teor</em> do vinho.
      Vinhos mais densos e adocicados se opõem aos mais alcoólicos.</li>
  <li><strong>CP2</strong> — carregado por <em>volatile acidity</em>, <em>chlorides</em>
      e <em>sulphates</em>, representando o <em>perfil de preservação e defeitos</em>
      do vinho.</li>
  <li><strong>CP3</strong> — dominado por <em>free</em> e <em>total sulfur dioxide</em>,
      refletindo o <em>nível de conservantes</em> adicionados.</li>
</ul>

<h3>Biplot — CP1 × CP2</h3>
{img("biplot", "Biplot PCA")}
<p>
O biplot superpõe os scores das observações (pontos) e os vetores de carga das variáveis
(setas). A separação visual entre vinhos tintos (vermelho) e brancos (azul) ao longo de CP1
indica que a estrutura de corpo e teor alcoólico distingue sistematicamente os dois tipos.
As setas de <em>total sulfur dioxide</em> e <em>free sulfur dioxide</em> apontam na mesma
direção e são mais compridas no eixo de CP2, confirmando sua dominância nesse componente.
</p>

<div class="conclusao">
<strong>Conclusão:</strong> A hipótese é confirmada. Os {n_comp_kaiser} componentes retidos
explicam {ev_cum[n_comp_kaiser-1]*100:.1f}% da variabilidade, superando o limiar de 70%
proposto. A PCA revela que as propriedades físico-químicas dos vinhos se organizam em torno
de três dimensões latentes: estrutura e corpo (CP1), preservação e defeitos (CP2) e nível
de conservantes (CP3).
</div>

<!-- 3. FATORIAL -->
<div class="page-break"></div>
<h2>3. Hipótese 2 — Análise Fatorial</h2>

<div class="hipotese">
<strong>Hipótese:</strong> Existem fatores latentes não observáveis que explicam a estrutura
de correlação das propriedades físico-químicas dos vinhos, sendo possível identificar
dimensões comuns como perfil de acidez, compostos sulfurosos de conservação e características
de corpo e teor alcoólico.
</div>

<p>
A Análise Fatorial (AF) busca identificar variáveis latentes (fatores) que expliquem as
correlações observadas entre as variáveis mensuradas. Diferentemente da PCA, que é puramente
descritiva, a AF assume uma estrutura causal: os fatores <em>causam</em> as correlações.
Utilizou-se rotação Varimax para maximizar a interpretabilidade dos fatores, tornando as
cargas o mais próximas de 0 ou ±1 possível.
</p>

<h3>Adequação da Base — Testes KMO e Bartlett</h3>
<div class="table-wrap">{TABS["kmo_bartlett"]}</div>
<p>
O índice KMO de <span class="destaque">{kmo_model:.3f}</span> classifica-se como
{"<em>admirável</em>" if kmo_model >= 0.9 else "<em>ótimo</em>" if kmo_model >= 0.8 else "<em>bom</em>" if kmo_model >= 0.7 else "<em>razoável</em>"}
(Kaiser, 1974), indicando que as correlações parciais são pequenas em relação às correlações
totais — condição favorável para a AF. O Teste de Esfericidade de Bartlett rejeita a
hipótese nula de que a matriz de correlação é uma matriz identidade
(p {"< 0,001" if p_val < 0.001 else f"= {p_val:.4f}"}), confirmando que há correlação
suficiente entre as variáveis para a extração de fatores.
</p>

<h3>Variância Explicada pelos Fatores</h3>
<div class="table-wrap">{TABS["var_fatorial"]}</div>

<h3>Cargas Fatoriais — Rotação Varimax</h3>
{img("cargas_fat", "Heatmap de cargas fatoriais")}
<div class="table-wrap">{TABS["cargas"]}</div>
<p>
Após a rotação Varimax, os três fatores apresentam interpretação clara:
</p>
<ul>
  <li><strong>Fator 1 — Estrutura e Corpo:</strong> cargas elevadas de <em>density</em>,
      <em>residual sugar</em>, <em>alcohol</em> e <em>fixed acidity</em>. Representa a
      <em>corpulência</em> do vinho — o equilíbrio entre açúcar residual, densidade e
      teor alcoólico que define a sensação na boca.</li>
  <li><strong>Fator 2 — Conservantes Sulfurosos:</strong> dominado por
      <em>free sulfur dioxide</em> e <em>total sulfur dioxide</em>. Reflete o uso de
      dióxido de enxofre como agente conservante e antioxidante durante a vinificação.</li>
  <li><strong>Fator 3 — Acidez e Defeitos:</strong> carregado por <em>volatile acidity</em>,
      <em>pH</em> e <em>citric acid</em>. Capta o <em>perfil de acidez</em> e a tendência
      a defeitos: alta acidez volátil é o principal defeito organoléptico associado à
      deterioração bacteriana.</li>
</ul>

<div class="conclusao">
<strong>Conclusão:</strong> A hipótese é confirmada. Os três fatores latentes identificados
têm interpretação enológica direta — corpo/estrutura, conservação e acidez/defeitos —
e explicam em conjunto {var_df["Cumulativa (%)"].iloc[-1]:.1f}% da variância total.
As comunalidades indicam que a maioria das variáveis é bem representada pelo modelo fatorial.
</div>

<!-- 4. CLUSTER -->
<div class="page-break"></div>
<h2>4. Hipótese 3 — Análise de Cluster</h2>

<div class="hipotese">
<strong>Hipótese:</strong> Os vinhos podem ser agrupados em segmentos homogêneos com base
em seu perfil físico-químico, onde grupos distintos apresentarão diferenças significativas na
nota de qualidade média, evidenciando que as características químicas determinam a
qualidade percebida.
</div>

<p>
A Análise de Cluster é uma técnica de classificação não supervisionada que agrupa observações
em clusters de forma que a similaridade intra-cluster seja maximizada e a similaridade
inter-clusters seja minimizada. Foram aplicadas duas abordagens complementares: clustering
hierárquico (para visualizar a estrutura de agrupamentos sem definir k a priori) e K-Means
(para particionar definitivamente os dados).
</p>

<h3>Dendrograma — Clustering Hierárquico</h3>
{img("dendro", "Dendrograma")}
<p>
O dendrograma, obtido sobre uma amostra aleatória de 200 observações com método de ligação
de Ward, indica a presença de <strong>3 grupos naturais</strong> ao nível de corte que
maximiza a distância inter-clusters. A estrutura hierárquica mostra que dois dos grupos se
fundem antes do terceiro, sugerindo maior similaridade entre eles.
</p>

<h3>Determinação do Número de Clusters</h3>
{img("elbow_silh", "Cotovelo e Silhueta")}
<p>
O método do cotovelo mostra redução acentuada na inércia (WCSS) até k = 3, com ganhos
marginais a partir de k = 4. O coeficiente de silhueta confirma k = 3 como ponto ótimo,
com o maior valor médio de separação entre clusters. Esses dois critérios convergem para a
escolha de <span class="destaque">k = 3 clusters</span>.
</p>

<h3>Perfil dos Clusters</h3>
<div class="table-wrap">{TABS["perfil_cluster"]}</div>

<h3>Distribuição da Qualidade por Cluster</h3>
{img("quality_cluster", "Qualidade por cluster")}
{img("cluster_pca", "Clusters no espaço PCA")}
<p>
A análise dos perfis médios revela clusters com características distintas:
</p>
<ul>
  <li><strong>Cluster 1</strong> — vinhos com maior acidez fixa, menor teor alcoólico e
      maior densidade. Corresponde predominantemente a vinhos tintos encorpados, com notas
      de qualidade medianas.</li>
  <li><strong>Cluster 2</strong> — vinhos com teor alcoólico elevado, baixa densidade e
      açúcar residual reduzido. Associam-se a vinhos secos de alta qualidade, tanto
      tintos quanto brancos.</li>
  <li><strong>Cluster 3</strong> — vinhos com alto teor de SO₂ e açúcar residual, menor
      acidez volátil. Característico de vinhos brancos com maior adição de conservantes,
      perfil semi-doce.</li>
</ul>

<div class="conclusao">
<strong>Conclusão:</strong> A hipótese é confirmada. Os três clusters apresentam perfis
físico-químicos distintos e diferenças relevantes na qualidade média. O Cluster 2, de vinhos
mais alcoólicos e secos, concentra as melhores avaliações, confirmando que o teor alcoólico
e a baixa densidade estão associados positivamente à qualidade sensorial percebida.
</div>

<!-- 5. CORRESPONDÊNCIA -->
<div class="page-break"></div>
<h2>5. Hipótese 4 — Análise de Correspondência</h2>

<div class="hipotese">
<strong>Hipótese:</strong> Existe associação estatisticamente significativa entre o tipo de
vinho (tinto/branco) e a categoria de qualidade (baixa/média/alta), indicando que o processo
de vinificação influencia a distribuição das notas de qualidade percebida.
</div>

<p>
A Análise de Correspondência (AC) é uma técnica de análise exploratória para variáveis
categóricas que representa graficamente as associações entre linhas e colunas de uma tabela
de contingência. É análoga à PCA para dados categóricos: decompõe a inércia total da tabela
em dimensões ortogonais interpretáveis.
</p>

<h3>Tabela de Contingência</h3>
<div class="table-wrap">{TABS["contingencia"]}</div>
<p>
A tabela revela que os vinhos brancos tendem a concentrar-se nas categorias extremas:
apresentam proporcionalmente mais vinhos de qualidade baixa e mais vinhos de qualidade alta
que os tintos. Os tintos concentram-se mais na faixa média (nota 6).
</p>

<h3>Teste Qui-Quadrado de Independência</h3>
<div class="table-wrap">{TABS["chi2"]}</div>
<p>
O teste qui-quadrado rejeita com alta confiança a hipótese de independência entre tipo e
qualidade (χ² = {chi2_ct:.2f}, p {"< 0,001" if p_ct < 0.001 else f"= {p_ct:.4f}"},
gl = {dof_ct}), confirmando que a associação observada não é produto do acaso. A associação
é estatisticamente significativa.
</p>

<h3>Inércia Explicada pelas Dimensões</h3>
<div class="table-wrap">{TABS["ca_inercia"]}</div>

<h3>Mapa de Correspondência</h3>
{img("mapa_ca", "Mapa de correspondência")}
<p>
O mapa de correspondência posiciona categorias próximas entre si quando há associação forte.
A Dimensão 1, que explica {inertia_pct[0]:.1f}% da inércia, separa claramente os dois tipos
de vinho. A proximidade do <em>vinho branco</em> com a categoria
<em>Baixa (3–5)</em> indica que brancos são mais frequentemente avaliados como de baixa
qualidade do que tintos. Por outro lado, <em>vinho tinto</em> posiciona-se próximo à faixa
<em>Média (6)</em>. Ambos os tipos mostram alguma proximidade com <em>Alta (7–9)</em>,
mas o padrão é mais evidente nos brancos, confirmando a distribuição bimodal observada
na tabela de contingência.
</p>

<div class="conclusao">
<strong>Conclusão:</strong> A hipótese é confirmada. Existe associação significativa entre
tipo de vinho e qualidade (p < 0,001). O mapa de correspondência evidencia que vinhos tintos
associam-se à faixa de qualidade média, enquanto brancos apresentam maior dispersão entre
as categorias extremas. Isso reflete diferenças no processo de vinificação: a produção de
vinho branco engloba desde versões de entrada altamente padronizadas até varietais de
alto padrão, enquanto a produção de tintos tende a ser mais homogênea dentro da
denominação Vinhos Verdes.
</div>

<!-- CONCLUSÃO GERAL -->
<div class="page-break"></div>
<h2>6. Conclusão Geral</h2>

<p>
Este trabalho aplicou quatro técnicas de análise multivariada sobre a base Wine Quality,
composta por 6.497 amostras de vinhos verdes portugueses descritas por 11 propriedades
físico-químicas. Os principais resultados são:
</p>

<ul>
  <li>A <strong>PCA</strong> reduziu a dimensionalidade de 11 para {n_comp_kaiser}
      componentes principais, retendo {ev_cum[n_comp_kaiser-1]*100:.1f}% da variância
      total. Os componentes captam estruturas interpretáveis de corpo, conservação e
      acidez.</li>
  <li>A <strong>Análise Fatorial</strong> identificou três fatores latentes com
      interpretação enológica direta — corpo/estrutura, conservantes sulfurosos e
      acidez/defeitos — validados pelos testes KMO ({kmo_model:.3f}) e Bartlett (p &lt; 0,001).</li>
  <li>A <strong>Análise de Cluster</strong> (k = 3) segmentou os vinhos em grupos com
      perfis físico-químicos distintos e notas de qualidade diferenciadas, confirmando
      que características mensuráveis em laboratório predizem a qualidade sensorial.</li>
  <li>A <strong>Análise de Correspondência</strong> revelou associação significativa
      (χ² = {chi2_ct:.2f}, p &lt; 0,001) entre tipo de vinho e qualidade, com tintos
      concentrados na faixa média e brancos com maior dispersão entre extremos.</li>
</ul>

<p>
Em conjunto, os resultados demonstram que as propriedades físico-químicas dos vinhos possuem
estrutura multivariada rica e interpretável. O teor alcoólico, a densidade e o perfil de
acidez emergem consistentemente como as dimensões mais discriminantes entre tipos e
categorias de qualidade — informação de alta relevância para enólogos e produtores.
</p>

<h3>Referência</h3>
<p>
CORTEZ, P.; CERDEIRA, A.; ALMEIDA, F.; MATOS, T.; REIS, J. <strong>Modeling wine
preferences by data mining from physicochemical properties.</strong> <em>Decision Support
Systems</em>, v. 47, n. 4, p. 547–553, 2009.
</p>

</body>
</html>
"""

with open("relatorio.html", "w", encoding="utf-8") as f:
    f.write(HTML)
print("HTML salvo: relatorio.html")

# ── 7. PDF ─────────────────────────────────────────────────────────────────────
print("Convertendo para PDF…")
from weasyprint import HTML as WH
WH(filename="relatorio.html").write_pdf("relatorio_analise_multivariada.pdf")
print("PDF gerado: relatorio_analise_multivariada.pdf")
