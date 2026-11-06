"""Cálculos estatísticos do trabalho (sem geração de relatório)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA

DADOS = Path(__file__).with_name("dados_clientes.csv")
REG = {1: "Norte", 2: "Sul", 3: "Centro"}

def carregar():
    df = pd.read_csv(DADOS)
    df["Regiao_nome"] = df.Regiao.map(REG)
    return df

# ---------------------------------------------------------------- utilitários
def box_m(X, g):
    """Teste M de Box (aproximação qui-quadrado)."""
    grupos = np.unique(g)
    k, p = len(grupos), X.shape[1]
    ns = np.array([np.sum(g == c) for c in grupos])
    S = [np.cov(X[g == c].T) for c in grupos]
    N = ns.sum()
    Sp = sum((n - 1) * s for n, s in zip(ns, S)) / (N - k)
    M = (N - k) * np.log(np.linalg.det(Sp)) - sum((n - 1) * np.log(np.linalg.det(s)) for n, s in zip(ns, S))
    c1 = (np.sum(1 / (ns - 1)) - 1 / (N - k)) * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    chi = M * (1 - c1)
    df = p * (p + 1) * (k - 1) / 2
    return dict(M=M, chi2=chi, gl=df, p=stats.chi2.sf(chi, df), dets={c: np.linalg.det(s) for c, s in zip(grupos, S)},
                det_pooled=np.linalg.det(Sp))

def mardia(X):
    X = np.asarray(X, float)
    n, p = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    D = Xc @ np.linalg.inv(S) @ Xc.T
    b1 = (D**3).sum() / n**2
    b2 = (np.diag(D)**2).sum() / n
    chi = n * b1 / 6
    gl = p * (p + 1) * (p + 2) / 6
    z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return dict(b1=b1, chi2=chi, gl=gl, p_ass=stats.chi2.sf(chi, gl),
                b2=b2, esperado=p * (p + 2), z=z, p_curt=2 * stats.norm.sf(abs(z)))

def vif(X):
    R = np.corrcoef(X.T)
    return np.diag(np.linalg.inv(R))

def bartlett_esfericidade(X):
    n, p = X.shape
    R = np.corrcoef(X.T)
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    gl = p * (p - 1) / 2
    return dict(chi2=chi, gl=gl, p=stats.chi2.sf(chi, gl), detR=np.linalg.det(R))

# ---------------------------------------------------------------- Parte I
def parte1(df):
    y = "Score_Satisfacao"
    r = {}
    r["desc"] = df.groupby("Regiao_nome")[y].agg(["count", "mean", "std", "min", "median", "max"]).loc[["Norte", "Sul", "Centro"]]
    modelo = ols(f"{y} ~ C(Regiao)", data=df).fit()
    r["anova"] = anova_lm(modelo, typ=1)
    ssb, ssw = r["anova"]["sum_sq"]
    r["eta2"] = ssb / (ssb + ssw)
    dfb, dfw = r["anova"]["df"]
    msw = ssw / dfw
    r["omega2"] = (ssb - dfb * msw) / (ssb + ssw + msw)
    r["resid"] = modelo.resid
    r["shapiro_grupo"] = {REG[g]: stats.shapiro(d[y]) for g, d in df.groupby("Regiao")}
    r["shapiro_resid"] = stats.shapiro(modelo.resid)
    r["ks_resid"] = stats.kstest((modelo.resid - modelo.resid.mean()) / modelo.resid.std(ddof=1), "norm")
    amostras = [d[y].values for _, d in df.groupby("Regiao")]
    r["levene"] = stats.levene(*amostras, center="mean")
    r["brown_forsythe"] = stats.levene(*amostras, center="median")
    r["bartlett"] = stats.bartlett(*amostras)
    r["kruskal"] = stats.kruskal(*amostras)
    r["skew_kurt"] = {REG[g]: (stats.skew(d[y]), stats.kurtosis(d[y])) for g, d in df.groupby("Regiao")}
    # Welch (robustez a heterocedasticidade)
    ns = np.array([len(a) for a in amostras]); ms = np.array([a.mean() for a in amostras]); vs = np.array([a.var(ddof=1) for a in amostras])
    w = ns / vs; mw = (w * ms).sum() / w.sum(); k = len(amostras)
    A = (w * (ms - mw)**2).sum() / (k - 1)
    lam = ((1 - w / w.sum())**2 / (ns - 1)).sum()
    B = 1 + 2 * (k - 2) / (k**2 - 1) * lam
    F = A / B; df2 = (k**2 - 1) / (3 * lam)
    r["welch"] = dict(F=F, gl1=k - 1, gl2=df2, p=stats.f.sf(F, k - 1, df2))
    tk = pairwise_tukeyhsd(df[y], df["Regiao_nome"], alpha=0.05)
    r["tukey"] = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
    # Bonferroni (t de Student com QME combinado)
    linhas = []
    pares = [("Norte", "Sul"), ("Norte", "Centro"), ("Sul", "Centro")]
    for a, b in pares:
        xa, xb = df.loc[df.Regiao_nome == a, y], df.loc[df.Regiao_nome == b, y]
        dif = xb.mean() - xa.mean()
        se = np.sqrt(msw * (1 / len(xa) + 1 / len(xb)))
        t = dif / se
        p = min(1, 2 * stats.t.sf(abs(t), dfw) * 3)
        linhas.append(dict(grupo1=a, grupo2=b, dif=dif, ep=se, t=t, p_bonf=p))
    r["bonf"] = pd.DataFrame(linhas)
    r["msw"] = msw; r["dfw"] = dfw
    return r

# ---------------------------------------------------------------- Parte II
DV2 = ["Gasto_Online", "Gasto_Loja", "NPS_Pontuacao"]
def parte2(df):
    r = {}
    X = df[DV2].values; g = df.Regiao.values
    r["desc"] = df.groupby("Regiao_nome")[DV2].agg(["mean", "std"]).loc[["Norte", "Sul", "Centro"]]
    r["corr"] = df[DV2].corr()
    r["bartlett_esf"] = bartlett_esfericidade(X)
    r["boxm"] = box_m(X, g)
    r["mardia_grupo"] = {REG[k]: mardia(df.loc[df.Regiao == k, DV2].values) for k in [1, 2, 3]}
    resid = X - df.groupby("Regiao")[DV2].transform("mean").values
    r["mardia_resid"] = mardia(resid)
    r["levene"] = {v: stats.levene(*[d[v].values for _, d in df.groupby("Regiao")], center="mean") for v in DV2}
    m = MANOVA.from_formula("Gasto_Online + Gasto_Loja + NPS_Pontuacao ~ C(Regiao)", data=df).mv_test()
    tab = m.results["C(Regiao)"]["stat"].copy()
    r["manova"] = tab
    # matrizes H e E, autovalores
    medias = df.groupby("Regiao")[DV2].mean()
    gm = X.mean(0)
    H = sum(np.sum(g == k) * np.outer(medias.loc[k] - gm, medias.loc[k] - gm) for k in [1, 2, 3])
    E = sum((df.loc[df.Regiao == k, DV2].values - medias.loc[k].values).T @ (df.loc[df.Regiao == k, DV2].values - medias.loc[k].values) for k in [1, 2, 3])
    ev = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(E, H))))[::-1]
    r["H"], r["E"], r["eig"] = H, E, ev
    wilks = float(tab.loc["Wilks' lambda", "Value"])
    s = min(len(DV2), 2)
    r["eta2_mult"] = 1 - wilks**(1 / s)
    # ANOVAs univariadas
    lin = []
    for v in DV2:
        a = anova_lm(ols(f"{v} ~ C(Regiao)", data=df).fit())
        ssb, ssw = a["sum_sq"]
        lin.append(dict(variavel=v, F=a["F"].iloc[0], p=a["PR(>F)"].iloc[0], p_bonf=min(1, a["PR(>F)"].iloc[0] * 3), eta2=ssb / (ssb + ssw)))
    r["univ"] = pd.DataFrame(lin)
    # função discriminante (combinação que mais separa regiões) — coeficientes padronizados
    vals, vecs = np.linalg.eig(np.linalg.solve(E, H))
    idx = np.argsort(np.real(vals))[::-1]
    vecs = np.real(vecs[:, idx])
    sd_within = np.sqrt(np.diag(E) / (len(df) - 3))
    std = vecs * sd_within[:, None]
    std = std / np.abs(std).max(0)
    r["disc_std"] = pd.DataFrame(std[:, :2], index=DV2, columns=["Função 1", "Função 2"])
    # Tukey para cada DV
    r["tukey"] = {}
    for v in DV2:
        tk = pairwise_tukeyhsd(df[v], df["Regiao_nome"])
        r["tukey"][v] = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
    return r

# ---------------------------------------------------------------- Parte III (LDA)
X3 = ["Gasto_Online", "Gasto_Loja", "Freq_Online", "Freq_Loja", "Tempo_Relacao"]
def lda_fit(X, y):
    g0, g1 = X[y == 0], X[y == 1]
    n0, n1 = len(g0), len(g1)
    m0, m1 = g0.mean(0), g1.mean(0)
    Sp = ((n0 - 1) * np.cov(g0.T) + (n1 - 1) * np.cov(g1.T)) / (n0 + n1 - 2)
    a = np.linalg.solve(Sp, m1 - m0)
    return dict(a=a, m0=m0, m1=m1, Sp=Sp, n0=n0, n1=n1)

def lda_classifica(fit, X, prior1=None):
    a, m0, m1 = fit["a"], fit["m0"], fit["m1"]
    z = X @ a
    corte = 0.5 * (m0 + m1) @ a
    if prior1 is not None:
        corte += np.log((1 - prior1) / prior1)
    return (z > corte).astype(int), z, corte

def parte3(df):
    r = {}
    X = df[X3].values.astype(float); y = df.Status_Fidelidade.values
    r["desc"] = df.groupby("Status_Fidelidade")[X3].agg(["mean", "std"]).T
    # testes univariados (Wilks univariado = F)
    lin = []
    for v in X3:
        a, b = df.loc[y == 0, v], df.loc[y == 1, v]
        F, p = stats.f_oneway(a, b)
        ssw = ((a - a.mean())**2).sum() + ((b - b.mean())**2).sum(); sst = ((df[v] - df[v].mean())**2).sum()
        lin.append(dict(variavel=v, media0=a.mean(), media1=b.mean(), lambda_wilks=ssw / sst, F=F, p=p))
    r["univ"] = pd.DataFrame(lin)
    r["boxm"] = box_m(X, y)
    r["mardia"] = {k: mardia(X[y == k]) for k in [0, 1]}
    r["corr"] = df[X3].corr()
    r["vif"] = pd.Series(vif(X), index=X3)
    fit = lda_fit(X, y)
    r["fit"] = fit
    a = fit["a"]
    sd = np.sqrt(np.diag(fit["Sp"]))
    r["coef_bruto"] = pd.Series(a, index=X3)
    std = a * sd
    r["coef_pad"] = pd.Series(std, index=X3)
    # ajuste à escala SAS/SPSS: normalizar para variância intra-grupo unitária da função
    var_z = a @ fit["Sp"] @ a
    r["coef_pad_norm"] = pd.Series(std / np.sqrt(var_z), index=X3)
    r["coef_bruto_norm"] = pd.Series(a / np.sqrt(var_z), index=X3)
    z = X @ (a / np.sqrt(var_z))
    r["const_norm"] = -X.mean(0) @ (a / np.sqrt(var_z))
    # cargas estruturais (correlações intra-grupo entre variáveis e função)
    Sp = fit["Sp"]
    Rw = Sp / np.outer(sd, sd)
    r["cargas"] = pd.Series(Rw @ (std / np.sqrt(var_z)), index=X3)
    # estatísticas da função
    n0, n1 = fit["n0"], fit["n1"]; N = n0 + n1
    T = np.cov(X.T) * (N - 1); W = Sp * (N - 2)
    lam = np.linalg.det(W) / np.linalg.det(T)
    p = len(X3)
    chi = -(N - 1 - (p + 2) / 2) * np.log(lam)
    r["wilks"] = dict(lam=lam, chi2=chi, gl=p, p=stats.chi2.sf(chi, p))
    D2 = (fit["m1"] - fit["m0"]) @ a
    r["D2"] = D2
    Fh = (n0 * n1 / N) * (N - p - 1) / (p * (N - 2)) * D2
    r["hotelling"] = dict(T2=n0 * n1 / N * D2, F=Fh, gl1=p, gl2=N - p - 1, p=stats.f.sf(Fh, p, N - p - 1))
    autoval = (1 - lam) / lam
    r["autovalor"] = autoval
    r["rcan"] = np.sqrt(autoval / (1 + autoval))
    r["centroides"] = {k: z[y == k].mean() for k in [0, 1]}
    r["cargas_total"] = pd.Series([np.corrcoef(X[:, j], z)[0, 1] for j in range(len(X3))], index=X3)
    r["z"] = z
    # classificação: resubstituição e leave-one-out (Lachenbruch)
    for nome, prior in [("iguais", None), ("proporcionais", n1 / N)]:
        pred, _, corte = lda_classifica(fit, X, prior)
        loo = np.empty(N, int)
        for i in range(N):
            msk = np.ones(N, bool); msk[i] = False
            f_i = lda_fit(X[msk], y[msk])
            pr = None if prior is None else y[msk].mean()
            loo[i] = lda_classifica(f_i, X[i:i + 1], pr)[0][0]
        r[f"cls_{nome}"] = dict(corte=corte, conf=pd.crosstab(y, pred), conf_loo=pd.crosstab(y, loo),
                                aper=(pred != y).mean(), loo=(loo != y).mean())
    # holdout 70/30 estratificado
    rng = np.random.default_rng(2024)
    idx_tr = np.concatenate([rng.choice(np.where(y == k)[0], int(round(0.7 * np.sum(y == k))), replace=False) for k in [0, 1]])
    mtr = np.zeros(N, bool); mtr[idx_tr] = True
    f_tr = lda_fit(X[mtr], y[mtr])
    pred_te = lda_classifica(f_tr, X[~mtr])[0]
    r["holdout"] = dict(n_tr=mtr.sum(), n_te=(~mtr).sum(), conf=pd.crosstab(y[~mtr], pred_te), erro=(pred_te != y[~mtr]).mean())
    # critério de chance proporcional
    p1 = n1 / N
    r["cpro"] = p1**2 + (1 - p1)**2
    r["cmax"] = max(p1, 1 - p1)
    # Press's Q
    acc = 1 - r["cls_iguais"]["loo"]
    r["pressQ"] = (N - acc * N * 2)**2 / (N * (2 - 1))
    return r

# ---------------------------------------------------------------- Parte IV (MDS)
def mds_classico(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    ev, V = np.linalg.eigh(B)
    idx = np.argsort(ev)[::-1]
    return ev[idx], V[:, idx]

def parte4(df):
    r = {}
    Z = (df[X3] - df[X3].mean()) / df[X3].std(ddof=1)
    D = squareform(pdist(Z.values, "euclidean"))
    r["D"] = D
    ev, V = mds_classico(D)
    r["ev"] = ev
    pos = ev[ev > 1e-9]
    r["var_expl"] = pos / pos.sum()
    coords = V[:, :2] * np.sqrt(ev[:2])
    r["coords"] = coords
    d_orig = pdist(Z.values); 
    stress_dim = []
    for k in range(1, 6):
        C = V[:, :k] * np.sqrt(ev[:k])
        dk = pdist(C)
        s1 = np.sqrt(((d_orig - dk)**2).sum() / (d_orig**2).sum())
        rsq = np.corrcoef(d_orig, dk)[0, 1]**2
        stress_dim.append(dict(dim=k, stress=s1, rsq=rsq, var_acum=pos[:k].sum() / pos.sum()))
    r["stress_dim"] = pd.DataFrame(stress_dim)
    r["d_orig"] = d_orig; r["d_2d"] = pdist(coords)
    # correlação das dimensões com as variáveis originais
    r["corr_dim"] = pd.DataFrame({f"Dim {k+1}": [np.corrcoef(coords[:, k], df[v])[0, 1] for v in X3] for k in range(2)}, index=X3)
    # médias das coordenadas por grupo
    tmp = pd.DataFrame(coords, columns=["D1", "D2"])
    tmp["Status"] = df.Status_Fidelidade.values; tmp["Regiao"] = df.Regiao_nome.values
    r["centro_status"] = tmp.groupby("Status")[["D1", "D2"]].mean()
    r["centro_regiao"] = tmp.groupby("Regiao")[["D1", "D2"]].mean().loc[["Norte", "Sul", "Centro"]]
    # razão distância média entre grupos / dentro de grupos no mapa
    s = df.Status_Fidelidade.values
    D2m = squareform(r["d_2d"])
    same = s[:, None] == s[None, :]
    iu = np.triu_indices(len(s), 1)
    # MDS não métrico (Kruskal) partindo da solução clássica
    from sklearn.manifold import MDS
    from sklearn.isotonic import IsotonicRegression
    nm = MDS(n_components=2, metric=False, dissimilarity="precomputed", n_init=1, max_iter=500, random_state=1)
    C_nm = nm.fit_transform(D, init=coords)
    d_nm = pdist(C_nm)
    dhat = IsotonicRegression().fit_transform(d_orig, d_nm)
    r["nm_stress"] = np.sqrt(((d_nm - dhat)**2).sum() / (d_nm**2).sum())
    r["nm_rsq"] = np.corrcoef(dhat, d_nm)[0, 1]**2
    r["nm_coords"] = C_nm
    r["dist_intra"] = D2m[iu][same[iu]].mean(); r["dist_inter"] = D2m[iu][~same[iu]].mean()
    return r

# ---------------------------------------------------------------- Parte V (CCA)
B1 = ["Freq_Online", "Freq_Loja", "Tempo_Relacao"]
B2 = ["Gasto_Online", "Gasto_Loja", "NPS_Pontuacao"]
def parte5(df):
    r = {}
    X = df[B1].values.astype(float); Y = df[B2].values.astype(float)
    n, p, q = len(df), X.shape[1], Y.shape[1]
    R = np.corrcoef(np.hstack([X, Y]).T)
    Rxx, Ryy, Rxy = R[:p, :p], R[p:, p:], R[:p, p:]
    r["R"] = pd.DataFrame(R, index=B1 + B2, columns=B1 + B2)
    M = np.linalg.solve(Rxx, Rxy) @ np.linalg.solve(Ryy, Rxy.T)
    ev, A = np.linalg.eig(M)
    idx = np.argsort(np.real(ev))[::-1]; ev = np.real(ev[idx]); A = np.real(A[:, idx])
    s = min(p, q)
    ev = ev[:s]; A = A[:, :s]
    rc = np.sqrt(np.clip(ev, 0, 1))
    # normalizar: variância das variáveis canônicas = 1
    A = A / np.sqrt(np.diag(A.T @ Rxx @ A))
    Bm = np.linalg.solve(Ryy, Rxy.T) @ A / rc
    Bm = Bm / np.sqrt(np.diag(Bm.T @ Ryy @ Bm))
    # sinais: carga dominante positiva
    for k in range(s):
        if (Rxx @ A[:, k]).sum() < 0:
            A[:, k] *= -1
        if np.corrcoef(Rxy.T @ A[:, k] if False else (Ryy @ Bm[:, k]), Rxy.T @ A[:, k])[0, 1] < 0 if s > 1 else False:
            pass
        # garantir correlação positiva entre U_k e V_k
        if A[:, k] @ Rxy @ Bm[:, k] < 0:
            Bm[:, k] *= -1
    r["rc"] = rc; r["rc2"] = rc**2
    r["A"] = pd.DataFrame(A, index=B1, columns=[f"U{k+1}" for k in range(s)])
    r["B"] = pd.DataFrame(Bm, index=B2, columns=[f"V{k+1}" for k in range(s)])
    LX = Rxx @ A; LY = Ryy @ Bm
    CX = Rxy @ Bm; CY = Rxy.T @ A
    r["LX"] = pd.DataFrame(LX, index=B1, columns=r["A"].columns)
    r["LY"] = pd.DataFrame(LY, index=B2, columns=r["B"].columns)
    r["CX"] = pd.DataFrame(CX, index=B1, columns=r["B"].columns)
    r["CY"] = pd.DataFrame(CY, index=B2, columns=r["A"].columns)
    # redundância
    vx = (LX**2).mean(0); vy = (LY**2).mean(0)
    r["red"] = pd.DataFrame(dict(var_prop_X=vx, red_X=vx * rc**2, var_prop_Y=vy, red_Y=vy * rc**2), index=[f"Par {k+1}" for k in range(s)])
    # testes sequenciais de Bartlett (Wilks) e aproximação F de Rao
    lin = []
    for k in range(s):
        lam = np.prod(1 - rc[k:]**2)
        pk, qk = p - k, q - k
        chi = -(n - 1 - (p + q + 1) / 2) * np.log(lam)
        gl = pk * qk
        # Rao
        if pk**2 + qk**2 - 5 > 0:
            t = np.sqrt((pk**2 * qk**2 - 4) / (pk**2 + qk**2 - 5))
        else:
            t = 1
        w = n - 1.5 - (p + q) / 2 - k + k  # conforme Rencher: n - (p+q+3)/2
        w = n - (p + q + 3) / 2
        df1 = pk * qk; df2 = w * t - pk * qk / 2 + 1
        F = (1 - lam**(1 / t)) / lam**(1 / t) * df2 / df1
        lin.append(dict(dim=k + 1, rc=rc[k], rc2=rc[k]**2, lambda_wilks=lam, chi2=chi, gl=gl, p_chi=stats.chi2.sf(chi, gl),
                        F=F, gl1=df1, gl2=df2, p_F=stats.f.sf(F, df1, df2)))
    r["testes"] = pd.DataFrame(lin)
    # critérios globais (todas as dimensões)
    l2 = rc**2
    r["global"] = dict(wilks=np.prod(1 - l2), pillai=l2.sum(), hotelling=(l2 / (1 - l2)).sum(), roy=l2[0])
    r["mardia"] = mardia(np.hstack([X, Y]))
    r["vif"] = pd.Series(np.concatenate([vif(X), vif(Y)]), index=B1 + B2)
    r["U"] = ((X - X.mean(0)) / X.std(0, ddof=1)) @ A
    r["V"] = ((Y - Y.mean(0)) / Y.std(0, ddof=1)) @ Bm
    # linearidade: correlação de Pearson vs Spearman entre blocos
    r["spearman"] = df[B1 + B2].corr("spearman").loc[B1, B2]
    return r

def tudo():
    df = carregar()
    return df, parte1(df), parte2(df), parte3(df), parte4(df), parte5(df)

if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    df, r1, r2, r3, r4, r5 = tudo()
    def show(r, skip=()):
        for k, v in r.items():
            if k in skip: continue
            print(f"--- {k}"); print(v)
    print("=========== P1"); show(r1, ["resid"])
    print("=========== P2"); show(r2)
    print("=========== P3"); show(r3, ["z", "fit"])
    print("=========== P4"); show(r4, ["D", "coords", "d_orig", "d_2d", "ev"]); print(r4["ev"][:6])
    print("=========== P5"); show(r5, ["U", "V"])
