"""Gera o relatório técnico (HTML -> PDF) do trabalho de Análise Multivariada."""
import io, base64, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import calculos as C

AQUI = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COR_REG = {"Norte": BLUE, "Sul": ORANGE, "Centro": AQUA}
COR_ST = {0: BLUE, 1: ORANGE}
ROT_ST = {0: "Inativo/Baixo Risco (0)", 1: "Ativo/Alto Valor (1)"}
ORD = ["Norte", "Sul", "Centro"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#bdbcb6",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def fig_html(fig, legenda, largura="100%"):
    return (f'<figure><img src="data:image/png;base64,{b64(fig)}" style="width:{largura}"/>'
            f'<figcaption>{legenda}</figcaption></figure>')

def fp(p):
    if p < 0.0001:
        return "&lt; 0,0001"
    return f"{p:.4f}".replace(".", ",")

def fn(x, d=3):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def tabela(linhas, cab, titulo=None, alinhar_esq=1):
    h = ""
    if titulo:
        h += f'<p class="tabtit">{titulo}</p>'
    h += '<table><thead><tr>' + "".join(f"<th>{c}</th>" for c in cab) + "</tr></thead><tbody>"
    for l in linhas:
        h += "<tr>" + "".join(f'<td class="{"l" if i < alinhar_esq else "r"}">{v}</td>' for i, v in enumerate(l)) + "</tr>"
    return h + "</tbody></table>"

NOMES = {"Gasto_Online": "Gasto Online", "Gasto_Loja": "Gasto Loja", "Freq_Online": "Freq. Online",
         "Freq_Loja": "Freq. Loja", "Tempo_Relacao": "Tempo de Relação", "NPS_Pontuacao": "NPS",
         "Score_Satisfacao": "Score de Satisfação"}

df, r1, r2, r3, r4, r5 = C.tudo()
N = len(df)
H = []  # partes do corpo

# =============================================================== 0. DADOS
cont = ["Gasto_Online", "Gasto_Loja", "Freq_Online", "Freq_Loja", "Score_Satisfacao", "Tempo_Relacao", "NPS_Pontuacao"]
desc = df[cont].describe().T
lin = [[NOMES[v], fn(desc.loc[v, "mean"], 2), fn(desc.loc[v, "std"], 2), fn(desc.loc[v, "min"], 2),
        fn(desc.loc[v, "50%"], 2), fn(desc.loc[v, "max"], 2), fn(stats.skew(df[v]), 2), fn(stats.kurtosis(df[v]), 2)] for v in cont]
reg_n = df.Regiao_nome.value_counts()
st_n = df.Status_Fidelidade.value_counts()
ct = pd.crosstab(df.Regiao_nome, df.Status_Fidelidade).loc[ORD]
chi_ct = stats.chi2_contingency(ct)

H.append(f"""
<h1>1. Base de dados e visão geral</h1>
<p>A base contém <b>{N} clientes</b> e foi construída por simulação a partir das regras e distribuições descritas no
enunciado. Como foi utilizado um gerador pseudoaleatório diferente do SAS, os valores individuais não coincidem
necessariamente com os do arquivo original. Portanto, este relatório deve ser entendido como uma <b>reprodução
independente do procedimento</b>, e não como uma cópia da base SAS; resultados numéricos e conclusões podem variar.
A distribuição observada foi: Norte = {reg_n['Norte']}, Sul = {reg_n['Sul']} e Centro = {reg_n['Centro']};
Status 0 = {st_n[0]} ({st_n[0]/N:.0%}) e Status 1 = {st_n[1]} ({st_n[1]/N:.0%}). Região e status não apresentaram associação
estatisticamente significativa (χ² = {fn(chi_ct[0],2)}; gl = {chi_ct[2]}; p = {fp(chi_ct[1])}). Não há valores ausentes.</p>
{tabela(lin, ["Variável", "Média", "Desvio-padrão", "Mínimo", "Mediana", "Máximo", "Assimetria", "Curtose (exc.)"],
        "Tabela 1 – Estatísticas descritivas das variáveis contínuas (n = 300)")}
""")

R = df[cont].corr()
fig, ax = plt.subplots(figsize=(6.4, 4.6))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
im = ax.imshow(R.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(cont))); ax.set_yticks(range(len(cont)))
ax.set_xticklabels([NOMES[c] for c in cont], rotation=40, ha="right"); ax.set_yticklabels([NOMES[c] for c in cont])
ax.grid(False)
for i in range(len(cont)):
    for j in range(len(cont)):
        ax.text(j, i, f"{R.values[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8,
                color="white" if abs(R.values[i, j]) > 0.6 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.outline.set_visible(False)
ax.set_title("Correlações de Pearson")
H.append(fig_html(fig, "Figura 1 – Matriz de correlações. Destacam-se Freq. Loja × Gasto Loja (0,98), Freq. Online × Gasto Online (0,80) e Gasto Online × NPS (0,46).", "78%"))

# =============================================================== PARTE I
a = r1["anova"]
sh = r1["shapiro_grupo"]
lin_d = [[g, int(r1["desc"].loc[g, "count"]), fn(r1["desc"].loc[g, "mean"], 3), fn(r1["desc"].loc[g, "std"], 3),
          fn(r1["desc"].loc[g, "median"], 2), fn(r1["skew_kurt"][g][0], 2), fn(r1["skew_kurt"][g][1], 2),
          fn(sh[g].statistic, 4), fp(sh[g].pvalue)] for g in ORD]
lin_a = [["Entre regiões (Regiao)", int(a["df"].iloc[0]), fn(a["sum_sq"].iloc[0], 3), fn(a["mean_sq"].iloc[0], 3), fn(a["F"].iloc[0], 3), fp(a["PR(>F)"].iloc[0])],
         ["Dentro das regiões (resíduo)", int(a["df"].iloc[1]), fn(a["sum_sq"].iloc[1], 3), fn(a["mean_sq"].iloc[1], 3), "", ""],
         ["Total", int(a["df"].sum()), fn(a["sum_sq"].sum(), 3), "", "", ""]]
tk = r1["tukey"].copy()
mp = {("Centro", "Norte"): ("Norte", "Centro"), ("Centro", "Sul"): ("Sul", "Centro"), ("Norte", "Sul"): ("Norte", "Sul")}
lin_t = []
for _, t in tk.iterrows():
    g1, g2 = mp[(t.group1, t.group2)]
    sinal = 1 if (t.group1, t.group2) == (g1, g2) else -1
    d, lo, up = sinal * t.meandiff, sinal * (t.lower if sinal == 1 else t.upper), sinal * (t.upper if sinal == 1 else t.lower)
    bf = r1["bonf"].set_index(["grupo1", "grupo2"]).loc[(g1, g2)]
    lin_t.append([f"{g2} − {g1}", fn(d, 3), f"[{fn(lo,3)}; {fn(up,3)}]", fp(t["p-adj"]) if t["p-adj"] >= 1e-4 else "&lt; 0,0001", fp(bf.p_bonf), "Sim" if t.reject else "Não"])
lin_t.sort(key=lambda l: ["Sul − Norte", "Centro − Norte", "Centro − Sul"].index(l[0]))

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
for i, g in enumerate(ORD):
    y = df.loc[df.Regiao_nome == g, "Score_Satisfacao"].values
    bp = ax.boxplot(y, positions=[i], widths=0.45, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5), boxprops=dict(facecolor=COR_REG[g] + "40", edgecolor=COR_REG[g], lw=1.2),
                    whiskerprops=dict(color=COR_REG[g]), capprops=dict(color=COR_REG[g]))
    jit = np.random.default_rng(i).uniform(-0.17, 0.17, len(y))
    ax.scatter(i + jit, y, s=8, color=COR_REG[g], alpha=0.45, lw=0)
    m = y.mean(); se = y.std(ddof=1) / np.sqrt(len(y))
    ax.errorbar(i + 0.33, m, yerr=1.96 * se, fmt="o", color=INK, ms=4, capsize=3, lw=1.2)
    ax.text(i + 0.4, m, f" {m:.2f}".replace(".", ","), va="center", fontsize=8.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(ORD); ax.set_ylabel("Score de Satisfação")
ax.set_title("Distribuição por região (ponto preto = média ± IC 95%)")
ax = axs[1]
res = np.sort(r1["resid"].values)
q = stats.norm.ppf((np.arange(1, len(res) + 1) - 0.5) / len(res))
ax.scatter(q, res, s=9, color=BLUE, alpha=0.6, lw=0)
ax.plot(q, q * res.std(ddof=1) + res.mean(), color=INK2, lw=1.2)
ax.set_xlabel("Quantis teóricos da Normal"); ax.set_ylabel("Resíduos do modelo")
ax.set_title("Q-Q plot dos resíduos")
fig.tight_layout()
fig1 = fig_html(fig, "Figura 2 – Score de Satisfação por região e diagnóstico de normalidade dos resíduos da ANOVA.")

w = r1["welch"]
H.append(f"""
<h1>2. Parte I – Comparação do Score de Satisfação entre regiões</h1>
<h2>2.1 Metodologia</h2>
<p>A variável resposta (Score de Satisfação) é <b>quantitativa contínua</b> e o fator explicativo (Região) é
<b>categórico com três níveis independentes</b> (cada cliente pertence a uma única região). O objetivo é testar
simultaneamente H<sub>0</sub>: μ<sub>Norte</sub> = μ<sub>Sul</sub> = μ<sub>Centro</sub>. A ferramenta adequada é a
<b>Análise de Variância (ANOVA) de um fator</b>, que decompõe a variabilidade total em uma parcela atribuída às
diferenças entre as médias das regiões e outra atribuída à variação dentro das regiões, comparando-as pela
estatística F. Realizar três testes t em pares não é adequado, pois inflaria a taxa de erro tipo I global
(1 − 0,95³ ≈ 14%). Havendo rejeição de H<sub>0</sub>, aplica-se um procedimento de <b>comparações múltiplas
post hoc</b> (Tukey HSD, apropriado para todas as comparações par a par com controle do erro global; Bonferroni
como confirmação) para identificar quais regiões diferem.</p>

<h2>2.2 Validação dos pressupostos</h2>
<ul>
<li><b>Independência das observações:</b> garantida pelo desenho – cada cliente é observado uma vez e pertence a uma só região.</li>
<li><b>Normalidade:</b> o teste de Shapiro-Wilk sobre os <b>resíduos</b> do modelo (que é o pressuposto formal) não rejeita a
normalidade (W = {fn(r1['shapiro_resid'].statistic,4)}; p = {fp(r1['shapiro_resid'].pvalue)}), resultado confirmado por Kolmogorov-Smirnov
(D = {fn(r1['ks_resid'].statistic,4)}; p = {fp(r1['ks_resid'].pvalue)}) e pelo Q-Q plot (Figura 2). Por grupo, Norte (p = {fp(sh['Norte'].pvalue)})
e Centro (p = {fp(sh['Centro'].pvalue)}) apresentam desvios marginais, explicados pelo truncamento da nota em 1 e 10
(efeito teto no Centro). Assimetrias e curtoses estão entre −0,6 e 0,6 e os grupos têm n ≥ 95, de modo que a ANOVA
é robusta (Teorema Central do Limite).</li>
<li><b>Homogeneidade das variâncias:</b> Levene (centrado na média) F = {fn(r1['levene'].statistic,3)}, p = {fp(r1['levene'].pvalue)};
Brown-Forsythe (mediana) F = {fn(r1['brown_forsythe'].statistic,3)}, p = {fp(r1['brown_forsythe'].pvalue)}; Bartlett χ² = {fn(r1['bartlett'].statistic,3)},
p = {fp(r1['bartlett'].pvalue)}. Nenhum rejeita a igualdade das variâncias (desvios-padrão entre 0,99 e 1,06).</li>
</ul>
{tabela(lin_d, ["Região", "n", "Média", "Desvio-padrão", "Mediana", "Assimetria", "Curtose", "Shapiro W", "p-valor"],
        "Tabela 2 – Estatísticas descritivas do Score de Satisfação por região")}

<h2>2.3 Resultados</h2>
{tabela(lin_a, ["Fonte de variação", "gl", "Soma de quadrados", "Quadrado médio", "F", "p-valor"], "Tabela 3 – Quadro da ANOVA de um fator")}
<p>A estatística F = {fn(a['F'].iloc[0],2)} (gl = 2; 297) é altamente significativa (p {fp(a['PR(>F)'].iloc[0])}): rejeita-se H<sub>0</sub>.
O tamanho de efeito é grande: η² = {fn(r1['eta2'],3)} e ω² = {fn(r1['omega2'],3)}, isto é, cerca de <b>24% da variabilidade da satisfação é
explicada pela região</b>. Como verificação de robustez, a ANOVA de Welch (que não exige variâncias iguais) fornece
F = {fn(w['F'],2)} (gl = 2; {fn(w['gl2'],1)}), p {fp(w['p'])}, e o teste não paramétrico de Kruskal-Wallis fornece
H = {fn(r1['kruskal'].statistic,2)}, p {fp(r1['kruskal'].pvalue)} – a conclusão é a mesma por todos os caminhos.</p>
{tabela(lin_t, ["Comparação", "Diferença de médias", "IC 95% (Tukey)", "p ajustado Tukey", "p ajustado Bonferroni", "Difere?"],
        f"Tabela 4 – Comparações múltiplas post hoc (QME = {fn(r1['msw'],4)}, gl = 297)")}
{fig1}

<h2>2.4 Interpretação negocial</h2>
<p>As três regiões diferem entre si, com ordenação nítida: <b>Norte ({fn(r1['desc'].loc['Norte','mean'],2)}) &lt; Sul ({fn(r1['desc'].loc['Sul','mean'],2)})
&lt; Centro ({fn(r1['desc'].loc['Centro','mean'],2)})</b>. A distância entre o Norte e o Centro é de aproximadamente 1,4 ponto em uma
escala de 10 – uma diferença substantiva, não apenas estatística. Recomenda-se: (i) priorizar o Norte em ações de
melhoria de experiência (atendimento, prazos de entrega, sortimento), pois é a região com maior espaço de ganho;
(ii) estudar as práticas operacionais do Centro como referência interna (benchmark); (iii) no Centro, como parte dos
clientes já atinge a nota máxima (efeito teto), a métrica de satisfação perde sensibilidade e deve ser complementada
por indicadores como NPS e recompra; (iv) metas de satisfação devem ser regionalizadas, já que uma meta única
penalizaria o Norte e seria pouco desafiadora para o Centro.</p>
""")

# =============================================================== PARTE II
DV = C.DV2
mv = r2["manova"]
nomes_mv = {"Wilks' lambda": "Lambda de Wilks", "Pillai's trace": "Traço de Pillai",
            "Hotelling-Lawley trace": "Traço de Hotelling-Lawley", "Roy's greatest root": "Maior raiz de Roy"}
lin_mv = [[nomes_mv[k], fn(float(mv.loc[k, "Value"]), 4), fn(float(mv.loc[k, "F Value"]), 3), fn(float(mv.loc[k, "Num DF"]), 0),
           fn(float(mv.loc[k, "Den DF"]), 1), fp(float(mv.loc[k, "Pr > F"]))] for k in nomes_mv]
lin_d2 = []
for g in ORD:
    lin_d2.append([g] + [f"{fn(r2['desc'].loc[g, (v, 'mean')], 2)} ({fn(r2['desc'].loc[g, (v, 'std')], 2)})" for v in DV])
bm = r2["boxm"]
lin_pres = [[f"Mardia – assimetria ({g})", fn(r2['mardia_grupo'][g]['chi2'], 2), fp(r2['mardia_grupo'][g]['p_ass'])] for g in ORD]
lin_pres += [[f"Mardia – curtose ({g})", fn(r2['mardia_grupo'][g]['z'], 2), fp(r2['mardia_grupo'][g]['p_curt'])] for g in ORD]
lin_pres += [["M de Box (homog. das matrizes de covariâncias)", fn(bm['chi2'], 2) + f" (gl = {int(bm['gl'])})", fp(bm['p'])],
             ["Esfericidade de Bartlett (correlação entre as VDs)", fn(r2['bartlett_esf']['chi2'], 2) + " (gl = 3)", fp(r2['bartlett_esf']['p'])]]
lin_pres += [[f"Levene – {NOMES[v]}", fn(r2['levene'][v].statistic, 3), fp(r2['levene'][v].pvalue)] for v in DV]
u = r2["univ"]
lin_u = [[NOMES[r.variavel], fn(r.F, 3), fp(r.p), fp(r.p_bonf), fn(r.eta2, 3)] for r in u.itertuples()]
lin_tk2 = []
for v in DV:
    t = r2["tukey"][v].set_index(["group1", "group2"])
    def gp(a, b):
        row = t.loc[(a, b)] if (a, b) in t.index else t.loc[(b, a)]
        return "✓" if row.reject else "–"
    lin_tk2.append([NOMES[v], gp("Norte", "Sul"), gp("Norte", "Centro"), gp("Sul", "Centro")])
ds = r2["disc_std"]
def sscp(M, titulo):
    return tabela([[NOMES[a]] + [fn(M[i, j], 2) for j in range(3)] for i, a in enumerate(DV)], [""] + [NOMES[v] for v in DV], titulo)
tab_H = sscp(r2["H"], "Tabela 7a – Matriz H de somas de quadrados e produtos cruzados da hipótese (efeito Região)")
tab_E = sscp(r2["E"], "Tabela 7b – Matriz E de somas de quadrados e produtos cruzados do erro")
pe, pp = r2["parcial_E"], r2["parcial_E_p"]
tab_pE = tabela([[NOMES[a]] + [("1,000" if a == b else f"{fn(pe.loc[a, b], 3)} (p {fp(pp.loc[a, b]) if pp.loc[a, b] < 1e-4 else '= ' + fp(pp.loc[a, b])})") for b in DV] for a in DV],
                [""] + [NOMES[v] for v in DV], "Tabela 7c – Correlações parciais obtidas da matriz E (gl = 297)")
rz, vt = r2["raizes"], r2["vetores"]
lin_rz = []
for k in range(3):
    lin_rz.append([f"{k+1}", fn(max(rz.raiz[k], 0), 6), fn(max(rz.pct[k], 0), 2) + "%"] + [f"{vt.iloc[j, k]:.6f}".replace(".", ",") for j in range(3)]
                  + ([fn(ds.iloc[j, k], 3) for j in range(3)] if k < 2 else ["–"] * 3))
tab_rz = tabela(lin_rz, ["Raiz", "Valor característico", "Percentual"] + [f"Vetor: {NOMES[v]}" for v in DV] + [f"Padr.: {NOMES[v]}" for v in DV],
                "Tabela 7d – Raízes e vetores característicos de E<sup>−1</sup>H (vetores normalizados por v'Ev = 1; à direita, coeficientes padronizados relativos)")
ls = r2["lsmeans"].set_index(["Regiao", "variavel"])
lin_ls = [[g] + [f"{fn(ls.loc[(g, v), 'lsmean'], 2)} ± {fn(ls.loc[(g, v), 'ep'], 2)}" for v in DV] for g in ORD]
tk_p = {}
for v in DV:
    t = r2["tukey"][v]
    for _, row in t.iterrows():
        tk_p[(v, frozenset([row.group1, row.group2]))] = row["p-adj"]
lin_pd = []
for row in r2["pdiff"].itertuples():
    a, b = row.par.split(" − ")
    pt = tk_p[(row.variavel, frozenset([a, b]))]
    lin_pd.append([NOMES[row.variavel], row.par, fn(row.dif, 2), fn(row.t, 3), fp(row.p), fp(pt) if pt >= 1e-4 else "&lt; 0,0001", "Sim" if pt < 0.05 else "Não"])

fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
for ax, v in zip(axs, DV):
    for i, g in enumerate(ORD):
        y = df.loc[df.Regiao_nome == g, v]
        m, se = y.mean(), y.std(ddof=1) / np.sqrt(len(y))
        ax.errorbar(i, m, yerr=1.96 * se, fmt="o", color=COR_REG[g], ms=7, capsize=4, lw=1.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(ORD); ax.set_xlim(-0.5, 2.5)
    ax.set_title(NOMES[v] + (" (R$)" if "Gasto" in v else " (0–100)"))
fig.tight_layout()
fig2 = fig_html(fig, "Figura 3 – Médias por região e intervalos de confiança de 95% para as três variáveis dependentes.")

H.append(f"""
<h1>3. Parte II – Efeito conjunto da Região sobre Gasto Online, Gasto Loja e NPS</h1>
<h2>3.1 Metodologia</h2>
<p>Agora há <b>três variáveis dependentes quantitativas medidas no mesmo cliente</b> e, portanto, correlacionadas
(Gasto Online × NPS: r = {fn(r2['corr'].loc['Gasto_Online','NPS_Pontuacao'],2)}; Gasto Online × Gasto Loja: r = {fn(r2['corr'].loc['Gasto_Online','Gasto_Loja'],2)};
esfericidade de Bartlett p {fp(r2['bartlett_esf']['p'])}). O fator continua sendo a Região (3 grupos independentes). O método
adequado é a <b>Análise de Variância Multivariada (MANOVA)</b>, que testa H<sub>0</sub>: <b>μ</b><sub>Norte</sub> = <b>μ</b><sub>Sul</sub> =
<b>μ</b><sub>Centro</sub>, em que cada <b>μ</b> é o vetor de médias (Gasto Online, Gasto Loja, NPS). A MANOVA compara a matriz de
somas de quadrados e produtos cruzados entre grupos (<b>H</b>) com a matriz dentro dos grupos (<b>E</b>), considerando
variâncias e covariâncias simultaneamente, e controla o erro tipo I global em um único teste. Os quatro critérios
clássicos são funções dos autovalores de <b>E</b><sup>−1</sup><b>H</b> (λ<sub>1</sub> = {fn(r2['eig'][0],4)}, λ<sub>2</sub> = {fn(r2['eig'][1],4)}).</p>

<h2>3.2 Validação dos pressupostos</h2>
<ul>
<li><b>Independência:</b> garantida pelo desenho amostral; <b>tamanho amostral</b>: todos os grupos têm n ≥ 95, muito acima do
número de variáveis dependentes (3).</li>
<li><b>Normalidade multivariada:</b> os testes de Mardia por grupo não rejeitam a normalidade (Tabela 6). Nos resíduos
combinados, a curtose multivariada é levemente inferior à esperada (b<sub>2</sub> = {fn(r2['mardia_resid']['b2'],2)} vs. 15; p = {fp(r2['mardia_resid']['p_curt'])}),
efeito do truncamento do NPS em 100 (distribuição achatada, platicúrtica), que não compromete o teste dado o tamanho amostral.</li>
<li><b>Homogeneidade das matrizes de covariâncias:</b> o teste M de Box não rejeita H<sub>0</sub> (p = {fp(bm['p'])}); os testes de Levene
univariados também não rejeitam.</li>
<li><b>Linearidade e ausência de multicolinearidade entre as VDs:</b> as correlações são moderadas (máx. 0,46), indicando
variáveis relacionadas, mas não redundantes – situação ideal para a MANOVA.</li>
</ul>
{tabela(lin_d2, ["Região", "Gasto Online (R$)", "Gasto Loja (R$)", "NPS"], "Tabela 5 – Médias (desvios-padrão) das variáveis dependentes por região")}
{tabela(lin_pres, ["Teste de pressuposto", "Estatística", "p-valor"], "Tabela 6 – Testes de pressupostos da MANOVA")}

<h2>3.3 Resultados</h2>
<p>Seguindo a saída do procedimento de modelos lineares gerais com a opção MANOVA (impressão das matrizes E e H), apresentam-se
primeiro as matrizes de somas de quadrados e produtos cruzados. A diagonal de <b>H</b> contém as somas de quadrados entre regiões de
cada variável (as mesmas das ANOVAs univariadas) e a diagonal de <b>E</b>, as somas de quadrados dentro das regiões; fora da diagonal
estão os produtos cruzados, que carregam a informação de covariância ignorada pelas análises univariadas.</p>
{tab_H}
{tab_E}
<p>As correlações parciais da matriz E medem a associação entre as variáveis dependentes <i>depois de remover o efeito da Região</i>.
Todas são significativas – em especial Gasto Online × NPS (0,43) –, o que confirma que as variáveis dependentes não são
independentes entre si e justifica a análise conjunta.</p>
{tab_pE}
<p>Os critérios multivariados são funções das raízes características de <b>E</b><sup>−1</sup><b>H</b>. Como há 3 regiões, existem no máximo
s = min(3 variáveis; 2 gl da hipótese) = 2 raízes não nulas.</p>
{tab_rz}
{tabela(lin_mv, ["Critério multivariado", "Valor", "F aproximado", "gl num.", "gl den.", "p-valor"], "Tabela 7 – Testes multivariados para a hipótese de ausência de efeito global da Região (S = 2; M = 0; N = 146,5)")}
<p>Todos os critérios rejeitam H<sub>0</sub> (p {fp(float(mv.loc["Wilks' lambda", "Pr > F"]))}): <b>o vetor de médias difere entre as regiões</b>.
O Lambda de Wilks ({fn(float(mv.loc["Wilks' lambda", "Value"]),3)}) confirma a diferença multivariada. Seu complemento
(1 − Λ = {fn(1-float(mv.loc["Wilks' lambda", "Value"]),3)}) é apenas uma transformação descritiva do Lambda e não deve ser interpretado
diretamente como percentual de variância explicado ou como η² multivariado. Para discutir magnitude, utilizam-se os
tamanhos de efeito das ANOVAs univariadas (Tabela 8), que mostram efeitos distintos para cada variável. Como o M de Box não foi
significativo, Wilks é o critério de referência; o Traço de Pillai, o mais robusto a violações, confirma o resultado.
A maior raiz de Roy, também significativa, indica que a maior parte da diferença se concentra em uma dimensão.
Os vetores característicos (Tabela 7d) mostram que a primeira raiz (72,7% da separação) é dominada por <b>Gasto Online</b>
e a segunda (27,3%) por <b>NPS</b>; como os vetores brutos dependem das unidades de medida (R$ × pontos), a leitura é feita pelos
coeficientes padronizados à direita da tabela.</p>
{tabela(lin_u, ["Variável dependente", "F (2; 297)", "p-valor", "p Bonferroni (×3)", "η²"], "Tabela 8 – ANOVAs univariadas de acompanhamento")}
{tabela(lin_ls, [ "Região"] + [NOMES[v] for v in DV], "Tabela 9 – Médias de mínimos quadrados (± erro-padrão) por região")}
<p>As médias de mínimos quadrados coincidem com as médias amostrais, pois o modelo tem um único fator. A comparação par a par
(diferença das médias de mínimos quadrados, teste t com o quadrado médio do erro) é apresentada sem ajuste e com o ajuste de
Tukey para múltiplas comparações. Sem ajuste, o par Sul × Centro em Gasto Loja aparece significativo (p = 0,039), mas deixa de
sê-lo com Tukey (p = 0,097): ilustra-se o risco de inflar o erro tipo I ao fazer várias comparações.</p>
{tabela(lin_pd, ["Variável", "Comparação", "Diferença", "t", "p (sem ajuste)", "p (Tukey)", "Difere (Tukey)?"], "Tabela 10 – Comparações par a par das médias de mínimos quadrados", alinhar_esq=2)}
{fig2}

<h2>3.4 Por que a MANOVA pode divergir de três ANOVAs separadas</h2>
<ol>
<li><b>Controle do erro tipo I:</b> três ANOVAs a 5% elevam a probabilidade de pelo menos um falso positivo para
1 − 0,95³ ≈ 14,3%. A MANOVA realiza um único teste ao nível nominal de 5%.</li>
<li><b>Uso da estrutura de covariância:</b> as ANOVAs tratam cada variável isoladamente e ignoram as correlações. A MANOVA
avalia o afastamento entre os centróides dos grupos em uma métrica que leva em conta as covariâncias (distância de
Mahalanobis). Duas variáveis correlacionadas carregam informação parcialmente redundante; a MANOVA não a “conta duas
vezes”, o que pode reduzir a evidência em relação às análises isoladas.</li>
<li><b>Efeitos de combinação:</b> grupos podem não diferir de forma relevante em nenhuma variável isolada e, ainda assim, diferir
em uma <i>combinação linear</i> delas (por exemplo, quando a relação entre gasto e NPS muda de sentido entre regiões).
Nesses casos a MANOVA detecta diferenças invisíveis às ANOVAs. O inverso também ocorre: uma diferença univariada
fraca pode não ser significativa no teste multivariado, que “dilui” o efeito entre os graus de liberdade das três variáveis.</li>
<li><b>Nesta base</b>, ambos os enfoques concordam (todas as ANOVAs significativas mesmo após Bonferroni), mas a MANOVA
acrescenta que as diferenças regionais <b>não</b> são três fenômenos independentes: elas se organizam em duas dimensões
(gasto digital e percepção/NPS), e o padrão de pares difere por variável (Tabela 10) – Sul lidera em gasto, Centro em NPS.</li>
</ol>

<h2>3.5 Interpretação negocial</h2>
<p>A região afeta o “pacote” de desempenho do cliente, mas de forma diferente para cada componente. O <b>Sul</b> é a
região de maior valor financeiro (Gasto Online médio de R$ {fn(r2['desc'].loc['Sul',('Gasto_Online','mean')],0)}, cerca de R$ 158 acima do Norte),
o <b>Centro</b> combina gasto intermediário com o maior NPS ({fn(r2['desc'].loc['Centro',('NPS_Pontuacao','mean')],1)}), e o <b>Norte</b> é inferior em todas as
métricas. O gasto em loja física varia pouco entre regiões (η² = {fn(u.eta2.iloc[1],3)}; só Norte × Sul difere), logo as
diferenças regionais são essencialmente de <b>canal digital</b> e de <b>percepção da marca</b>. Implicações: investir em
ativação digital no Norte; no Centro, converter a boa percepção em gasto (programas de indicação, ofertas
personalizadas); no Sul, proteger a base de alto valor e monitorar se o NPS acompanha o gasto. Avaliar só a
satisfação (Parte I) levaria a concluir que o Centro é a melhor região; a visão multivariada mostra que, em valor
financeiro, o Sul é superior.</p>
""")

# =============================================================== PARTE III
X3 = C.X3
lin_u3 = [[NOMES[r.variavel], fn(r.media0, 2), fn(r.media1, 2), fn(r.lambda_wilks, 4), fn(r.F, 2), fp(r.p)] for r in r3["univ"].itertuples()]
bm3 = r3["boxm"]
lin_p3 = [[f"Mardia – assimetria (Status {k})", fn(r3['mardia'][k]['chi2'], 2), fp(r3['mardia'][k]['p_ass'])] for k in [0, 1]]
lin_p3 += [[f"Mardia – curtose (Status {k})", fn(r3['mardia'][k]['z'], 2), fp(r3['mardia'][k]['p_curt'])] for k in [0, 1]]
lin_p3 += [["M de Box", fn(bm3['chi2'], 2) + f" (gl = {int(bm3['gl'])})", fp(bm3['p'])]]
lin_vif = [[NOMES[v], fn(r3['vif'][v], 2)] for v in X3]
lin_coef = [[NOMES[v], fn(r3['coef_bruto_norm'][v], 5), fn(r3['coef_pad_norm'][v], 3), fn(r3['cargas'][v], 3), fn(r3['cargas_total'][v], 3)] for v in X3]
lin_coef.append(["Constante", fn(r3['const_norm'], 4), "", "", ""])
wl = r3["wilks"]
cls = r3["cls_iguais"]; clp = r3["cls_proporcionais"]
def conf_tab(cf, titulo):
    cf = cf.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    tot = cf.sum(1)
    lin = [[ROT_ST[k], cf.loc[k, 0], cf.loc[k, 1], tot[k], f"{cf.loc[k, k] / tot[k]:.1%}".replace(".", ",")] for k in [0, 1]]
    return tabela(lin, ["Grupo real", "Previsto 0", "Previsto 1", "Total", "% correto"], titulo)
ho = r3["holdout"]
cz = r3["centroides"]
# índice de gasto não explicado pela frequência (para interpretação)
res_on = df.Gasto_Online - 35 * df.Freq_Online
res_lj = df.Gasto_Loja - 40 * df.Freq_Loja
r_on = np.corrcoef(res_on, r3["z"])[0, 1]

fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
ax = axs[0]
z = r3["z"]; y = df.Status_Fidelidade.values
bins = np.linspace(z.min() - 0.5, z.max() + 0.5, 40)
for k in [0, 1]:
    ax.hist(z[y == k], bins=bins, color=COR_ST[k], alpha=0.75, label=ROT_ST[k], edgecolor="white", linewidth=0.6)
corte_z = (cz[0] + cz[1]) / 2
ax.axvline(corte_z, color=INK, lw=1.2, ls="--")
ax.text(corte_z, ax.get_ylim()[1] * 0.55, "  ponto de corte", fontsize=8.5, color=INK)
ax.set_xlabel("Escore discriminante"); ax.set_ylabel("Nº de clientes")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
ax.set_title("Distribuição do escore discriminante por grupo")
ax = axs[1]
for k in [0, 1]:
    m = y == k
    ax.scatter(df.Freq_Online[m], df.Gasto_Online[m], s=10, color=COR_ST[k], alpha=0.7, lw=0, label=ROT_ST[k])
xx = np.linspace(df.Freq_Online.min(), df.Freq_Online.max(), 10)
ax.set_xlabel("Freq. Online (transações/ano)"); ax.set_ylabel("Gasto Online (R$)")
ax.set_title("Mesma frequência, gasto maior no grupo 1")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
fig.tight_layout()
fig3 = fig_html(fig, "Figura 4 – Separação obtida pela função discriminante (esq.) e a origem da separação: para cada nível de frequência online, os clientes Ativos/Alto Valor gastam sistematicamente mais (dir.).")

H.append(f"""
<h1>4. Parte II (Questão 3) – Função discriminante para o Status de Fidelidade</h1>
<h2>4.1 Metodologia</h2>
<p>O problema tem uma <b>variável dependente categórica com dois grupos conhecidos a priori</b> (Status 0 e 1) e
<b>cinco preditoras quantitativas contínuas</b>. Deseja-se (a) encontrar a combinação linear das preditoras que
maximiza a separação entre os grupos e (b) classificar novos clientes. Essa é exatamente a proposta da
<b>Análise Discriminante Linear de Fisher</b>: a função Z = a<sub>1</sub>X<sub>1</sub> + … + a<sub>5</sub>X<sub>5</sub> é escolhida de modo a
maximizar a razão entre a variabilidade entre grupos e a variabilidade dentro dos grupos, o que, para dois grupos,
leva a <b>a</b> = <b>S</b><sub>p</sub><sup>−1</sup>(<b>x̄</b><sub>1</sub> − <b>x̄</b><sub>0</sub>), em que <b>S</b><sub>p</sub> é a matriz de covariâncias combinada.
A forma linear é justificada quando as matrizes de covariâncias dos grupos são iguais (caso contrário, seria
indicada a forma quadrática). A classificação usa o ponto de corte entre os centróides dos grupos, com
probabilidades a priori iguais e, como sensibilidade, proporcionais aos tamanhos dos grupos.</p>

<h2>4.2 Validação dos pressupostos</h2>
<ul>
<li><b>Normalidade multivariada em cada grupo:</b> Mardia não rejeita (Tabela 12).</li>
<li><b>Igualdade das matrizes de covariâncias:</b> M de Box χ² = {fn(bm3['chi2'],2)}, gl = {int(bm3['gl'])}, p = {fp(bm3['p'])} – não se rejeita a
homogeneidade ao nível de 5%, justificando a função <b>linear</b>.</li>
<li><b>Multicolinearidade:</b> há colinearidade forte entre Gasto Loja e Freq. Loja (r = 0,98; VIF ≈ 28) e moderada entre Gasto Online e
Freq. Online (r = 0,80; VIF ≈ 3,4). Isso não prejudica a capacidade de classificação, mas torna os coeficientes
padronizados instáveis e de sinais opostos; por isso a interpretação combina coeficientes e cargas estruturais.</li>
<li><b>Tamanho amostral:</b> 160 e 140 clientes, isto é, mais de 28 observações por preditora no menor grupo (mínimo usual: 20).</li>
</ul>
{tabela(lin_u3, ["Variável", "Média Status 0", "Média Status 1", "Λ de Wilks", "F (1; 298)", "p-valor"], "Tabela 11 – Testes univariados de igualdade de médias entre os grupos")}
{tabela(lin_p3, ["Teste de pressuposto", "Estatística", "p-valor"], "Tabela 12 – Pressupostos da análise discriminante")}
{tabela(lin_vif, ["Variável", "VIF"], "Tabela 13 – Fatores de inflação da variância")}

<h2>4.3 A função discriminante</h2>
<p>A função é altamente significativa: Λ de Wilks = {fn(wl['lam'],4)}, χ² = {fn(wl['chi2'],1)} (gl = 5), p {fp(wl['p'])}; autovalor =
{fn(r3['autovalor'],3)}; <b>correlação canônica = {fn(r3['rcan'],4)}</b> (93% da variância do escore é explicada pela pertença ao grupo).
A distância de Mahalanobis entre os centróides é D² = {fn(r3['D2'],2)} (T² de Hotelling: F = {fn(r3['hotelling']['F'],1)}; gl = 5; 294).
Centróides: Status 0 = {fn(cz[0],3)}; Status 1 = {fn(cz[1],3)} (escala com variância intra-grupo unitária).</p>
{tabela(lin_coef, ["Variável", "Coef. não padronizado", "Coef. padronizado", "Carga estrutural (intra-grupos)", "Correlação com o escore (amostra total)"], "Tabela 14 – Coeficientes e cargas da função discriminante")}
<p><b>Pesos padronizados.</b> Em valor absoluto, os maiores pesos padronizados são de Gasto Online
({fn(r3['coef_pad_norm']['Gasto_Online'],2)}), Gasto Loja ({fn(r3['coef_pad_norm']['Gasto_Loja'],2)}), Freq. Online ({fn(r3['coef_pad_norm']['Freq_Online'],2)}) e
Freq. Loja ({fn(r3['coef_pad_norm']['Freq_Loja'],2)}); Tempo de Relação tem peso praticamente nulo ({fn(r3['coef_pad_norm']['Tempo_Relacao'],3)}).
Entretanto, a colinearidade elevada entre Gasto Loja e Freq. Loja (VIF ≈ 28) e, em menor grau, entre Gasto Online e
Freq. Online torna esses coeficientes <b>sensíveis e potencialmente instáveis</b>: os sinais opostos dentro de cada par
gasto/frequência são consequência dessa colinearidade. Assim, não é adequado interpretar isoladamente os sinais e
magnitudes dos coeficientes, nem concluir que a função representa exatamente um índice de ticket médio. As cargas
estruturais e as correlações com o escore também são relativamente modestas; a evidência mais clara é a diferença
univariada em <b>Gasto Online</b> (Tabela 11), acompanhada por diferenças menores em Gasto Loja e Freq. Loja. Vale notar que,
pela regra de geração dos dados, o status de fidelidade só atua sobre o Gasto Online; as diferenças observadas no canal
físico são, portanto, provavelmente flutuação amostral desta base. A função separa muito bem os grupos nesta base
simulada, mas a contribuição individual das variáveis deve ser interpretada com cautela.</p>

<h2>4.4 Capacidade de classificação</h2>
{conf_tab(cls['conf'], "Tabela 15 – Matriz de confusão por resubstituição (a priori iguais)")}
{conf_tab(cls['conf_loo'], "Tabela 16 – Matriz de confusão por validação cruzada “deixe-um-fora” (Lachenbruch)")}
{conf_tab(ho['conf'], f"Tabela 17 – Validação com amostra de teste (função estimada em {ho['n_tr']} clientes, aplicada a {ho['n_te']} novos)")}
<p>A taxa de erro aparente (APER), a taxa de erro estimada por validação cruzada e a taxa de erro na amostra de teste foram,
respectivamente, {fn(cls['aper']*100,1)}%, {fn(cls['loo']*100,1)}% e {fn(ho['erro']*100,1)}%. Com probabilidades a priori proporcionais (π<sub>1</sub> = {fn(st_n[1]/N,3)}) o resultado é idêntico
(erro de {fn(clp['loo']*100,1)}%). O desempenho supera com folga os critérios de referência: chance proporcional C<sub>pro</sub> = {fn(r3['cpro']*100,1)}% e
chance máxima C<sub>max</sub> = {fn(r3['cmax']*100,1)}% (a regra de 25% acima da chance exigiria ≥ 62,8%); o Q de Press = {fn(r3['pressQ'],1)} é muito superior ao valor crítico
χ²<sub>1; 0,01</sub> = 6,63. A separação perfeita é um resultado desta base simulada (o status adiciona R$ 200 ao gasto online, com ruído de
desvio-padrão R$ 15) e não deve ser generalizada: não se deve atribuir toda a separação a um único indicador nem tratar
a função como regra causal, e o desempenho deve ser validado em dados externos antes de qualquer aplicação real.</p>
{fig3}

<h2>4.5 Interpretação negocial</h2>
<p>Nesta base, o que mais distingue o cliente Ativo/Alto Valor é o <b>Gasto Online</b>: a frequência online e o tempo de
relação são estatisticamente iguais nos dois grupos, enquanto o gasto online é, em média, cerca de R$ 190 maior no
grupo 1. Isso sugere, como <b>hipótese gerencial a validar em dados reais</b>, que o valor gasto no canal digital – e não a
contagem de transações – seja um bom indicador inicial do perfil de alto valor. Implicações possíveis: (i) usar o gasto
online como indicador de acompanhamento do programa de fidelidade; (ii) testar campanhas de aumento de cesta
(cross-sell, up-sell, frete grátis acima de um valor) para clientes do grupo 0, medindo o efeito com grupo de controle;
(iii) aplicar a função como regra de pontuação somente após validá-la em dados externos, pois a separação perfeita aqui
observada é específica desta amostra simulada.</p>
""")

# =============================================================== PARTE IV (MDS)
sd = r4["stress_dim"]
def kruskal(s):
    return "perfeito" if s < 0.0251 else "excelente" if s < 0.05 else "bom" if s < 0.10 else "razoável" if s < 0.20 else "pobre"
lin_sd = [[int(r.dim), fn(r.stress, 4), kruskal(r.stress), fn(r.rsq, 4), f"{r.var_acum:.1%}".replace(".", ",")] for r in sd.itertuples()]
cd = r4["corr_dim"]
lin_cd = [[NOMES[v], fn(cd.loc[v, "Dim 1"], 3), fn(cd.loc[v, "Dim 2"], 3)] for v in X3]
s2 = sd.iloc[1]
co = r4["coords"]
# centróides
lin_cen = [[ROT_ST[k], fn(r4['centro_status'].loc[k, 'D1'], 3), fn(r4['centro_status'].loc[k, 'D2'], 3)] for k in [0, 1]]
lin_cen += [[g, fn(r4['centro_regiao'].loc[g, 'D1'], 3), fn(r4['centro_regiao'].loc[g, 'D2'], 3)] for g in ORD]

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (campo, cores, rot) in zip(axs, [("Status_Fidelidade", COR_ST, ROT_ST), ("Regiao_nome", COR_REG, {g: g for g in ORD})]):
    chaves = [0, 1] if campo == "Status_Fidelidade" else ORD
    for k in chaves:
        m = (df[campo] == k).values
        ax.scatter(co[m, 0], co[m, 1], s=12, color=cores[k], alpha=0.7, lw=0, label=rot[k])
        ax.scatter(co[m, 0].mean(), co[m, 1].mean(), s=90, color=cores[k], edgecolor="white", lw=1.8, marker="D", zorder=5)
    ax.axhline(0, color="#bdbcb6", lw=0.8); ax.axvline(0, color="#bdbcb6", lw=0.8)
    ax.set_xlabel("Dimensão 1 – volume geral de compras"); ax.set_ylabel("Dimensão 2 – canal (online + / loja −)")
    ax.legend(loc="lower left", fontsize=8, markerscale=1.5)
    ax.set_title("Mapa 2D colorido por " + ("Status de Fidelidade" if campo == "Status_Fidelidade" else "Região"))
    for txt, (x, yv) in {"Mais compras\nonline": (2.2, 2.4), "Mais compras\nem loja": (2.0, -2.6), "Baixo volume": (-3.1, 0.2)}.items():
        ax.text(x, yv, txt, fontsize=7.5, color=INK2, ha="center", style="italic")
fig.tight_layout()
fig4 = fig_html(fig, "Figura 5 – Configuração bidimensional do MDS (losangos = centróides dos grupos).")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
ax = axs[0]
rng = np.random.default_rng(0)
idx = rng.choice(len(r4["d_orig"]), 6000, replace=False)
ax.scatter(r4["d_orig"][idx], r4["d_2d"][idx], s=3, color=BLUE, alpha=0.35, lw=0)
mx = r4["d_orig"].max()
ax.plot([0, mx], [0, mx], color=INK2, lw=1)
ax.set_xlabel("Distância original (5 dimensões)"); ax.set_ylabel("Distância no mapa 2D")
ax.set_title("Diagrama de Shepard (amostra de 6.000 pares)")
ax = axs[1]
ax.plot(sd.dim, sd.stress, "-o", color=BLUE, lw=2, ms=6)
for t, c in [(0.20, "razoável"), (0.10, "bom"), (0.05, "excelente")]:
    ax.axhline(t, color=GRID, lw=1, ls="--"); ax.text(4.95, t + 0.008, c, fontsize=7.5, color=INK2, ha="right")
ax.set_xticks(sd.dim); ax.set_xlabel("Nº de dimensões"); ax.set_ylabel("Stress-1 de Kruskal")
ax.set_title("Stress × dimensionalidade")
fig.tight_layout()
fig5 = fig_html(fig, "Figura 6 – Qualidade do ajuste: diagrama de Shepard da solução 2D (esq.) e curva de stress (dir.).")

H.append(f"""
<h1>5. Parte III – Mapeamento perceptual dos perfis (Escalonamento Multidimensional)</h1>
<h2>5.1 Metodologia</h2>
<p>O objetivo é representar os 300 clientes em um mapa bidimensional que preserve, tanto quanto possível, as
<b>distâncias (dissimilaridades)</b> entre eles. Essa é a definição do <b>Escalonamento Multidimensional (MDS)</b>:
a partir de uma matriz de dissimilaridades n × n, busca-se uma configuração de pontos em poucas dimensões cujas
distâncias euclidianas reproduzam as originais. Utilizaram-se as cinco variáveis comportamentais (Gasto Online, Gasto
Loja, Freq. Online, Freq. Loja e Tempo de Relação). Como estão em escalas muito distintas (R$, contagens, meses),
foram <b>padronizadas (escore z)</b> antes do cálculo da <b>distância euclidiana</b>, para que nenhuma variável domine a
distância por causa da unidade de medida. Como as dissimilaridades são métricas (razão), adotou-se o <b>MDS métrico
clássico (Torgerson)</b>, que obtém as coordenadas pela decomposição espectral da matriz de produtos escalares
duplamente centrada; como verificação, ajustou-se também o <b>MDS não métrico de Kruskal</b>, que preserva apenas a
ordem das distâncias.</p>

<h2>5.2 Validação</h2>
<p>O MDS não exige normalidade nem outros pressupostos distribucionais; as condições a verificar são (i) a escolha
adequada da medida de distância (euclidiana em variáveis padronizadas, justificada acima), (ii) a ausência de
autovalores negativos relevantes na solução clássica – todos os autovalores são não negativos, o que confirma a
natureza euclidiana das distâncias – e (iii) a qualidade do ajuste, medida pelo stress e pelo RSQ.</p>

<h2>5.3 Métricas de ajuste</h2>
{tabela(lin_sd, ["Dimensões", "Stress-1 de Kruskal", "Classificação (Kruskal)", "RSQ", "Variância acumulada (autovalores)"], "Tabela 18 – Indicadores de ajuste da solução métrica clássica por número de dimensões")}
<p>Na solução bidimensional, o <b>stress-1</b> calculado entre as dissimilaridades originais e as distâncias do mapa é
<b>{fn(s2.stress,3)}</b>. A classificação “razoável” usa apenas como referência a escala interpretativa usualmente associada a
Kruskal; ela não significa que o algoritmo ajustado tenha sido o MDS não métrico. O <b>RSQ é {fn(s2.rsq,3)}</b> e as duas dimensões
concentram {fn(s2.var_acum*100,1)}% da soma dos autovalores. Em análise complementar, o MDS não métrico de Kruskal em 2D obteve
stress-1 = {fn(r4['nm_stress'],3)} e RSQ = {fn(r4['nm_rsq'],3)}. Os resultados dos dois procedimentos devem ser distinguidos: o clássico
métrico procura reproduzir as distâncias, enquanto o não métrico preserva principalmente sua ordenação. A curva de
stress (Figura 6) mostra melhora expressiva em 3 dimensões (stress {fn(sd.stress.iloc[2],3)}); optou-se pela solução 2D por facilitar a
visualização e atender ao objetivo de síntese.</p>
{fig5}

<h2>5.4 Interpretação da configuração</h2>
{tabela(lin_cd, ["Variável", "Dimensão 1", "Dimensão 2"], "Tabela 19 – Correlação entre as variáveis originais e as coordenadas do mapa")}
<p><b>Dimensão 1 (horizontal) – volume geral de relacionamento:</b> todas as variáveis se correlacionam positivamente com ela
(0,47 a 0,77). Clientes à direita compram mais, nos dois canais. <b>Dimensão 2 (vertical) – preferência de canal:</b> opõe as
variáveis online e de tempo de relação (positivas) às de loja física (negativas, ≈ −0,70). Clientes no alto são
predominantemente digitais; embaixo, predominantemente de loja física.</p>
{tabela(lin_cen, ["Grupo", "Centróide Dim. 1", "Centróide Dim. 2"], "Tabela 20 – Posição média dos grupos no mapa")}
{fig4}
<p><b>Proximidade e agrupamento.</b> No mapa, a distância entre dois pontos é proporcional à dissimilaridade de comportamento: clientes
próximos têm perfis de compra parecidos e podem receber a mesma abordagem. A configuração não mostra “ilhas” bem
separadas, e sim um <b>contínuo</b> com densidade maior no centro, organizado ao longo de dois gradientes (volume e canal). Os quadrantes
definem perfis naturais: <i>digitais intensos</i> (direita-alto), <i>lojistas intensos</i> (direita-baixo), <i>digitais leves</i>
(esquerda-alto) e <i>clientes de baixo envolvimento</i> (esquerda). Os clientes Ativos/Alto Valor deslocam-se para a direita
(centróide {fn(r4['centro_status'].loc[1,'D1'],2)} vs. {fn(r4['centro_status'].loc[0,'D1'],2)}) e os do Sul também (maior gasto), mas com grande
sobreposição: a distância média entre clientes de grupos de status diferentes ({fn(r4['dist_inter'],2)}) é quase igual à distância média
dentro do mesmo grupo ({fn(r4['dist_intra'],2)}). Isso é coerente com a Questão 3: o status não é visível no comportamento “bruto” (volume e
canal), mas apenas em uma combinação específica (gasto por transação) que a análise discriminante isolou.</p>

<h2>5.5 Interpretação negocial</h2>
<p>O mapa é uma ferramenta de segmentação visual: (i) a <b>posição horizontal</b> indica o potencial de valor e deve orientar a
intensidade do investimento de relacionamento; (ii) a <b>posição vertical</b> indica o canal preferido e deve orientar onde a
comunicação é feita (aplicativo/e-mail para o alto do mapa; encarte, vendedor e eventos em loja para a parte de baixo);
(iii) clientes do quadrante esquerdo-baixo são candidatos a campanhas de reativação. Como os perfis formam um contínuo,
recomenda-se tratar a segmentação por gradientes (escores) e não por fronteiras rígidas, e reconstruir o mapa
periodicamente para acompanhar migrações de clientes entre perfis.</p>
""")

# =============================================================== PARTE V (CCA)
B1, B2 = C.B1, C.B2
ts = r5["testes"]
rc, rc2 = r5["rc"], r5["rc2"]
mg = r5["mv_global"]
mr = r5["mardia"]
red = r5["red"]
A_, B_ = r5["A"], r5["B"]
LX, LY, CX, CY = r5["LX"], r5["LY"], r5["CX"], r5["CY"]
av = r5["autoval"]

# Tabela de correlações canônicas (formato PROC CANCORR)
lin_rc = [[k + 1, fn(rc[k], 4), fn(r5['ep'][k], 4), fn(rc2[k], 4), fn(av.autovalor[k], 4),
           fn(av.prop[k], 4), fn(av.acum[k], 4)] for k in range(3)]
# Testes multivariados globais
nomes_mv = {"Wilks' lambda": "Lambda de Wilks", "Pillai's trace": "Traço de Pillai",
            "Hotelling-Lawley trace": "Traço de Hotelling-Lawley", "Roy's greatest root": "Maior raiz de Roy"}
lin_mg = [[nomes_mv[k], fn(float(mg.loc[k, "Value"]), 4), fn(float(mg.loc[k, "F Value"]), 2), fn(float(mg.loc[k, "Num DF"]), 0),
           fn(float(mg.loc[k, "Den DF"]), 2), fp(float(mg.loc[k, "Pr > F"]))] for k in nomes_mv]
# Testes por função (qui-quadrado da aula + F de Rao)
lin_ts = [[f"{int(r.dim)}ª função em diante" if r.dim < 3 else "3ª função", fn(r.lambda_wilks, 5), fn(r.chi2, 2), int(r.gl), fp(r.p_chi),
           fn(r.F, 2), f"{int(r.gl1)}; {fn(r.gl2, 1)}", fp(r.p_F)] for r in ts.itertuples()]
# Redundância no formato do SAS
def tab_red(prop_own, prop_opp, titulo, rot):
    lin = []
    ca = co = 0
    for k in range(3):
        ca += prop_own[k]; co += prop_opp[k]
        lin.append([f"{k+1}", fn(prop_own[k], 4), fn(ca, 4), fn(rc2[k], 4), fn(prop_opp[k], 4), fn(co, 4)])
    return tabela(lin, ["Função canônica", "Proporção (próprias)", "Acumulada", "R² canônico", f"Proporção (opostas)", "Acumulada"], titulo)
red_dep = tab_red(red.var_prop_Y.values, red.red_Y.values,
                  "Tabela 25 – Variância padronizada do Bloco 2 (dependentes) explicada por suas próprias variáveis canônicas e pelas variáveis canônicas opostas (Bloco 1)", "")
red_ind = tab_red(red.var_prop_X.values, red.red_X.values,
                  "Tabela 26 – Variância padronizada do Bloco 1 (independentes) explicada por suas próprias variáveis canônicas e pelas variáveis canônicas opostas (Bloco 2)", "")
# Pesos / cargas / cargas cruzadas (formato da aula)
def tab_interp(k):
    lin = []
    for v in B1:
        lin.append([f"Bloco 1 (indep.)", NOMES[v], fn(A_.loc[v].iloc[k], 4), fn(LX.loc[v].iloc[k], 4), fn(CX.loc[v].iloc[k], 4), f"{CX.loc[v].iloc[k]**2:.1%}".replace(".", ",")])
    for v in B2:
        lin.append([f"Bloco 2 (dep.)", NOMES[v], fn(B_.loc[v].iloc[k], 4), fn(LY.loc[v].iloc[k], 4), fn(CY.loc[v].iloc[k], 4), f"{CY.loc[v].iloc[k]**2:.1%}".replace(".", ",")])
    return tabela(lin, ["Bloco", "Variável", "Peso (coef. padronizado)", "Carga (loading)", "Carga cruzada (cross-loading)", "Carga cruzada²"],
                  f"Tabela {27 + k} – Função canônica {k+1}: pesos, cargas e cargas cruzadas", alinhar_esq=2)
# variância de cada dependente explicada pelo bloco independente (2 funções significativas)
expl = {v: CY.loc[v].iloc[0]**2 + CY.loc[v].iloc[1]**2 for v in B2}
lin_expl = [[NOMES[v], f"{CY.loc[v].iloc[0]**2:.1%}".replace(".", ","), f"{CY.loc[v].iloc[1]**2:.1%}".replace(".", ","), f"{expl[v]:.1%}".replace(".", ",")] for v in B2]

def eq(coef, nomes):
    t = ""
    for i, v in enumerate(nomes):
        c = coef.loc[v]
        sinal = (" − " if c < 0 else " + ") if i else ("−" if c < 0 else "")
        t += f"{sinal}{fn(abs(c), 4)}·{NOMES[v]}"
    return t
eqs = "".join(f'<p class="eq">Função {k+1}: &nbsp; {eq(B_.iloc[:, k], B2)} &nbsp;=&nbsp; {eq(A_.iloc[:, k], B1)} &nbsp; (r<sub>c</sub> = {fn(rc[k], 4)})</p>' for k in range(2))

# validação em subamostras
val = r5["validacao"]
lin_val = []
for nome, nn, sub in val:
    lin_val.append([nome, nn, fn(sub['rc'][0], 3), fn(sub['rc'][1], 3), fn(sub['rc'][2], 3),
                    fn(sub['LX'][1, 0], 2), fn(sub['LY'][1, 0], 2), fn(sub['LX'][0, 1], 2), fn(sub['LY'][0, 1], 2)])
lin_val.append(["Amostra completa", N, fn(rc[0], 3), fn(rc[1], 3), fn(rc[2], 3), fn(LX.loc['Freq_Loja', 'U1'], 2), fn(LY.loc['Gasto_Loja', 'V1'], 2),
                fn(LX.loc['Freq_Online', 'U2'], 2), fn(LY.loc['Gasto_Online', 'V2'], 2)])

Rx = r5["R"]
lin_rxy = [[NOMES[a]] + [fn(Rx.loc[a, b], 3) for b in B2] for a in B1]

fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for k, ax in enumerate(axs):
    ax.scatter(r5["U"][:, k], r5["V"][:, k], s=9, color=[BLUE, ORANGE][k], alpha=0.6, lw=0)
    ax.set_xlabel(f"U{k+1} – variável canônica do Bloco 1 (engajamento)")
    ax.set_ylabel(f"V{k+1} – variável canônica do Bloco 2 (retorno)")
    ax.set_title(f"Função canônica {k+1}: r = {rc[k]:.3f}".replace(".", ","))
fig.tight_layout()
fig6 = fig_html(fig, "Figura 7 – Escores das duas funções canônicas significativas.")

fig, ax = plt.subplots(figsize=(6.6, 3.6))
nomes_all = [NOMES[v] for v in B1 + B2]
car1 = list(LX["U1"]) + list(LY["V1"]); car2 = list(LX["U2"]) + list(LY["V2"])
yy = np.arange(len(nomes_all))[::-1]
ax.barh(yy + 0.2, car1, height=0.36, color=BLUE, label="Função 1 (canal físico)")
ax.barh(yy - 0.2, car2, height=0.36, color=ORANGE, label="Função 2 (canal digital)")
ax.set_yticks(yy); ax.set_yticklabels(nomes_all); ax.axvline(0, color=INK2, lw=0.8)
ax.axhline(2.5, color="#bdbcb6", lw=0.8, ls="--")
ax.text(1.02, 4, "Bloco 1", fontsize=8, color=INK2, rotation=90, va="center", transform=ax.get_yaxis_transform())
ax.text(1.02, 1, "Bloco 2", fontsize=8, color=INK2, rotation=90, va="center", transform=ax.get_yaxis_transform())
ax.set_xlim(-0.3, 1.1); ax.set_xlabel("Carga canônica (correlação com a variável canônica)")
ax.legend(loc="lower right", fontsize=8); ax.set_title("Cargas canônicas das funções significativas")
fig.tight_layout()
fig7 = fig_html(fig, "Figura 8 – Estrutura de cargas: a Função 1 é formada pelo canal físico e a Função 2 pelo canal digital.", "72%")

H.append(f"""
<h1>6. Parte IV – Relação entre Engajamento Operacional e Retorno Financeiro/Percepção</h1>
<p>A análise segue as etapas da correlação canônica: (1) especificação dos objetivos, (2) plano de análise, (3) pressupostos,
(4) cálculo do modelo, (5) interpretação dos resultados e (6) validação.</p>

<h2>6.1 Objetivos e justificativa do método</h2>
<p>Deseja-se avaliar a associação entre dois conjuntos de variáveis <b>métricas</b>: o <b>Bloco 1 – Engajamento Operacional</b>
(Freq. Online, Freq. Loja, Tempo de Relação), tratado como conjunto <b>independente</b>, e o <b>Bloco 2 – Retorno Financeiro e
Percepção</b> (Gasto Online, Gasto Loja, NPS), tratado como conjunto <b>dependente</b>. Os objetivos são verificar se existe relação
entre os blocos e explicar a natureza dessa relação, medindo a contribuição de cada variável.</p>
<p>As alternativas mais simples não são adequadas: (i) analisar as 3 × 3 = 9 correlações cruzadas gera excesso de resultados
e não isola o efeito de cada variável, pois as variáveis de cada bloco são correlacionadas entre si; (ii) uma série de três
regressões múltiplas, uma para cada variável dependente, produz equações correlacionadas, já que as dependentes são
correlacionadas; (iii) a análise fatorial estuda o inter-relacionamento dentro de <i>um</i> conjunto, e aqui há dois. A
<b>correlação canônica</b>, extensão da regressão múltipla para várias dependentes, resolve esses problemas: constrói, para
cada bloco, um índice composto (soma ponderada das variáveis), U = <b>a</b>'<b>X</b> e V = <b>b</b>'<b>Y</b>, e escolhe os pesos de modo a
<b>maximizar a correlação entre os dois índices</b>. Os pesos e as correlações canônicas são obtidos pelos autovetores e
autovalores de <b>R</b><sub>11</sub><sup>−1</sup><b>R</b><sub>12</sub><b>R</b><sub>22</sub><sup>−1</sup><b>R</b><sub>21</sub>; os pares seguintes maximizam a
correlação restante, sendo não correlacionados com os anteriores. Com 3 variáveis em cada bloco existem até 3 funções
canônicas. Como as unidades são diferentes (R$, contagens, meses, pontos), as variáveis foram padronizadas.</p>
{tabela(lin_rxy, ["Bloco 1 \\ Bloco 2"] + [NOMES[b] for b in B2], "Tabela 21 – Correlações entre os blocos (as 9 correlações que a análise canônica resume)")}

<h2>6.2 Plano de análise e pressupostos</h2>
<ul>
<li><b>Tamanho da amostra:</b> pela regra de pelo menos 10 observações por variável, seriam necessárias 60 observações
(6 variáveis); a amostra de 300 clientes (50 por variável) atende com folga.</li>
<li><b>Linearidade:</b> a correlação canônica capta apenas relações lineares. As correlações de Pearson e de Spearman entre os
blocos são praticamente iguais (ex.: Freq. Loja × Gasto Loja 0,979 vs. 0,976; Freq. Online × Gasto Online 0,795 vs. 0,764).</li>
<li><b>Normalidade multivariada</b> (necessária para os testes de significância): Mardia não rejeita a assimetria (χ² = {fn(mr['chi2'],2)}; p = {fp(mr['p_ass'])});
a curtose é levemente inferior à esperada (b<sub>2</sub> = {fn(mr['b2'],2)} vs. 48; p = {fp(mr['p_curt'])}) pelo truncamento do NPS em 100 –
desvio de pequena magnitude diante de n = 300.</li>
<li><b>Multicolinearidade dentro dos blocos:</b> VIFs entre {fn(r5['vif'].min(),2)} e {fn(r5['vif'].max(),2)} (calculados dentro de cada bloco). Ainda assim,
como recomendado, a interpretação se apoia principalmente nas cargas e cargas cruzadas, e não apenas nos pesos.</li>
</ul>

<h2>6.3 Cálculo do modelo</h2>
{tabela(lin_rc, ["Função", "Correlação canônica", "Erro-padrão aprox.", "R² canônico", "Autovalor", "Proporção", "Acumulada"], "Tabela 22 – Correlações canônicas e autovalores")}
<p>A primeira correlação canônica é {fn(rc[0],4)} (R² = {fn(rc2[0],4)}) e a segunda, {fn(rc[1],4)} (R² = {fn(rc2[1],4)}); a terceira é praticamente nula
({fn(rc[2],4)}). Escritas com os pesos canônicos padronizados (dependentes = independentes), as duas primeiras funções são:</p>
{eqs}

<h2>6.4 Significância estatística das funções</h2>
<p>As funções devem ser testadas em conjunto e isoladamente. O teste conjunto usa os quatro critérios multivariados:</p>
{tabela(lin_mg, ["Estatística", "Valor", "F aproximado", "gl num.", "gl den.", "p-valor"], "Tabela 23 – Testes multivariados para o conjunto das funções canônicas (s = 3; m = −0,5; N = 146)")}
<p>Todos rejeitam a hipótese nula de que as funções canônicas não são significantes (p {fp(float(mg.loc["Wilks' lambda", "Pr > F"]))}). Para testar cada função,
utiliza-se a estatística qui-quadrado de Bartlett, χ² = −[N − 1 − 0,5(p + q + 1)] · ln Π(1 − R²<sub>ci</sub>), com p × q graus de liberdade
para o teste da primeira função em diante, (p − 1)(q − 1) para a segunda em diante e assim por diante; a aproximação F de Rao
é apresentada como confirmação.</p>
{tabela(lin_ts, ["Hipótese testada", "Λ de Wilks", "χ²", "gl", "p (χ²)", "F de Rao", "gl", "p (F)"], "Tabela 24 – Testes sequenciais de cada função canônica")}
<p><b>As funções 1 e 2 são estatisticamente significantes</b>; a função 3 não é (p = {fp(ts.p_chi.iloc[2])}) e é descartada.</p>
{fig6}

<h2>6.5 Significância prática: índice de redundância</h2>
<p>O R² canônico estima a variância compartilhada entre os <i>índices</i> U e V, e não entre os conjuntos de variáveis originais; por
isso pode superestimar a relação. O <b>índice de redundância</b> corrige esse viés: é a proporção da variância das variáveis de um
bloco explicada pela variável canônica do bloco oposto (variância extraída × R² canônico).</p>
{red_dep}
{red_ind}
<p>Para o bloco dependente, a função 1 tem redundância de <b>{fn(red.red_Y.iloc[0],4)}</b> e a função 2 de <b>{fn(red.red_Y.iloc[1],4)}</b>: juntas, as variáveis de
engajamento explicam <b>{fn((red.red_Y.iloc[0]+red.red_Y.iloc[1])*100,1)}% da variância</b> do conjunto Gasto Online, Gasto Loja e NPS. Ambas as funções têm,
portanto, significância prática (diferentemente da função 3, cuja redundância é {fn(red.red_Y.iloc[2],4)}). A redundância não chega a valores mais altos
porque o NPS praticamente não é explicado pelo Bloco 1 (ver Tabela 29).</p>

<h2>6.6 Interpretação: pesos, cargas e cargas cruzadas</h2>
<p>Os <b>pesos</b> canônicos são análogos aos betas da regressão, mas são instáveis entre amostras e difíceis de interpretar quando há
multicolinearidade. Por isso, a interpretação se baseia sobretudo nas <b>cargas canônicas</b> (correlação de cada variável com o índice
do próprio bloco: quanto maior, mais importante a variável na formação do índice) e nas <b>cargas cruzadas</b> (correlação de cada
variável com o índice do bloco oposto; ao quadrado, indicam quanto da variância da variável é explicada pelo outro bloco).</p>
{tab_interp(0)}
{tab_interp(1)}
{tabela(lin_expl, ["Variável dependente", "Pela função 1", "Pela função 2", "Total (funções 1 e 2)"], "Tabela 29 – Variância de cada variável dependente explicada pelo bloco de engajamento (cargas cruzadas ao quadrado)")}
{fig7}
<ul>
<li><b>Função 1 – eixo do canal físico (r<sub>c</sub> = {fn(rc[0],3)}):</b> o índice de engajamento é praticamente a Freq. Loja (carga {fn(LX.loc['Freq_Loja','U1'],3)}) e o índice de
retorno é praticamente o Gasto Loja (carga {fn(LY.loc['Gasto_Loja','V1'],3)}). A carga cruzada do Gasto Loja é {fn(CY.loc['Gasto_Loja','U1'],3)}: o engajamento explica
{CY.loc['Gasto_Loja','U1']**2:.0%} da variância do gasto em loja por esta função. Quem vai mais à loja gasta proporcionalmente mais na loja.</li>
<li><b>Função 2 – eixo do canal digital (r<sub>c</sub> = {fn(rc[1],3)}):</b> o índice de engajamento é dominado por Freq. Online (carga {fn(LX.loc['Freq_Online','U2'],3)}),
com contribuição do Tempo de Relação ({fn(LX.loc['Tempo_Relacao','U2'],3)}); o índice de retorno é dominado por Gasto Online ({fn(LY.loc['Gasto_Online','V2'],3)}).
A carga cruzada do Gasto Online é {fn(CY.loc['Gasto_Online','U2'],3)} ({CY.loc['Gasto_Online','U2']**2:.0%} da variância explicada). O Tempo de Relação tem peso quase nulo
({fn(A_.loc['Tempo_Relacao','U2'],3)}) apesar da carga moderada: sua importância vem da correlação com a Freq. Online (0,45), ilustrando por que pesos
e cargas podem divergir. O NPS recebe peso negativo ({fn(B_.loc['NPS_Pontuacao','V2'],3)}) com carga quase nula, ou seja, atua apenas como ajuste
(supressor) do índice e não deve ser interpretado como relação negativa com o engajamento.</li>
<li><b>NPS:</b> suas cargas cruzadas são baixas nas duas funções; somente {expl['NPS_Pontuacao']:.1%} da sua variância é explicada pelo
engajamento operacional (Tabela 29), contra {expl['Gasto_Loja']:.0%} do Gasto Loja e {expl['Gasto_Online']:.0%} do Gasto Online.</li>
</ul>

<h2>6.7 Validação</h2>
<p>Para verificar a estabilidade do modelo em outra amostra, os clientes foram divididos aleatoriamente em duas metades e a
análise foi refeita em cada uma.</p>
{tabela(lin_val, ["Amostra", "n", "r<sub>c1</sub>", "r<sub>c2</sub>", "r<sub>c3</sub>", "Carga Freq. Loja (U1)", "Carga Gasto Loja (V1)", "Carga Freq. Online (U2)", "Carga Gasto Online (V2)"],
        "Tabela 30 – Validação por subamostras")}
<p>As correlações canônicas e as cargas principais praticamente se repetem nas duas metades, e a terceira função continua
irrelevante: a estrutura encontrada é estável. Como os dados são simulados, recomenda-se ainda repetir a análise em dados
reais antes de usar os resultados operacionalmente.</p>

<h2>6.8 Relação teórica e interpretação negocial</h2>
<p>A relação entre o padrão de uso e o retorno é forte, mas <b>organizada por canal</b>: há dois “circuitos” praticamente
independentes – frequência em loja → gasto em loja e frequência online (associada ao tempo de relação) → gasto online.
O engajamento em um canal não se traduz em retorno no outro canal. Já a <b>percepção da marca (NPS) praticamente não é
explicada pelo engajamento operacional</b>: comprar mais vezes ou estar há mais tempo no programa não torna o cliente mais
promotor; o NPS se relaciona com outros fatores (satisfação, região e status de fidelidade, vistos nas partes anteriores).
Implicações: (i) metas de receita podem ser desdobradas em metas de frequência por canal; (ii) programas de omnicanalidade
precisam de incentivos específicos para levar o cliente de um canal ao outro, pois isso não ocorre espontaneamente; (iii) ações
para melhorar o NPS devem atuar na experiência e na satisfação, e não apenas em estímulos de frequência; (iv) o tempo de
relação parece gerar valor apenas indiretamente, via frequência digital – a retenção deve vir acompanhada de ativação.
Como nas demais partes, essas recomendações são hipóteses gerenciais a validar em dados reais.</p>
""")

# =============================================================== CONCLUSÃO
H.append(f"""
<h1>7. Síntese para a tomada de decisão</h1>
<table><thead><tr><th>Questão</th><th>Método</th><th>Principal achado</th><th>Ação recomendada</th></tr></thead><tbody>
<tr><td class="l">1. Satisfação por região</td><td class="l">ANOVA de um fator + Tukey</td><td class="l">Norte &lt; Sul &lt; Centro, todas diferentes (η² = 0,24)</td><td class="l">Plano de experiência para o Norte; Centro como benchmark; metas regionalizadas</td></tr>
<tr><td class="l">2. Vetor (Gasto Online, Gasto Loja, NPS)</td><td class="l">MANOVA</td><td class="l">Efeito global significativo (Λ = {fn(float(mv.loc["Wilks' lambda","Value"]),3)}); Sul lidera em gasto, Centro em NPS</td><td class="l">Ativação digital no Norte; monetizar a boa percepção do Centro</td></tr>
<tr><td class="l">3. Separação por Status</td><td class="l">Análise discriminante linear de Fisher</td><td class="l">Gasto Online tem a maior diferença média; há também diferenças menores em Gasto Loja e Freq. Loja. Erro observado: {fn(cls['loo']*100,1)}%</td><td class="l">Tratar como indicador inicial; validar em dados externos antes de definir elegibilidade ou risco</td></tr>
<tr><td class="l">4. Mapa perceptual</td><td class="l">Escalonamento multidimensional</td><td class="l">Dois gradientes: volume e canal (stress {fn(s2.stress,3)}; RSQ {fn(s2.rsq,3)})</td><td class="l">Segmentação contínua por valor (eixo 1) e canal (eixo 2)</td></tr>
<tr><td class="l">5. Engajamento × Retorno</td><td class="l">Correlação canônica</td><td class="l">Duas funções significativas, uma por canal (0,98 e 0,92); redundância de {fn((red.red_Y.iloc[0]+red.red_Y.iloc[1])*100,1)}%; NPS não explicado pelo engajamento</td><td class="l">Metas de receita via frequência por canal; NPS via experiência</td></tr>
</tbody></table>
<p>Em conjunto, as análises indicam diferenças regionais na satisfação e no conjunto de indicadores financeiros/perceptuais;
associação forte entre o status de fidelidade e o perfil de gastos; e relações entre frequência e gasto predominantemente
dentro de cada canal. Como há colinearidade entre variáveis de gasto e frequência, a interpretação dos coeficientes
discriminantes individuais exige cautela. As recomendações devem ser tratadas como <b>hipóteses gerenciais a validar em
dados reais</b>, e não como relações causais demonstradas pela simulação.</p>
<p class="nota">Nota metodológica: todos os testes foram conduzidos ao nível de significância de 5%. A base é sintética e foi reproduzida
de forma independente, com gerador pseudoaleatório diferente do SAS; por isso, os resultados não devem ser apresentados
como idênticos aos de uma eventual base SAS original. A separação perfeita observada na análise discriminante é específica
desta amostra e não garante desempenho semelhante em clientes reais. Recomenda-se validar o modelo em dados externos
antes de qualquer uso operacional.</p>
""")

CSS = """
@page { size: A4; margin: 2cm 1.8cm 2cm 1.8cm;
  @bottom-center { content: counter(page); font-size: 9pt; color: #52514e; } }
@page:first { @bottom-center { content: none; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.6pt; line-height: 1.45; color: #0b0b0b; }
h1 { font-size: 14pt; color: #184f95; border-bottom: 1.5px solid #184f95; padding-bottom: 3px; margin-top: 22px; page-break-after: avoid; }
h2 { font-size: 11pt; color: #0b0b0b; margin: 14px 0 6px; page-break-after: avoid; }
p, li { text-align: justify; }
ul, ol { padding-left: 18px; }
li { margin-bottom: 4px; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 12px; font-size: 8.4pt; page-break-inside: avoid; }
th { background: #eef3fa; color: #0b0b0b; font-weight: bold; padding: 4px 6px; border-top: 1px solid #184f95; border-bottom: 1px solid #184f95; text-align: center; }
td { padding: 3px 6px; border-bottom: 0.5px solid #e6e5e1; }
td.r { text-align: right; } td.l { text-align: left; }
.tabtit { font-weight: bold; font-size: 8.8pt; margin: 10px 0 2px; page-break-after: avoid; }
figure { margin: 8px 0 12px; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.4pt; color: #52514e; margin-top: 3px; }
.capa { text-align: center; padding-top: 5cm; page-break-after: always; }
.capa .inst { font-size: 12pt; color: #52514e; }
.capa .tit { font-size: 22pt; font-weight: bold; color: #184f95; margin: 1.5cm 0 0.4cm; }
.capa .sub { font-size: 13pt; }
.capa .info { margin-top: 5cm; font-size: 10.5pt; color: #52514e; }
.sumario { page-break-after: always; }
.sumario li { margin-bottom: 6px; }
.eq { text-align: center; font-size: 9pt; margin: 4px 0; }
.nota { font-size: 8.4pt; color: #52514e; }
"""

capa = """
<div class="capa">
  <div class="inst">Curso de Ciência de Dados e Inteligência Artificial<br/>Análise Multivariada</div>
  <div class="tit">Modelagem Estatística Multivariada<br/>e Avaliação de Performance</div>
  <div class="sub">Relatório técnico – Comportamento, satisfação e padrão financeiro<br/>de clientes de uma rede de varejo</div>
  <div class="info">Aluno(a): ____________________________<br/>Novembro de 2026</div>
</div>
<div class="sumario">
<h1>Sumário</h1>
<ol>
<li>Base de dados e visão geral</li>
<li>Parte I – Comparação do Score de Satisfação entre regiões (ANOVA)</li>
<li>Parte II – Efeito conjunto da Região sobre Gasto Online, Gasto Loja e NPS (MANOVA)</li>
<li>Parte II (Questão 3) – Função discriminante para o Status de Fidelidade</li>
<li>Parte III – Mapeamento perceptual dos perfis (Escalonamento Multidimensional)</li>
<li>Parte IV – Relação entre Engajamento Operacional e Retorno Financeiro/Percepção (Correlação Canônica)</li>
<li>Síntese para a tomada de decisão</li>
</ol>
<p>Cada seção apresenta: <b>Metodologia</b> (justificativa da escolha do método), <b>Validação</b> (testes de pressupostos),
<b>Resultados e Tabelas</b> e <b>Interpretação Negocial</b>.</p>
</div>
"""

import re as _re
_corpo = "".join(H)
_ordem = []
for _m in _re.finditer(r"Tabela (\d+[a-z]?) –", _corpo):
    if _m.group(1) not in _ordem:
        _ordem.append(_m.group(1))
_mapa = {k: str(i + 1) for i, k in enumerate(_ordem)}
_corpo = _re.sub(r"Tabela (\d+[a-z]?)\b", lambda m: "Tabela " + _mapa.get(m.group(1), m.group(1)), _corpo)
H = [_corpo]
html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Trabalho de Análise Multivariada</title>
<style>{CSS}</style></head><body>{capa}{''.join(H)}</body></html>"""
(AQUI / "relatorio_trabalho_am.html").write_text(html, encoding="utf-8")
from weasyprint import HTML
HTML(string=html).write_pdf(AQUI / "relatorio_trabalho_am.pdf")
print("PDF gerado:", AQUI / "relatorio_trabalho_am.pdf")
