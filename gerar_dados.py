"""
Geração de dataset sintético com distribuições baseadas nos
dados reais publicados pelo IBGE (Censo 2022, PNAD Contínua 2023).
Referências: IBGE (2023), PNUD Atlas do Desenvolvimento Humano.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

REGIOES = {
    "Norte":        {"n": 80,  "renda": (950,  350),  "analfab": (12.5, 4.5), "exp_vida": (73.2, 2.0),
                     "urban": (72, 14), "mort_inf": (19.5, 5.0), "pib": (22, 9),
                     "agua": (72, 12), "esgoto": (25, 12), "internet": (72, 10),
                     "desemprego": (11.5, 3.5), "gini": (0.52, 0.05)},
    "Nordeste":     {"n": 180, "renda": (850,  300),  "analfab": (15.2, 5.0), "exp_vida": (73.8, 1.8),
                     "urban": (69, 15), "mort_inf": (17.8, 5.5), "pib": (19, 8),
                     "agua": (77, 13), "esgoto": (38, 15), "internet": (72, 11),
                     "desemprego": (13.2, 4.0), "gini": (0.53, 0.05)},
    "Centro-Oeste": {"n": 50,  "renda": (1800, 450),  "analfab": (5.8,  2.5), "exp_vida": (75.5, 1.5),
                     "urban": (88, 8),  "mort_inf": (12.5, 3.5), "pib": (45, 18),
                     "agua": (89, 7),  "esgoto": (60, 15), "internet": (83, 8),
                     "desemprego": (8.5, 2.5),  "gini": (0.50, 0.04)},
    "Sudeste":      {"n": 130, "renda": (2200, 600),  "analfab": (4.5,  2.0), "exp_vida": (76.8, 1.5),
                     "urban": (93, 6),  "mort_inf": (11.2, 3.0), "pib": (55, 25),
                     "agua": (94, 5),  "esgoto": (78, 12), "internet": (88, 7),
                     "desemprego": (9.8, 3.0),  "gini": (0.49, 0.04)},
    "Sul":          {"n": 60,  "renda": (2100, 500),  "analfab": (3.8,  1.8), "exp_vida": (77.2, 1.4),
                     "urban": (86, 9),  "mort_inf": (10.5, 2.8), "pib": (52, 20),
                     "agua": (93, 5),  "esgoto": (72, 13), "internet": (87, 7),
                     "desemprego": (7.2, 2.2),  "gini": (0.46, 0.03)},
}

PORTE_LIMIARES = [(20000, "Pequeno"), (100000, "Médio"), (float("inf"), "Grande")]

rows = []
for regiao, p in REGIOES.items():
    n = p["n"]
    pop = np.random.lognormal(mean=9.5, sigma=1.2, size=n).astype(int)
    pop = np.clip(pop, 3000, 12_000_000)

    renda      = np.random.normal(*p["renda"], n).clip(300, 8000)
    analfab    = np.random.normal(*p["analfab"], n).clip(0, 40)
    exp_vida   = np.random.normal(*p["exp_vida"], n).clip(65, 82)
    urban      = np.random.normal(*p["urban"], n).clip(10, 100)
    mort_inf   = np.random.normal(*p["mort_inf"], n).clip(3, 45)
    pib        = np.random.normal(*p["pib"], n).clip(5, 200)
    agua       = np.random.normal(*p["agua"], n).clip(20, 100)
    esgoto     = np.random.normal(*p["esgoto"], n).clip(5, 100)
    internet   = np.random.normal(*p["internet"], n).clip(20, 98)
    desemprego = np.random.normal(*p["desemprego"], n).clip(2, 30)
    gini       = np.random.normal(*p["gini"], n).clip(0.30, 0.70)

    # correlações realistas entre variáveis
    delta = (renda - p["renda"][0]) / p["renda"][1]
    exp_vida   += delta * 0.5
    analfab    -= delta * 1.2
    esgoto     += delta * 5
    internet   += delta * 4
    mort_inf   -= delta * 1.5
    exp_vida    = exp_vida.clip(65, 82)
    analfab     = analfab.clip(0, 40)
    esgoto      = esgoto.clip(5, 100)
    internet    = internet.clip(20, 98)
    mort_inf    = mort_inf.clip(3, 45)

    for i in range(n):
        porte = next(r for lim, r in PORTE_LIMIARES if pop[i] < lim)
        rows.append({
            "municipio":          f"{regiao[:2].upper()}{i+1:03d}",
            "regiao":             regiao,
            "populacao":          int(pop[i]),
            "porte":              porte,
            "renda_per_capita":   round(renda[i], 1),
            "taxa_analfabetismo": round(analfab[i], 2),
            "expectativa_vida":   round(exp_vida[i], 1),
            "taxa_urbanizacao":   round(urban[i], 1),
            "mortalidade_infantil": round(mort_inf[i], 1),
            "pib_per_capita":     round(pib[i], 1),
            "acesso_agua":        round(agua[i], 1),
            "acesso_esgoto":      round(esgoto[i], 1),
            "acesso_internet":    round(internet[i], 1),
            "taxa_desemprego":    round(desemprego[i], 1),
            "gini":               round(gini[i], 3),
        })

df = pd.DataFrame(rows)

# IDH sintético (0.3–0.9), correlacionado com renda e educação
idhm_r = ((df["renda_per_capita"] - 300) / 7700) ** (1/3)
idhm_e = 1 - df["taxa_analfabetismo"] / 40
idhm_l = (df["expectativa_vida"] - 65) / 17
df["idhm"] = (idhm_r * idhm_e * idhm_l) ** (1/3)
df["idhm"] = df["idhm"].clip(0.3, 0.9).round(3)

# Faixa IDH para análise de correspondência
def faixa_idhm(v):
    if v < 0.500: return "Muito Baixo"
    if v < 0.600: return "Baixo"
    if v < 0.700: return "Médio"
    if v < 0.800: return "Alto"
    return "Muito Alto"

df["faixa_idhm"] = df["idhm"].apply(faixa_idhm)

df.to_csv("dados/municipios_brasil_2022.csv", index=False)
print(f"Dataset salvo: {len(df)} municípios, {df.shape[1]} variáveis")
print(df[["regiao","renda_per_capita","taxa_analfabetismo","idhm"]].groupby("regiao").mean().round(2))
