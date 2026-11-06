"""
Geração da base sintética do trabalho, reproduzindo em Python a lógica do
DATA step SAS fornecido no enunciado (300 clientes, semente 12345).

Observação: o gerador de números aleatórios do Python (NumPy) não é o mesmo do
SAS; portanto os valores individuais diferem do arquivo gerado no SAS, mas as
distribuições, os efeitos de região/fidelidade e a estrutura de correlação são
idênticos aos especificados.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def sas_round(x, unit=1.0):
    # ROUND do SAS: arredonda metade para longe de zero
    return np.sign(x) * np.floor(np.abs(x) / unit + 0.5) * unit

def gerar(seed=12345, n=300):
    rng = np.random.default_rng(seed)
    linhas = []
    for i in range(1, n + 1):
        regiao = int(rng.choice([1, 2, 3], p=[0.33, 0.33, 0.34]))
        status = int(rng.random() < 0.4)
        shift_reg = {1: 50, 2: 150, 3: 100}[regiao]
        shift_fid = 200 if status == 1 else 0

        e1 = rng.normal(0, 15)
        e2 = rng.normal(0, 15)
        e3 = rng.normal(0, 2)

        tempo = sas_round(abs(rng.normal(24, 10)) + 1)
        freq_on = sas_round(abs(rng.normal(12, 4) + tempo * 0.2))
        freq_lj = sas_round(abs(rng.normal(8, 3)))

        gasto_on = sas_round(300 + freq_on * 35 + shift_reg + shift_fid + e1, 0.01)
        gasto_lj = sas_round(200 + freq_lj * 40 + shift_reg * 0.5 + e2, 0.01)

        score = sas_round(min(10, max(1, 6 + regiao * 0.8 + rng.normal(0, 1))), 0.1)
        nps = sas_round(min(100, max(0, 50 + score * 4 + status * 15 + e3)))

        linhas.append(dict(ID=i, Regiao=regiao, Status_Fidelidade=status,
                           Gasto_Online=round(gasto_on, 2), Gasto_Loja=round(gasto_lj, 2),
                           Freq_Online=int(freq_on), Freq_Loja=int(freq_lj),
                           Score_Satisfacao=round(score, 1), Tempo_Relacao=int(tempo),
                           NPS_Pontuacao=int(nps)))
    return pd.DataFrame(linhas)

if __name__ == "__main__":
    df = gerar()
    out = Path(__file__).with_name("dados_clientes.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} clientes salvos em {out}")
    print(df.describe().round(2).T)
