import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from math import sqrt
from scipy.stats import norm

st.set_page_config(page_title="Escanteios FT – Min 60", layout="wide")

ARQUIVO = "escanteios.csv"
MIN_AMOSTRA = 100

# ======================
# CARREGAR BANCO
# ======================
try:
    df = pd.read_csv(ARQUIVO)
except:
    df = pd.DataFrame(
        columns=[
            "data",
            "campeonato",
            "jogo",
            "tipo",
            "linha",
            "odd",
            "stake",
            "ritmo",
            "confianca",
            "resultado",
        ]
    )

# ======================
# FUNÇÕES
# ======================
def calcular_lucro(row):
    if row["resultado"] == "Win":
        return (row["odd"] - 1) * row["stake"]
    elif row["resultado"] == "Loss":
        return -row["stake"]
    return 0


def analise_segmentada(df, coluna):
    resumo = []

    for valor in df[coluna].unique():
        temp = df[df[coluna] == valor]
        n = len(temp)

        if n < 20:
            continue

        lucro = temp["lucro"].sum()
        stake_total = temp["stake"].sum()
        roi = lucro / stake_total if stake_total > 0 else 0
        winrate = (temp["resultado"] == "Win").mean()

        resumo.append([valor, n, roi, winrate])

    return pd.DataFrame(
        resumo, columns=[coluna, "Entradas", "ROI", "Winrate"]
    )


# ======================
# ABAS
# ======================
aba_entrada, aba_visao, aba_analise = st.tabs(
    ["📥 Entradas", "📊 Visão Geral", "🔬 Análise Profunda"]
)

# ======================
# ABA 1 — ENTRADAS
# ======================
with aba_entrada:
    st.subheader("➕ Nova Entrada – Escanteios FT (Min 60)")

    data = st.date_input("Data", date.today())
    campeonato = st.text_input("Campeonato")
    jogo = st.text_input("Jogo", value="—")

    tipo = st.selectbox("Tipo", ["Over", "Under"])
    linha = st.number_input("Linha", step=0.5)
    odd = st.number_input("Odd", step=0.01)
    stake = st.number_input("Stake", value=1.0)

    ritmo = st.selectbox(
        "Contexto do Jogo (min 60)",
        [
            "Jogo equilibrado",
            "Favorito perdendo",
            "Favorito empatando",
            "Favorito ganhando",
            "Favorito ganhando bem",
        ],
    )

    confianca = st.selectbox("Confiança", ["Alta", "Média", "Baixa"])
    resultado = st.selectbox("Resultado", ["Win", "Loss", "Void"])

    if st.button("Salvar Entrada"):
        nova = pd.DataFrame(
            [
                [
                    data,
                    campeonato,
                    jogo,
                    tipo,
                    linha,
                    odd,
                    stake,
                    ritmo,
                    confianca,
                    resultado,
                ]
            ],
            columns=df.columns,
        )

        df = pd.concat([df, nova], ignore_index=True)
        df.to_csv(ARQUIVO, index=False)

        st.success("Entrada salva com sucesso!")

    st.subheader("📄 Banco de Dados")
    st.dataframe(df)

# ======================
# ABA 2 — VISÃO GERAL
# ======================
with aba_visao:
    if df.empty:
        st.info("Nenhuma entrada cadastrada.")
    else:
        df["lucro"] = df.apply(calcular_lucro, axis=1)

        lucro_total = df["lucro"].sum()
        stake_total = df["stake"].sum()
        roi = lucro_total / stake_total if stake_total > 0 else 0
        winrate = (df["resultado"] == "Win").mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Entradas", len(df))
        col2.metric("ROI (%)", f"{roi*100:.2f}")
        col3.metric("Winrate (%)", f"{winrate*100:.2f}")

        df["banca"] = df["lucro"].cumsum()
        st.line_chart(df["banca"])

# ======================
# ABA 3 — ANÁLISE PROFUNDA
# ======================
with aba_analise:
    if len(df) < MIN_AMOSTRA:
        st.warning(
            f"Amostra insuficiente para análise profunda "
            f"({len(df)}/{MIN_AMOSTRA})"
        )
    else:
        df["lucro"] = df.apply(calcular_lucro, axis=1)

        total = len(df)
        winrate = (df["resultado"] == "Win").mean()
        odd_media = df["odd"].mean()
        breakeven = 1 / odd_media if odd_media > 0 else 0
        ev = df["lucro"].mean()

        # Teste Z
        se = sqrt(winrate * (1 - winrate) / total)
        z = (winrate - breakeven) / se if se > 0 else 0
        p_valor = 1 - norm.cdf(z)

        # Drawdown
        df["banca"] = df["lucro"].cumsum()
        df["max_banca"] = df["banca"].cummax()
        drawdown = (df["banca"] - df["max_banca"]).min()

        st.subheader("🔬 Validação Estatística do Método")

        col1, col2, col3 = st.columns(3)
        col1.metric("Entradas", total)
        col2.metric("EV médio", f"{ev:.4f}")
        col3.metric("Drawdown Máx.", f"{drawdown:.2f}")

        st.write(
            f"""
            **Winrate real:** {winrate*100:.2f}%  
            **Winrate mínimo (break-even):** {breakeven*100:.2f}%  
            **Odd média:** {odd_media:.2f}  
            **p-valor:** {p_valor:.4f}
            """
        )

        if p_valor < 0.05 and ev > 0:
            st.success("🟢 Método estatisticamente VALIDADO")
        elif ev > 0:
            st.warning("🟡 EV positivo, mas ainda pode ser variância")
        else:
            st.error("🔴 Método NEGATIVO no longo prazo")

        st.divider()

        st.subheader("📊 Análises Segmentadas")

        st.write("**Over vs Under**")
        st.dataframe(analise_segmentada(df, "tipo"))

        st.write("**Por Linha**")
        st.dataframe(analise_segmentada(df, "linha"))

        st.write("**Por Contexto de Jogo**")
        st.dataframe(analise_segmentada(df, "ritmo"))

        st.write("**Por Confiança**")
        st.dataframe(analise_segmentada(df, "confianca"))
