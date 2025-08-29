import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from unidecode import unidecode
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import unicodedata
from difflib import get_close_matches
from funcoes import *
import ipywidgets as widgets
from scipy.optimize import curve_fit
import scipy.stats as stats8
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import scipy.optimize
from scipy.stats import norm
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize







def _normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # remove acentos
    s = s.strip().lower()
    # uniformiza separadores e espaços
    for ch in ["-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())  # colapsa espaços
    return s

def carregar_arquivo(
    nome_base,
    pasta,
    sheet_name=0,
    extensoes=('.xlsx', '.xls', '.xlsm', '.csv'),
    search_subdirs=False,
    match_mode="fuzzy",  # "exact" | "startswith" | "contains" | "fuzzy"
):
    """
    Carrega um arquivo cujo nome (sem extensão) corresponda a `nome_base`
    segundo `match_mode`. Se `search_subdirs=True`, varre subpastas.
    """

    if not os.path.isdir(pasta):
        raise FileNotFoundError(f"Pasta não existe: {pasta}")

    # Coleta candidatos
    candidatos = []
    if search_subdirs:
        for root, _, files in os.walk(pasta):
            for f in files:
                if f.lower().endswith(tuple(e.lower() for e in extensoes)):
                    candidatos.append(Path(root) / f)
    else:
        for f in os.listdir(pasta):
            if f.lower().endswith(tuple(e.lower() for e in extensoes)):
                candidatos.append(Path(pasta) / f)

    if not candidatos:
        raise FileNotFoundError(f"Nenhum arquivo com extensões {extensoes} encontrado em {pasta}.")

    alvo_norm = _normalize_name(Path(str(nome_base)).stem)
    # prepara lista (stem normalizado -> Path)
    stems_norm = { _normalize_name(p.stem): p for p in candidatos }

    escolhido = None
    if match_mode == "exact":
        escolhido = stems_norm.get(alvo_norm)
    elif match_mode == "startswith":
        for stem, pth in stems_norm.items():
            if stem.startswith(alvo_norm):
                escolhido = pth
                break
    elif match_mode == "contains":
        for stem, pth in stems_norm.items():
            if alvo_norm in stem:
                escolhido = pth
                break
    else:  # fuzzy (padrão)
        # tenta startswith/contains primeiro (mais previsível), depois fuzzy
        for stem, pth in stems_norm.items():
            if stem.startswith(alvo_norm):
                escolhido = pth
                break
        if escolhido is None:
            for stem, pth in stems_norm.items():
                if alvo_norm in stem:
                    escolhido = pth
                    break
        if escolhido is None:
            # difflib para pegar o mais próximo
            match = get_close_matches(alvo_norm, list(stems_norm.keys()), n=1, cutoff=0.6)
            if match:
                escolhido = stems_norm[match[0]]

    if escolhido is None:
        # monta ajuda de diagnóstico
        lista = "\n  - " + "\n  - ".join(sorted(p.name for p in candidatos))
        raise FileNotFoundError(
            f"Arquivo com base '{nome_base}' não encontrado em {pasta}.\n"
            f"Arquivos candidatos encontrados:{lista}\n"
            f"Dica: tente match_mode='contains' ou 'fuzzy', ou ajuste o nome_base."
        )

    ext = escolhido.suffix.lower()

    try:
        if ext == '.csv':
            try:
                df = pd.read_csv(escolhido)
            except UnicodeDecodeError:
                df = pd.read_csv(escolhido, encoding='latin1')
        else:
            df = pd.read_excel(escolhido, sheet_name=sheet_name)

        df.columns = [str(c).strip() for c in df.columns]
        print(f"Arquivo '{escolhido.name}' carregado com sucesso de: {escolhido.parent}")
        return df

    except Exception as e:
        raise RuntimeError(f"Erro ao carregar '{escolhido}': {e}")







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize

# ==========================================================
# 1) Parametrização por ano
# ==========================================================
def parametrizacao(df, 
                   parametro='MASSA DO VOLUME (kg)', 
                   confianca=0.95,
                   floc=False, 
                   completo=True,
                   dist='gamma'):
    df = df.copy()

    # garante coluna Ano
    if 'Ano' not in df.columns and 'ANO DE GERAÇÃO (DATA SIGRE)' in df.columns:
        df['Ano'] = pd.to_datetime(df['ANO DE GERAÇÃO (DATA SIGRE)'],
                                   errors='coerce').dt.year.astype('Int64')

    resultados = []
    for ano in sorted(df['Ano'].dropna().unique()):
        dados = df.loc[df['Ano'] == ano, parametro].astype(float).dropna()
        if floc:
            dados = dados[dados > 0]
        if len(dados) < 5:
            continue

        # === ESCOLHA DA DISTRIBUIÇÃO ===
        if dist == 'gamma':
            shape, loc, scale = st.gamma.fit(dados, floc=0) if floc else st.gamma.fit(dados)
            if completo:
                mean = st.gamma.mean(shape, loc, scale)
                std  = st.gamma.std(shape, loc, scale)
                conf_interval = st.gamma.interval(confianca, shape, loc, scale)

        elif dist in ('lognorm', 'lognormal'):
            shape, loc, scale = st.lognorm.fit(dados, floc=0) if floc else st.lognorm.fit(dados)
            if completo:
                mean = st.lognorm.mean(shape, loc, scale)
                std  = st.lognorm.std(shape, loc, scale)
                conf_interval = st.lognorm.interval(confianca, shape, loc, scale)

        else:
            raise ValueError("dist deve ser 'gamma' ou 'lognorm'")

        # === MONTA RESULTADO ===
        if completo:
            resultados.append({
                'Ano': ano,
                'Shape': shape,
                'Loc': loc,
                'Scale': scale,
                'Mean': mean,
                'Std': std,
                'Conf_Interval': conf_interval,
                'Total_obs': dados.sum(),
                'N': len(dados),
                'min_obs': dados.min(),
                'max_obs': dados.max()
            })
        else:
            resultados.append({
                'Ano': ano,
                'Shape': shape,
                'Loc': loc,
                'Scale': scale,
                'N': len(dados)
            })

    return pd.DataFrame(resultados)


# ==========================================================
# 2) Ajuste de regressão nos parâmetros
# ==========================================================
def regressao_parametros(df,
                         parametro='MASSA DO VOLUME (kg)',
                         regressao='linear',
                         reg_bounds=True,                         
                         floc=True,
                         min_n=5,
                         plotar=True,
                         dist='gamma'):

    df = df.copy()

    # cria coluna Ano se não existir
    if 'Ano' not in df.columns and 'ANO DE GERAÇÃO (DATA SIGRE)' in df.columns:
        df['Ano'] = pd.to_datetime(df['ANO DE GERAÇÃO (DATA SIGRE)'],
                                   errors='coerce').dt.year.astype('Int64')

    # remove zeros se floc=True
    if floc:
        df = df[df[parametro] > 0]

    # roda parametrização ano a ano
    params_df = parametrizacao(df,                               
                               parametro=parametro,
                               confianca=0.95,
                               floc=floc,
                               completo=True,
                               dist=dist)

    # aplica filtro de min_n
    counts = df.groupby('Ano')[parametro].count()
    anos_validos = counts[counts >= min_n].index
    params_df = params_df[params_df['Ano'].isin(anos_validos)].sort_values('Ano')

    if params_df.empty:
        return None, None, None, None, params_df

    # extrai séries
    anos = params_df['Ano'].values.astype(float)
    alpha_list = params_df['Shape'].values.astype(float)
    scale_list = params_df['Scale'].values.astype(float)
    loc_list   = params_df['Loc'].values.astype(float)

    # define função de regressão
    if regressao == 'linear':
        def fit_func(x, a, b): return a*x + b
    elif regressao == 'polinomial':
        def fit_func(x, a, b, c): return a*x**2 + b*x + c
    elif regressao == 'exponencial':
        def fit_func(x, a, b, c): return a*np.exp(b*x) + c
    else:
        raise ValueError("regressao deve ser 'linear', 'polinomial' ou 'exponencial'")

    # normaliza anos
    anos_norm = anos - anos.min()

    # bounds específicos
    if reg_bounds:
        if regressao == 'linear':
            bounds_alpha = ([0, -np.inf], [np.inf, np.inf])
            bounds_scale = ([0, -np.inf], [np.inf, np.inf])
            bounds_loc   = ([-np.inf, -np.inf], [np.inf, np.inf])
        else:  # polinomial ou exponencial
            bounds_alpha = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            bounds_scale = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            bounds_loc   = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    else:
        bounds_alpha = bounds_scale = bounds_loc = (-np.inf, np.inf)

    # ajustes
    popt_alpha, _ = scipy.optimize.curve_fit(fit_func, anos_norm, alpha_list, bounds=bounds_alpha)
    popt_scale, _ = scipy.optimize.curve_fit(fit_func, anos_norm, scale_list, bounds=bounds_scale)    
    popt_loc, _   = scipy.optimize.curve_fit(fit_func, anos_norm, loc_list,   bounds=bounds_loc)

    print("Coeficientes da regressão:")
    print("  Shape:", popt_alpha)
    print("  Scale:", popt_scale)
    print("  Loc:", popt_loc)

    popt_mean = None  # redundante

    # plot
    if plotar:
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        label_shape = 'Shape (α)' if dist == 'gamma' else 'Shape (σ no log / s)'
        label_scale = 'Scale (θ)' if dist == 'gamma' else 'Scale (mediana)'

        axs[0].plot(anos.astype(int), alpha_list, 'o-', label=f'{label_shape} observado')
        axs[0].plot(anos.astype(int), fit_func(anos_norm, *popt_alpha), 'r--', label=f'{label_shape} ajustado')
        axs[0].set_title(label_shape); axs[0].grid(True); axs[0].legend()

        axs[1].plot(anos.astype(int), scale_list, 'o-', label=f'{label_scale} observado')
        axs[1].plot(anos.astype(int), fit_func(anos_norm, *popt_scale), 'r--', label=f'{label_scale} ajustado')
        axs[1].set_title(label_scale); axs[1].grid(True); axs[1].legend()

        axs[2].plot(anos.astype(int), loc_list, 'o-', label='Loc observado')
        axs[2].plot(anos.astype(int), fit_func(anos_norm, *popt_loc), 'r--', label='Loc ajustado')
        axs[2].set_title('Loc'); axs[2].grid(True); axs[2].legend()

        plt.suptitle(f"Evolução dos parâmetros {dist} | Regressão: {regressao}")
        plt.show()

    return popt_alpha, popt_scale, popt_loc, popt_mean, params_df


# ==========================================================
# 3) Previsão no tempo (média e IC)
# ==========================================================
def prever_media(params_df, anos_futuros=5, regressao='linear', plotar=True, dist='gamma'):
    """
    Projeta os parâmetros da distribuição no tempo e,
    a partir deles, calcula a média prevista + IC 95%.
    """

    anos = params_df['Ano'].values.astype(float)
    alpha_list = params_df['Shape'].values.astype(float)
    scale_list = params_df['Scale'].values.astype(float)
    loc_list   = params_df['Loc'].values.astype(float)

    # normaliza anos
    anos_norm = anos - anos.min()
    anos_proj = np.arange(0, anos_norm.max() + anos_futuros + 1)
    anos_reais = anos.min() + anos_proj

    # --- define função de regressão ---
    if regressao == 'linear':
        def fit_func(x, a, b): return a*x + b
        bounds2 = ([0, -np.inf], [np.inf, np.inf])
        bounds_loc = ([-np.inf, -np.inf], [np.inf, np.inf])
    elif regressao == 'exponencial':
        def fit_func(x, a, b, c): return a * np.exp(b*x) + c
        bounds2 = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        bounds_loc = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    elif regressao == 'polinomial':
        def fit_func(x, a, b, c): return a*x**2 + b*x + c
        bounds2 = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        bounds_loc = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    else:
        raise ValueError("regressao deve ser 'linear', 'exponencial' ou 'polinomial'")

    # --- regressão dos parâmetros ---
    popt_shape, _ = scipy.optimize.curve_fit(fit_func, anos_norm, alpha_list, bounds=bounds2, maxfev=20000)
    pred_shape = fit_func(anos_proj, *popt_shape)

    popt_scale, _ = scipy.optimize.curve_fit(fit_func, anos_norm, scale_list, bounds=bounds2, maxfev=20000)
    pred_scale = fit_func(anos_proj, *popt_scale)

    if np.allclose(loc_list, 0):
        pred_loc = np.zeros_like(anos_proj)
        popt_loc = [0]
    else:
        popt_loc, _ = scipy.optimize.curve_fit(fit_func, anos_norm, loc_list, bounds=bounds_loc, maxfev=20000)
        pred_loc = fit_func(anos_proj, *popt_loc)

    # debug
    print("Coeficientes previsão:")
    print("  Shape:", popt_shape)
    print("  Scale:", popt_scale)
    print("  Loc:", popt_loc)

    # --- calcula médias e ICs ---
    mean, ic_inf, ic_sup = [], [], []
    for a, s, l in zip(pred_shape, pred_scale, pred_loc):
        if dist == 'gamma':
            mean.append(st.gamma.mean(a, l, s))
            ci = st.gamma.interval(0.95, a, l, s)
        elif dist in ('lognorm', 'lognormal'):
            mean.append(st.lognorm.mean(a, loc=l, scale=s))
            ci = st.lognorm.interval(0.95, a, loc=l, scale=s)
        ic_inf.append(ci[0])
        ic_sup.append(ci[1])

    df_pred = pd.DataFrame({
        "Ano": anos_reais,
        "Shape": pred_shape,
        "Scale": pred_scale,
        "Loc": pred_loc,
        "Media_Prevista": mean,
        "IC_inferior": ic_inf,
        "IC_superior": ic_sup
    })

    # --- gráfico ---
    if plotar:
        plt.figure(figsize=(10,6))
        mean_obs = params_df['Mean'].values
        plt.plot(params_df['Ano'], mean_obs, 'o-', color="blue", label="Média observada")
        plt.fill_between(params_df['Ano'], params_df['min_obs'], params_df['max_obs'],
                         color="blue", alpha=0.2, label="Min/Max observado")
        plt.plot(df_pred["Ano"], df_pred["Media_Prevista"], 'b--', label="Média prevista")
        plt.fill_between(df_pred["Ano"], df_pred["IC_inferior"], df_pred["IC_superior"],
                         color="red", alpha=0.2, label="IC 95% previsto")
        plt.xlabel("Ano"); plt.ylabel("Massa média (kg)")
        plt.title(f"Média por tonel - Observado vs Previsto ({dist}, {regressao})")
        plt.legend(); plt.grid(True)
        plt.show()

    return df_pred


def testar_distribuicoes(df, parametro='MASSA DO VOLUME (kg)', locais=None, anos=None, min_n=10):
    """
    Testa diferentes distribuições (Gamma, LogNormal, Weibull, Normal) 
    para cada Local de Geração e Ano, comparando log-likelihood, AIC e KS test.

    Retorna um DataFrame com resultados.
    """
    if 'Ano' not in df.columns and 'ANO DE GERAÇÃO (DATA SIGRE)' in df.columns:

        df['Ano'] = pd.to_datetime(df['ANO DE GERAÇÃO (DATA SIGRE)'],
                               errors='coerce').dt.year.astype('Int64')

    if locais is None:
        locais = df['LOCAL DA GERAÇÃO'].unique()
    if anos is None:
        anos = df['Ano'].unique()

    resultados = []

    for local in locais:
        for ano in anos:
            dados = df[(df['LOCAL DA GERAÇÃO']==local) & (df['Ano']==ano)][parametro].dropna()
            if len(dados) < min_n:
                continue  # ignora grupos com poucos registros

            # --- Ajuste Gamma ---
            try:
                alpha, loc, scale = st.gamma.fit(dados, floc=0)
                ll = np.sum(st.gamma.logpdf(dados, alpha, loc=loc, scale=scale))
                aic = 2*2 - 2*ll
                ks_p = st.kstest(dados, "gamma", args=(alpha, loc, scale)).pvalue
                resultados.append([local, ano, "Gamma", ll, aic, ks_p])
            except:
                pass

            # --- Ajuste LogNormal ---
            try:
                s, loc, scale = st.lognorm.fit(dados, floc=0)
                ll = np.sum(st.lognorm.logpdf(dados, s, loc=loc, scale=scale))
                aic = 2*2 - 2*ll
                ks_p = st.kstest(dados, "lognorm", args=(s, loc, scale)).pvalue
                resultados.append([local, ano, "LogNormal", ll, aic, ks_p])
            except:
                pass

            # --- Ajuste Weibull ---
            try:
                c, loc, scale = st.weibull_min.fit(dados, floc=0)
                ll = np.sum(st.weibull_min.logpdf(dados, c, loc=loc, scale=scale))
                aic = 2*2 - 2*ll
                ks_p = st.kstest(dados, "weibull_min", args=(c, loc, scale)).pvalue
                resultados.append([local, ano, "Weibull", ll, aic, ks_p])
            except:
                pass

            # --- Ajuste Normal ---
            try:
                mu, sigma = st.norm.fit(dados)
                ll = np.sum(st.norm.logpdf(dados, mu, sigma))
                aic = 2*2 - 2*ll
                ks_p = st.kstest(dados, "norm", args=(mu, sigma)).pvalue
                resultados.append([local, ano, "Normal", ll, aic, ks_p])
            except:
                pass

    resultados_df = pd.DataFrame(resultados, 
                                 columns=["Local", "Ano", "Distribuicao", "LogLikelihood", "AIC", "KS_p"])
    return resultados_df
def analisar_por_ano(df, unidade='FPSO-CST', parametro='MASSA DO VOLUME (kg)',
                     dist='gamma', min_n=6, ncols=3, bins=30):
    """
    Ajusta e plota uma distribuição escolhida ('gamma','lognorm','weibull','norm')
    para cada ano de uma unidade.
    Retorna DataFrame com parâmetros e métricas (LogLik, AIC, KS_p).
    """
    # garante coluna Ano
    if 'Ano' not in df.columns:
        if 'AnoMes' in df.columns:
            df['Ano'] = pd.to_datetime(df['AnoMes'], errors='coerce').dt.year.astype('Int64')
        elif 'ANO DE GERAÇÃO (DATA SIGRE)' in df.columns:
            df['Ano'] = pd.to_datetime(df['ANO DE GERAÇÃO (DATA SIGRE)'], errors='coerce').dt.year.astype('Int64')
        else:
            raise ValueError("Coluna 'Ano' não encontrada.")

    df = df[df['LOCAL DA GERAÇÃO'] == unidade].dropna(subset=[parametro,'Ano'])
    df = df[df[parametro] > 0]

    anos = sorted(df['Ano'].unique())
    n = len(anos)
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,3.5*nrows))
    axes = axes.flatten()

    resultados = []

    for i, ano in enumerate(anos):
        ax = axes[i]
        data = df.loc[df['Ano']==ano, parametro].astype(float).dropna().values
        if len(data) < min_n:
            ax.axis("off")
            continue

        # Ajuste e label
        if dist=='gamma':
            params = st.gamma.fit(data, floc=0)
            pdf = st.gamma.pdf
            label = f"Gamma (α={params[0]:.2f}, β={params[2]:.2f})"
            ll = np.sum(st.gamma.logpdf(data, *params))
            pks = st.kstest(data, 'gamma', args=params).pvalue
        elif dist=='lognorm':
            params = st.lognorm.fit(data, floc=0)
            pdf = st.lognorm.pdf
            label = f"LogNorm (σ={params[0]:.2f}, scale={params[2]:.2f})"
            ll = np.sum(st.lognorm.logpdf(data, *params))
            pks = st.kstest(data, 'lognorm', args=params).pvalue
        elif dist in ('weibull','weibull_min'):
            params = st.weibull_min.fit(data, floc=0)
            pdf = st.weibull_min.pdf
            label = f"Weibull (c={params[0]:.2f}, λ={params[2]:.2f})"
            ll = np.sum(st.weibull_min.logpdf(data, *params))
            pks = st.kstest(data, 'weibull_min', args=params).pvalue
        elif dist in ('norm','normal'):
            params = st.norm.fit(data)
            pdf = st.norm.pdf
            label = f"Normal (µ={params[0]:.2f}, σ={params[1]:.2f})"
            ll = np.sum(st.norm.logpdf(data, *params))
            pks = st.kstest(data, 'norm', args=params).pvalue
        else:
            raise ValueError("Distribuição não suportada")

        # AIC
        k = len(params)
        aic = 2*k - 2*ll

        # Curva ajustada
        x = np.linspace(data.min(), data.max(), 200)
        y = pdf(x, *params)

        # Plot
        ax.hist(data, bins=bins, density=True, alpha=0.5, label="Dados")
        ax.plot(x, y, 'r-', label=label)
        ax.set_title(f"Ano {ano}, KS p={pks:.3f}")
        ax.legend(fontsize=7); ax.grid(True)

        resultados.append({"Ano":ano,"Dist":dist,"Params":params,
                           "LogLik":ll,"AIC":aic,"KS_p":pks})

    # remove subplots extras
    for j in range(i+1,len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{unidade} - Ajuste {dist}")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(resultados)


from scipy.stats import norm

# --- Modelos ---

# assintótico simples
def modelo_assintotico(x, L, a, b, x0):
    return L - a * np.exp(-b * (x - x0))

# piecewise linear
def modelo_piecewise(x, x0, k1, b1, k2, b2):
    return np.piecewise(
        x,
        [x < x0, x >= x0],
        [lambda x: k1*x + b1, lambda x: k2*x + b2]
    )

# logarítmico simples (x0 fixo = min(x))
def modelo_logaritmico(x, a, b, x0):
    return a + b * np.log(x - x0 + 1)

# piecewise logarítmico (x0 fixo = min(x))
def modelo_piecewise_log(x, x_c, a1, b1, a2, b2, x0):
    return np.piecewise(
        x,
        [x < x_c, x >= x_c],
        [
            lambda t: a1 + b1 * np.log(t - x0 + 1),
            lambda t: a2 + b2 * np.log(t - x_c + 1)
        ]
    )

# --- Função principal ---
def projetar_futuro(df, anos_futuros=None, confiança=0.95,
                    residuo_alvo='BORRA CAT I', tipo_modelo="linear", n_sim=1000):
    """
    Projeta o acúmulo de resíduos ao longo do tempo com diferentes modelos de ajuste
    (linear, assintótico, logarítmico, piecewise linear ou piecewise logarítmico),
    calculando intervalos de confiança e gerando gráficos opcionais para um resíduo alvo.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo os dados de inventário, com colunas:
        - 'ANO DE GERAÇÃO (DATA SIGRE)': datas ou anos das amostras
        - 'TIPO DE RESÍDUO': categoria do resíduo
        - 'MASSA DO VOLUME (kg)': massa em kg do resíduo
    anos_futuros : int ou None, opcional (default=None)
        Número de anos a projetar além do último ano observado.
        Se None, não projeta além do histórico.
    confiança : float, opcional (default=0.95)
        Nível de confiança para os intervalos (ex.: 0.95 = 95%).
    residuo_alvo : str, opcional (default='BORRA CAT I')
        Resíduo específico para o qual será exibido o gráfico com previsões e intervalos.
    tipo_modelo : {'linear', 'assintotico', 'logaritmico', 'piecewise', 'piecewise_log'}, opcional
        Modelo de ajuste a ser utilizado:
        - 'linear': regressão linear simples
        - 'assintotico': curva assintótica exponencial
        - 'logaritmico': ajuste logarítmico simples (x0 = min(x))
        - 'piecewise': regressão linear segmentada em dois trechos
        - 'piecewise_log': regressão logarítmica segmentada em dois trechos
    n_sim : int, opcional (default=1000)
        Número de simulações Monte Carlo para modelos não lineares (assintótico/log),
        usadas para estimar os intervalos de confiança.

    Retorno
    -------
    dfs_residuos : dict
        Dicionário com DataFrames ajustados para cada resíduo,
        contendo colunas ['Ano', 'Acumulado (kg)', 'yhat', 'Erro %'].
    resumo_modelos : pandas.DataFrame
        DataFrame resumo com os parâmetros do modelo ajustado para cada resíduo.
    projecoes_df : pandas.DataFrame
        DataFrame com projeções futuras (anos, valores previstos e intervalos).

    Notas
    -----
    - A função plota automaticamente gráficos para o `residuo_alvo` escolhido.
    - O cálculo dos intervalos varia conforme o modelo:
        * Linear: IC analítico via `statsmodels`.
        * Assintótico e logarítmico: IC via Monte Carlo a partir da matriz de covariância dos parâmetros.
        * Piecewise linear e log: IC via desvio padrão dos resíduos.
    """    
    
 
    
    
    df = df.copy()
    df['Ano'] = pd.to_datetime(df['ANO DE GERAÇÃO (DATA SIGRE)'],
                               errors='coerce').dt.year.astype('Int64')

    # somar por ano toda a massa
    somatorio_ano = df.groupby(['Ano', 'TIPO DE RESÍDUO'])['MASSA DO VOLUME (kg)'].sum().reset_index()
    somatorio_ano['Acumulado (kg)'] = somatorio_ano.groupby('TIPO DE RESÍDUO')['MASSA DO VOLUME (kg)'].cumsum()

    dfs_residuos = {}
    result = []
    projecoes = []

    for residuo in somatorio_ano['TIPO DE RESÍDUO'].unique():
        df_residuo = somatorio_ano[somatorio_ano['TIPO DE RESÍDUO'] == residuo].copy()

        if df_residuo.shape[0] > 1:
            x = df_residuo['Ano'].values.astype(float)
            y = df_residuo['Acumulado (kg)'].values.astype(float)

            # --- Ajuste por modelo ---
            if tipo_modelo == "linear":
                modelo = sm.OLS.from_formula('Q("Acumulado (kg)") ~ Ano', data=df_residuo).fit()
                df_residuo['yhat'] = modelo.fittedvalues
                result.append({
                    "Residuo": residuo,
                    "Intercepto": modelo.params['Intercept'],
                    "Coef_Ano": modelo.params['Ano'],
                    "R2": modelo.rsquared,
                    "p-valor": modelo.pvalues['Ano']
                })

            elif tipo_modelo == "assint":
                x0 = x.min()
                p0 = [y.max()*1.1, y.max(), 0.1]
                params, cov = curve_fit(lambda t, L, a, b: modelo_assintotico(t, L, a, b, x0),
                                        x, y, p0=p0, maxfev=10000)
                df_residuo['yhat'] = modelo_assintotico(x, *params, x0)
                result.append({
                    "Residuo": residuo,
                    "Modelo": "Assintótico",
                    "Parametros": params
                })

            elif tipo_modelo == "piece":
                p0 = [x.mean(), 1, y.min(), 0.1, y.max()]
                params, cov = curve_fit(modelo_piecewise, x, y, p0=p0, maxfev=10000)
                df_residuo['yhat'] = modelo_piecewise(x, *params)
                result.append({
                    "Residuo": residuo,
                    "Modelo": "Piecewise Linear",
                    "Parametros": params
                })

            elif tipo_modelo == "log":
                x0 = x.min()
                p0 = [y.min(), (y.max()-y.min())/np.log(x.max()-x.min()+1)]
                params, cov = curve_fit(
                    lambda t, a, b: modelo_logaritmico(t, a, b, x0),
                    x, y, p0=p0, maxfev=20000
                )
                df_residuo['yhat'] = modelo_logaritmico(x, *params, x0)
                result.append({
                    "Residuo": residuo,
                    "Modelo": "Logarítmico",
                    "Parametros": (params[0], params[1], x0)
                })

            elif tipo_modelo == "piece_log":
                x0 = x.min()
                p0 = [x.mean(), y.min(), 1, y.min(), 1]
                params, cov = curve_fit(
                    lambda t, x_c, a1, b1, a2, b2: modelo_piecewise_log(t, x_c, a1, b1, a2, b2, x0),
                    x, y, p0=p0, maxfev=20000
                )
                df_residuo['yhat'] = modelo_piecewise_log(x, *params, x0)
                result.append({
                    "Residuo": residuo,
                    "Modelo": "Piecewise Logarítmico",
                    "Parametros": (*params, x0)
                })

            # erro percentual
            df_residuo['Erro %'] = abs(y - df_residuo['yhat']) / y * 100
            dfs_residuos[residuo] = df_residuo

        else:
            print(f'{residuo}: poucos dados')
            continue

        # --- Anos futuros ---
        max_ano = int(df_residuo['Ano'].max())
        anos_proj = list(range(max_ano+1, max_ano+anos_futuros+1)) if anos_futuros else []

        if anos_proj:
            x_proj = np.array(anos_proj, dtype=float)

            if tipo_modelo == "linear":
                pred = modelo.get_prediction(pd.DataFrame({"Ano": x_proj}))
                pred_ci = pred.summary_frame(alpha=1 - confiança)
                df_proj = pd.DataFrame({
                    "Ano": x_proj.astype(int),
                    "yhat": pred_ci['mean'],
                    "IC_inf": pred_ci['mean_ci_lower'],
                    "IC_sup": pred_ci['mean_ci_upper'],
                    "Residuo": residuo
                })

            elif tipo_modelo == "assint":
                try:
                    sims = []
                    for _ in range(n_sim):
                        sampled_params = np.random.multivariate_normal(params, cov)
                        sims.append(modelo_assintotico(x_proj, *sampled_params, x0))
                    sims = np.array(sims)
                    df_proj = pd.DataFrame({
                        "Ano": x_proj.astype(int),
                        "yhat": sims.mean(axis=0),
                        "IC_inf": np.percentile(sims, (1 - confiança)/2*100, axis=0),
                        "IC_sup": np.percentile(sims, (1 + confiança)/2*100, axis=0),
                        "Residuo": residuo
                    })
                except np.linalg.LinAlgError:
                    resid = y - df_residuo['yhat']
                    sigma = resid.std()
                    z = norm.ppf(1 - (1 - confiança) / 2)
                    y_proj = modelo_assintotico(x_proj, *params, x0)
                    df_proj = pd.DataFrame({
                        "Ano": x_proj.astype(int),
                        "yhat": y_proj,
                        "IC_inf": y_proj - z * sigma,
                        "IC_sup": y_proj + z * sigma,
                        "Residuo": residuo
                    })

            elif tipo_modelo == "piece":
                y_proj = modelo_piecewise(x_proj, *params)
                resid = y - df_residuo['yhat']
                sigma = resid.std()
                z = norm.ppf(1 - (1 - confiança) / 2)
                df_proj = pd.DataFrame({
                    "Ano": x_proj.astype(int),
                    "yhat": y_proj,
                    "IC_inf": y_proj - z * sigma,
                    "IC_sup": y_proj + z * sigma,
                    "Residuo": residuo
                })

            elif tipo_modelo == "log":
                try:
                    sims = []
                    for _ in range(n_sim):
                        sampled_params = np.random.multivariate_normal(params, cov)
                        sims.append(modelo_logaritmico(x_proj, *sampled_params, x0))
                    sims = np.array(sims)
                    df_proj = pd.DataFrame({
                        "Ano": x_proj.astype(int),
                        "yhat": sims.mean(axis=0),
                        "IC_inf": np.percentile(sims, (1 - confiança)/2*100, axis=0),
                        "IC_sup": np.percentile(sims, (1 + confiança)/2*100, axis=0),
                        "Residuo": residuo
                    })
                except np.linalg.LinAlgError:
                    resid = y - df_residuo['yhat']
                    sigma = resid.std()
                    z = norm.ppf(1 - (1 - confiança) / 2)
                    y_proj = modelo_logaritmico(x_proj, *params, x0)
                    df_proj = pd.DataFrame({
                        "Ano": x_proj.astype(int),
                        "yhat": y_proj,
                        "IC_inf": y_proj - z * sigma,
                        "IC_sup": y_proj + z * sigma,
                        "Residuo": residuo
                    })

            elif tipo_modelo == "piece_log":
                y_proj = modelo_piecewise_log(x_proj, *params, x0)
                resid = y - df_residuo['yhat']
                sigma = resid.std()
                z = norm.ppf(1 - (1 - confiança) / 2)
                df_proj = pd.DataFrame({
                    "Ano": x_proj.astype(int),
                    "yhat": y_proj,
                    "IC_inf": y_proj - z * sigma,
                    "IC_sup": y_proj + z * sigma,
                    "Residuo": residuo
                })

            projecoes.append(df_proj)

            # --- Gráfico se for resíduo alvo ---
            if residuo_alvo and residuo == residuo_alvo:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_residuo['Ano'], y=df_residuo['Acumulado (kg)'],
                                         mode='lines+markers', name='Observado'))
                fig.add_trace(go.Scatter(x=df_residuo['Ano'], y=df_residuo['yhat'],
                                         mode='lines+markers', name='Ajuste',
                                         line=dict(dash='dash', color='orange')))
                fig.add_trace(go.Scatter(x=df_residuo['Ano'], y=df_residuo['Erro %'],
                                         mode='lines+markers', name='Erro (%)',
                                         yaxis='y2', line=dict(color='red')))

                if not df_proj.empty:
                    anos_proj_ext = [df_residuo['Ano'].max()] + list(df_proj['Ano'])
                    yhat_proj_ext = [df_residuo['yhat'].iloc[-1]] + list(df_proj['yhat'])

                    fig.add_trace(go.Scatter(x=anos_proj_ext, y=yhat_proj_ext,
                                             mode='lines+markers', name='Projeção futura',
                                             line=dict(dash='dot', color='green')))

                    fig.add_trace(go.Scatter(
                        x=list(df_proj['Ano']) + list(df_proj['Ano'][::-1]),
                        y=list(df_proj['IC_sup']) + list(df_proj['IC_inf'][::-1]),
                        fill='toself',
                        fillcolor='rgba(0,128,0,0.2)',
                        line=dict(width=0),
                        name=f'IC {int(confiança*100)}% Futuro',
                        hovertemplate="<b>Ano:</b> %{x}<br><b>limite:</b> %{y:.2f}<extra></extra>"))

                fig.update_layout(
                    title=f"Previsão ({tipo_modelo}) e Erro (%) - {residuo}",
                    xaxis=dict(title="Ano"),
                    yaxis=dict(title="Acumulado (kg)", side="left"),
                    yaxis2=dict(title="Erro (%)", overlaying="y", side="right",
                                showgrid=False, tickformat=".1f"),
                    legend=dict(orientation="h", y=-0.2)
                )
                fig.show()

    resumo_modelos = pd.DataFrame(result)
    projecoes_df = pd.concat(projecoes, ignore_index=True) if projecoes else pd.DataFrame()

    return dfs_residuos, resumo_modelos, projecoes_df





