# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:16:19 2024

@author: efbitencourt
"""

#%%
# Importando bibliotecas para Analise

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.forecasting.theta import ThetaModel
import statsmodels.api as sm

#%% importação da base e separação por UN
nps_sabesp = pd.read_excel('Historico NPS.xlsx')
nps_sabesp.info()
nps_oc = nps_sabesp[nps_sabesp['UN'] == 'OC'] 
nps_ol = nps_sabesp[nps_sabesp['UN'] == 'OL']
nps_on = nps_sabesp[nps_sabesp['UN'] == 'ON']
nps_oo = nps_sabesp[nps_sabesp['UN'] == 'OO']
nps_os = nps_sabesp[nps_sabesp['UN'] == 'OS']
nps_of = nps_sabesp[nps_sabesp['UN'] == 'OF']
nps_oi = nps_sabesp[nps_sabesp['UN'] == 'OI']
nps_oj = nps_sabesp[nps_sabesp['UN'] == 'OJ']
nps_om = nps_sabesp[nps_sabesp['UN'] == 'OM']
nps_op = nps_sabesp[nps_sabesp['UN'] == 'OP']
nps_or = nps_sabesp[nps_sabesp['UN'] == 'OR']
nps_ot = nps_sabesp[nps_sabesp['UN'] == 'OT']
nps_ou = nps_sabesp[nps_sabesp['UN'] == 'OU']
nps_ov = nps_sabesp[nps_sabesp['UN'] == 'OV']
nps_ox = nps_sabesp[nps_sabesp['UN'] == 'OX']

#%%
############# ANÁLISE OC ###################

#%%
plt.figure(figsize=(10, 6))
plt.plot(nps_oc.Indicador)
plt.title('NPS OC')
plt.xlabel('Tempo')
plt.ylabel('NPS')
plt.show()

#%%
from statsmodels.tsa.seasonal import seasonal_decompose

# Criar a serie temporal - array
nps_oc_ts = pd.Series(nps_oc.Indicador)

#%% Decomposicao pelo modelo MULTIPLICATIVO
decompm_oc = seasonal_decompose(nps_oc_ts, model='multiplicative', period=4)

# observando os valores da decomposicao pelo modelo multiplicativo
print(decompm_oc.trend)
print(decompm_oc.seasonal)
print(decompm_oc.resid)

#%% Plotar a decomposicao (Selecionar todos os comandos)
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(decompm_oc.trend)
plt.title('Tendencia')

plt.subplot(4, 1, 2)
plt.plot(decompm_oc.seasonal)
plt.title('Componente Sazonal')

plt.subplot(4, 1, 3)
plt.plot(decompm_oc.resid)
plt.title('Resi­duos')

plt.subplot(4, 1, 4)
plt.plot(nps_oc_ts, label='Original')
plt.plot(decompm_oc.trend * decompm_oc.seasonal * decompm_oc.resid, label='Reconstrui­da')
plt.title('Original vs. Reconstruida')
plt.legend()

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
from scipy import stats

###############################################################################
### Metodo de previsao com DRIFT
###############################################################################

# Funcao para calcular a previsao pelo modelo de Drift
def drift_forecast(time_series, steps_ahead, confidence=0.95):
    n = len(time_series)
    
    # Previsao pelo modelo de drift
    drift = (time_series[-1] - time_series[0]) / (n - 1)
    forecast = [time_series[-1] + (i + 1) * drift for i in range(steps_ahead)]
    
    # Calculo dos erros 
    predicted_values = []
    for i in range(1, n):
        predicted_values.append(time_series[i-1] + drift)
    
    # Calcular os erros como a diferenca entre os valores reais e os previstos
    errors = time_series[1:] - np.array(predicted_values)
    erro2=errors*errors
   
    # Calcular o desvio dos erros
    std_errors = np.sqrt(erro2.mean())
 
    # variavel auxiliar para o intervalo de confianca
    calc = len(errors)
  
    # Calcular intervalo de confianca usando o desvio padrao dos erros
    z_value = stats.norm.ppf((1 + confidence) / 2)
   
    # Limites inferior e superior
    lower_bound = [forecast[i] -
                   z_value * std_errors * np.sqrt((i+1)*(1+(i+1)/(calc -1)))
                   for i in range(steps_ahead)]
    upper_bound = [forecast[i] +
                   z_value * std_errors * np.sqrt((i+1)*(1+(i+1)/(calc -1))) 
                   for i in range(steps_ahead)]
        
    return forecast, lower_bound, upper_bound

# In[73]: Funcao para plotar a serie original e a previsao pelo modelo de Drift
def plot_forecast_with_drift(time_series, forecast, lower_bound,
                                            upper_bound, steps_ahead):
    # Plotar a serie original
    plt.plot(range(len(time_series)), time_series, label="Serie Original",
             marker='o')
    
    # Criar eixo de tempo para previsao
    forecast_range = range(len(time_series), len(time_series) + steps_ahead)
    
    # Plotar a previsao
    plt.plot(forecast_range, forecast, label="Previsao (Drift)", marker='o', 
             linestyle='--')
    
    # Plotar intervalo de confianca
    plt.fill_between(forecast_range, lower_bound, upper_bound, color='gray', 
                     alpha=0.05, label='Intervalo de Confianca')
    
    # Detalhes do grafico
    plt.xlabel("Tempo")
    plt.ylabel("Valores")
    plt.legend()
    plt.title("Serie Original e Previsao pelo Modelo de Drift com Intervalo de Confianca")
    plt.show()
    
#%% transformando em array para poder aplicar na função
nps_oc_ts_array = np.array(nps_oc_ts)

# In[74]: Definir numero de passos a  frente
h = 4

# Gerar previsao e intervalos de confianca pelo modelo de Drift
forecast_drift, lower_bound, upper_bound = drift_forecast(nps_oc_ts_array, h)

forecast_drift
lower_bound
upper_bound

# Chamar a funcao de plotagem
plot_forecast_with_drift(nps_oc_ts_array, forecast_drift, lower_bound, upper_bound, h)

#%%
###############################################################################
### Metodo de previsao com NAIVE SAZONAL
###############################################################################

# Funcao para rodar o modelo Naive Sazonal com intervalos de confianca
def seasonal_naive_forecast(time_series, season_length, steps_ahead, confidence=0.95):
    """
    time_series: Série temporal
    season_length: Período sazonal (ex: 12 meses para sazonalidade mensal)
    steps_ahead: O número de períodos à frente para previsão
    confidence: Nível de confiança (95% por padrão)
    """
    # Prever o valor com base na sazonalidade anterior
    forecast = [time_series.iloc[-season_length + i] for i in range(steps_ahead)]
    
    # Previsão dos valores dentro da série histórica para obter resíduos
    predicted_values = [time_series.iloc[i - season_length] for i in range(season_length, len(time_series))]
  
    # Calcular os erros residuais
    residuals = time_series.iloc[season_length:] - np.array(predicted_values)
    
    # Calcular o quadrado dos erros (resíduos)
    errors2 = residuals ** 2
    
    # Calcular o desvio padrão dos erros
    std_residuals = np.sqrt(errors2.mean())
 
    # Calcular o valor crítico z para o intervalo de confiança
    z_value = stats.norm.ppf((1 + confidence) / 2)
    
    # Calcular a margem de erro
    margin_of_error = z_value * std_residuals
    
    # Definir limites inferiores e superiores dos intervalos de confiança
    lower_bound = [forecast[i] - margin_of_error * np.sqrt((i // season_length) + 1) for i in range(steps_ahead)]
    upper_bound = [forecast[i] + margin_of_error * np.sqrt((i // season_length) + 1) for i in range(steps_ahead)]
    
    return forecast, lower_bound, upper_bound

# In[76]: Funcao para plotar a serie original, previsao e intervalos de confianca
def plot_seasonal_naive_forecast(time_series, forecast, lower_bound,
                                 upper_bound, steps_ahead):
    # Plotar a serie original
    plt.plot(range(len(time_series)), time_series, label="Serie Original",
             marker='o')
    
    # Criar eixo de tempo para previsao
    forecast_range = range(len(time_series), len(time_series) + steps_ahead)
    
    # Plotar a previsao
    plt.plot(forecast_range, forecast, label="Previsao (Naive Sazonal)", 
             marker='o', linestyle='--')
    
    # Plotar intervalo de confianca
    plt.fill_between(forecast_range, lower_bound, upper_bound, color='gray',
                     alpha=0.2, label='Intervalo de Confianca')
    
    # Detalhes do grafico
    plt.xlabel("Tempo")
    plt.ylabel("Valores")
    plt.legend()
    plt.title("Serie Original e Previsao pelo Modelo Naive Sazonal com Intervalo de Confianca")
    plt.show()
    
#%%# In[80]: Definir os parametros de sazonalidade e numero de passos a  frente
season_length = 4  # Por exemplo, 12 meses para sazonalidade anual
steps_ahead = 4  # Prevendo os proximos 12 peri­odos (um ano a  frente - h)

# In[81]: Gerar previsao e intervalos de confianca pelo modelo Naive Sazonal
forecast, lower_bound, upper_bound = seasonal_naive_forecast(nps_oc_ts, 
                                                season_length, steps_ahead)

forecast
lower_bound
upper_bound

# Chamar a funcao de plotagem
plot_seasonal_naive_forecast(nps_oc_ts, forecast, lower_bound, upper_bound, 
                             steps_ahead)

#%%
### Modelos de SUAVIZACAO EXPONENCIAL - Simples
###############################################################################

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW

#%%
###############################################################################
# Suavização Exponencial Simples (SES) - Modelagem simples sem tendência/sazonalidade
###############################################################################

# Criando o modelo SES
ses_model = SimpleExpSmoothing(nps_oc_ts).fit()
print(ses_model.summary()) #AIC e BIC, quanto menor, melhor

ses_forecast = ses_model.forecast(4)

# Visualização dos valores previstos
print("Previsões (SES):")
print(ses_forecast)

# Valores ajustados

print("Valores Ajustados (SES):")
print(ses_model.fittedvalues)

# In[84]: Visualização dos dados e previsões com intervalos de confiança
plt.plot(nps_oc_ts, label="Base de Dados")
plt.plot(ses_model.fittedvalues, label="Ajustado (SES)")
plt.plot(np.arange(len(nps_oc_ts), len(nps_oc_ts) + 4), ses_forecast, label="Previsão (SES)")
plt.title("Suavização Exponencial Simples (SES)")
plt.legend()
plt.show()

#%%
# Modelo de Holt com Tendência
###############################################################################

# Modelo Holt
holt_model = Holt(nps_oc_ts).fit()
print(holt_model.summary()) #AIC e BIC, quanto menor, melhor

holt_forecast = holt_model.forecast(4)

# Visualizando os valores previstos
print("Previsão com Holt: ")
print(holt_forecast)

# In[86]: Visualização dos dados e previsões
plt.plot(nps_oc_ts, label="Dados Originais")
plt.plot(holt_model.fittedvalues, label="Ajustado (Holt)")
plt.plot(np.arange(len(nps_oc_ts), len(nps_oc_ts) + 4), holt_forecast, label="Previsão (Holt)")
plt.title("Modelo de Holt com Tendência")
plt.legend()
plt.show()


# In[87]:
###############################################################################
# Modelo Holt-Winters com Tendência (Holt-Winters atribuindo a tendencia)
###############################################################################

ajuste2 = HW(nps_oc_ts, trend='add', seasonal=None).fit()
print(ajuste2.summary()) #AIC e BIC, quanto menor, melhor

fitted_ajuste2 = ajuste2.fittedvalues
print("Valores ajustados (Holt-Winters com tendência):")
print(fitted_ajuste2)

# Previsão de 5 passos à frente
prevajuste2 = ajuste2.forecast(5)
print("Previsão para os próximos 5 períodos:")
print(prevajuste2)

# In[88]: Visualização dos dados ajustados e previsão
plt.plot(base, label="Dados originais")
plt.plot(fitted_ajuste2, label="Ajustado (Holt-Winters com tendência)")
plt.plot(np.arange(len(base), len(base) + 5), prevajuste2, label="Previsão")
plt.title("Holt-Winters com Tendência")
plt.legend()
plt.show()

# In[107]:
###############################################################################
## MODELO ETS
###############################################################################

from sklearn.metrics import mean_absolute_percentage_error as mape
from statsmodels.stats.diagnostic import acorr_ljungbox

nps_oc_1=pd.Series(nps_oc.iloc[:,3].values, #última coluna, que é o NPS
                  index=pd.date_range(start='2019-01-01', periods=len(nps_oc),
                                      freq='QE')) #gerando uma series com data no formato AAAA-MM-DD a cada 3 meses

# Separar a base de dados em treino e teste (janela de dados)
treino_oc = nps_oc_1[:-4]
reais_oc = nps_oc_1[-4:]

# In[108]: Visualizando a série temporal de treino e teste
plt.figure(figsize=(10, 6))
plt.plot(nps_oc_1.index, nps_oc_1, label="Série Completa", color='gray')
plt.plot(treino_oc.index, treino_oc, label="Treino", color='blue')
plt.plot(reais_oc.index, reais_oc, label="Reais", color='red')
plt.title("Série Temporal - Treino e Teste")
plt.xlabel("Data")
plt.ylabel("Valores")
plt.legend()
plt.grid(True)
plt.show()

# In[109]: Definir todas as combinações possíveis de modelos para ETS
configs = [
    {'trend': None, 'seasonal': None},
    {'trend': 'add', 'seasonal': None},
    {'trend': None, 'seasonal': 'add'},
    {'trend': 'add', 'seasonal': 'add'},
    {'trend': None, 'seasonal': 'multiplicative'}, 
    {'trend': 'multiplicative', 'seasonal': None},
    {'trend': 'multiplicative', 'seasonal': 'multiplicative'},
    {'trend': 'add', 'seasonal': 'multiplicative'}, 
    {'trend': 'multiplicative', 'seasonal': 'add'} 
]

best_aic = float('inf')
best_config = None
best_model = None

# In[110]: Ajustar os modelos com diferentes configurações e comparar AIC
for config in configs:
    try:
        model = ExponentialSmoothing(treino_oc, seasonal_periods=4, trend=config['trend'], seasonal=config['seasonal']).fit()
        aic = model.aic

        if aic < best_aic:
            best_aic = aic
            best_config = config
            best_model = model
    except Exception as e:
        pass  # Ignorar configurações que não funcionam

# In[111]: Exibir o melhor modelo com base no AIC
print(f"Melhor configuração: {best_config} com AIC = {best_aic}")
print(best_model.summary())

# In[112]: Previsão de 4 passos à frente com o melhor modelo
best_forecasts = best_model.forecast(steps=4)
print("Previsão para os próximos 4 períodos:")
print(best_forecasts)

# In[113]: Cálculo do MAPE entre a previsão e os valores reais
quali_ETS = mape(reais_oc, best_forecasts)*100
print("MAPE ETS:", quali_ETS)

# In[114]: Visualização da série de treino, valores reais (teste) e previsões
plt.figure(figsize=(10, 6))
plt.plot(treino_oc.index, treino_oc, label="Treino", color='blue')
plt.plot(reais_oc.index, reais_oc, label="Reais", color='red')
plt.plot(pd.date_range(start=reais_oc.index[0], periods=len(best_forecasts), freq='QE'),
         best_forecasts, label="Previsão", color='green')
plt.title("Melhor Modelo ETS com Base no AIC")
plt.xlabel("Data")
plt.ylabel("Valores")
plt.legend()
plt.grid(True)
plt.show()

#%%
###############################################################################
# Teste de Ljung-Box nos resíduos

# Resíduos do modelo ETS
residuals = best_model.resid
#len(residuals)

# Teste de Ljung-Box
ljung_box_result = acorr_ljungbox(residuals, return_df=True)
print("Resultado do Teste de Ljung-Box:")
print(ljung_box_result)

# In[116]: Interpretação do p-valor
if ljung_box_result['lb_pvalue'].values[0] > 0.05:
    print("Aceitamos H0: Os resíduos são independentes (iid). O modelo está bem ajustado.")
else:
    print("Rejeitamos H0: Os resíduos não são iid. O modelo apresenta falhas de ajuste.")

# In[117]: Visualização dos resíduos
plt.figure(figsize=(10, 6))
plt.plot(residuals, label="Resíduos")
plt.title("Resíduos do Melhor Modelo ETS")
plt.grid(True)
plt.show()

#%% Avaliando o MAPE de modelos feitos separadamente
modelos = []
mapes = []

# Modelo Drift
n = len(treino_oc)
drift_slope = (treino_oc.iloc[-1] - treino_oc.iloc[0]) / (n - 1)
drift_forecast = treino_oc.iloc[-1] + drift_slope * np.arange(1, len(reais_oc) + 1)
drift_forecast = pd.Series(drift_forecast, index=reais_oc.index)
print(drift_forecast)
mape_drift = mape(reais_oc, drift_forecast)*100
modelos.append("Drift")
mapes.append(mape_drift)
print(mape_drift)

#%%
# Modelo Naive Sazonal
naive_sazonal_forecast = pd.Series([treino_oc.iloc[-4 + (i % 4)]
                                    for i in range(len(reais_oc))],
                                   index=reais_oc.index)
print(naive_sazonal_forecast)
mape_naive_sazonal = mape(reais_oc, naive_sazonal_forecast)*100
modelos.append("Naive Sazonal")
mapes.append(mape_naive_sazonal)
print(mape_naive_sazonal)

#%% Gerado pelo ETS
modelos.append("Modelo sugerido pelo ETS")
mapes.append(quali_ETS)

#%% # Suavização Exponencial Simples (SES)
ses_model = SimpleExpSmoothing(treino_oc).fit()
ses_forecast = ses_model.forecast(steps=len(reais_oc))
print(ses_forecast)
mape_ses = mape(reais_oc, ses_forecast)*100
modelos.append("SES")
mapes.append(mape_ses)
print(mape_ses)

#%% # Holt com Tendência
holt_model = Holt(treino_oc).fit()
holt_forecast = holt_model.forecast(steps=len(reais_oc))
print(holt_forecast)
mape_holt = mape(reais_oc, holt_forecast)*100
modelos.append("Holt")
mapes.append(mape_holt)
print(mape_holt)

#%%
# Holt-Winters Aditivo
hw_add_model = ExponentialSmoothing(treino_oc, seasonal_periods=4, trend='add', seasonal='add').fit()
hw_add_forecast = hw_add_model.forecast(steps=len(reais_oc))
print(hw_add_forecast)
mape_hw_add = mape(reais_oc, hw_add_forecast)*100
modelos.append("Holt-Winters Aditivo")
mapes.append(mape_hw_add)
print(mape_hw_add)

#%%
# Holt-Winters Multiplicativo
hw_mult_model = ExponentialSmoothing(treino_oc, seasonal_periods=4, trend='add', seasonal='mul').fit()
hw_mult_forecast = hw_mult_model.forecast(steps=len(reais_oc))
print(hw_mult_forecast)
mape_hw_mult = mape(reais_oc, hw_mult_forecast)*100
modelos.append("Holt-Winters Multiplicativo")
mapes.append(mape_hw_mult)
print(mape_hw_mult)

#%%
###############################################################################
# Comparação dos modelos com base no MAPE
mape_comparison = pd.DataFrame({'Modelo': modelos, 'MAPE': mapes})
mape_comparison = mape_comparison.sort_values(by='MAPE', ascending=False).reset_index(drop=True)
print(mape_comparison)

# In[106]:
# Visualizar os MAPE dos modelos em ordem decrescente
plt.figure(figsize=(10, 6))
plt.barh(mape_comparison['Modelo'], mape_comparison['MAPE'], color='red')
plt.xlabel("MAPE")
plt.title("MAPE Comparação de Modelos")
plt.grid(False)
plt.show()