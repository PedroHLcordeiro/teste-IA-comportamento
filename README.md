# teste-IA-comportamento
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Carregar os dados (substitua isso pelo carregamento real do seu conjunto de dados)
data = pd.read_csv("seu_arquivo.csv")

# Selecionar colunas relevantes para a análise (ajuste conforme necessário)
features = ['feature1', 'feature2', 'feature3']
X = data[features]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar o modelo de Isolation Forest
model = IsolationForest(contamination=0.05)  # Ajuste a taxa de contaminação conforme necessário
model.fit(X_scaled)

# Prever anomalias no conjunto de dados
predictions = model.predict(X_scaled)

# Adicione as previsões ao DataFrame original
data['is_anomaly'] = (predictions == -1)

# Exiba ou salve os resultados conforme necessário
print(data)
