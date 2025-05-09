import kagglehub
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Download e leitura do dataset
print("Baixando o dataset...")
path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")
print("Path to dataset files:", path)

print("Lendo o arquivo CSV...")
dataset = pd.read_csv(path + "/02-14-2018.csv")
print(f"Dataset lido com {dataset.shape[0]} linhas e {dataset.shape[1]} colunas.")

# 2. Limpeza de dados
print("\nRemovendo valores faltantes...")
missing = dataset.isnull().sum().sum()
print(f"Valores faltantes antes: {missing}")
dataset = dataset.dropna()
print(f"Valores faltantes após remoção: {dataset.isnull().sum().sum()}")

print("\nRemovendo duplicatas...")
duplicates = dataset.duplicated().sum()
print(f"Linhas duplicadas: {duplicates}")
dataset = dataset.drop_duplicates()
print(f"Linhas após remoção de duplicatas: {dataset.shape[0]}")

# 3. Checando valores infinitos
colunas_numericas = dataset.select_dtypes(include=[np.number])
colunas_com_inf = colunas_numericas.columns[np.isinf(colunas_numericas).any()].tolist()
print("Colunas com valores infinitos:", colunas_com_inf)

# 4. Análise exploratória
print("\nDistribuição das classes:")
print(dataset['Label'].value_counts())

print("\nExibindo histograma das classes...")
fig = px.histogram(dataset, x='Label', color='Label', title='Distribuição das Classes')
fig.update_layout(xaxis_title='Classe', yaxis_title='Contagem')
fig.show()

# Distribuição por hora
print("\nDistribuição das classes por hora do dia...")
timestamp_col = 'Timestamp'
label_col = 'Label'
try:
    sample_df = dataset.sample(10000, random_state=42)
    sample_df['datetime'] = pd.to_datetime(sample_df[timestamp_col], format='%d/%m/%Y %H:%M:%S', errors='raise')
    sample_df['hour'] = sample_df['datetime'].dt.hour
    hour_label_counts = sample_df.groupby(['hour', label_col]).size().reset_index(name='count')
    fig = px.bar(hour_label_counts, x='hour', y='count', color=label_col, barmode='group', labels={'hour': 'Hora do Dia', 'count': 'Contagem'})
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    fig.update_layout(xaxis_title='Hora do Dia', yaxis_title='Contagem')
    fig.show()
except Exception as e:
    print(f"Erro ao processar '{timestamp_col}': {e}")
    print("Valores de exemplo:")
    print(dataset[[timestamp_col, label_col]].head())

# Matriz de dispersão de features importantes
important_features = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
    'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Bwd Pkt Len Max',
    'Flow IAT Mean', 'Flow Byts/s'
]
print("\nExibindo matriz de dispersão das principais features...")
sample_size = 5000
df_sample = dataset.sample(n=min(sample_size, len(dataset)), random_state=42)
for feature in important_features:
    if df_sample[feature].isnull().sum() > 0:
        df_sample[feature] = df_sample[feature].fillna(df_sample[feature].median())
fig = px.scatter_matrix(
    df_sample,
    dimensions=important_features,
    color='Label' if 'Label' in dataset.columns else None,
    opacity=0.7,
    height=1200,
    width=1300
)
fig.update_traces(diagonal_visible=False)
fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
fig.show()

# 5. Balanceamento das classes
print("\nBalanceando as classes...")
dfBenign = dataset[dataset['Label'] == 'Benign']
dfSsh = dataset[dataset['Label'] == 'SSH-Bruteforce']
dfFtp = dataset[dataset['Label'] == 'FTP-BruteForce']
min_size = min(len(dfBenign), len(dfSsh), len(dfFtp))
dfBenignAm = dfBenign.sample(n=min_size, random_state=42)
dfSshAm = dfSsh.sample(n=min_size, random_state=42)
dfFtpAm = dfFtp.sample(n=min_size, random_state=42)
dfBalanced = pd.concat([dfBenignAm, dfSshAm, dfFtpAm])
Amostdataset = dfBalanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Tamanho do dataset balanceado: {Amostdataset.shape}")

# 6. Feature engineering: discretização de Timestamp
print("\nDiscretizando coluna Timestamp...")
Amostdataset['Timestamp'] = pd.to_datetime(Amostdataset['Timestamp'], format="%d/%m/%Y %H:%M:%S")
Amostdataset['month'] = Amostdataset['Timestamp'].dt.month
Amostdataset['day'] = Amostdataset['Timestamp'].dt.day
Amostdataset['hour'] = Amostdataset['Timestamp'].dt.hour
Amostdataset['minute'] = Amostdataset['Timestamp'].dt.minute
Amostdataset = Amostdataset.drop(columns=['Timestamp'])

# 7. Divisão treino/teste
print("\nDividindo em treino e teste...")
X = Amostdataset.drop('Label', axis=1)
y = Amostdataset['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Tamanho treino: {X_train.shape}, teste: {X_test.shape}")

# 8. Tratamento de valores infinitos e NaN
print("\nTratando valores infinitos e NaN...")
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# 9. Label encoding
label_mapping = {
    'Benign': 0,
    'FTP-BruteForce': 1,
    'SSH-Bruteforce': 2
}
y_train = y_train.replace(label_mapping)
y_test = y_test.replace(label_mapping)

# 10. Treinamento do modelo
print("\nTreinando modelo de Árvore de Decisão (ID3)...")
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(X_train, y_train)
y_pred = id3_model.predict(X_test)

# 11. Avaliação do modelo
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 12. Matriz de confusão
print("\nExibindo matriz de confusão...")
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=id3_model.classes_, yticklabels=id3_model.classes_)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig("matriz_confusao.png")
plt.close()

# 13. Visualização da árvore de decisão
print("\nExibindo árvore de decisão...")
class_names_str = ['Benign', 'FTP-BruteForce', 'SSH-Bruteforce']
plt.figure(figsize=(20, 10))
plot_tree(
    id3_model,
    feature_names=X_train.columns,
    class_names=class_names_str,
    filled=True
)
plt.title("Árvore de Decisão ID3")
plt.savefig("arvore_decisao.png")
plt.close()

print("\nAs figuras foram salvas como 'matriz_confusao.png' e 'arvore_decisao.png'.")
