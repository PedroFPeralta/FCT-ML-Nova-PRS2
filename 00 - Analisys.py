import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

train_path = "train_data.csv"
test_path = "test_data.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Treino - Primeiras Linhas:")
print(train_data.head())

print("\nTeste - Primeiras Linhas:")
print(test_data.head())

print("\nInformações dos Dados de Treino:")
print(train_data.info())

print("\nInformações dos Dados de Teste:")
print(test_data.info())

# Analisar valores ausentes
print("\nValores Ausentes - Treino:")
print(train_data.isnull().sum())

print("\nValores Ausentes - Teste:")
print(test_data.isnull().sum())

msno.bar(train_data, color="blue", figsize=(10, 6))
plt.title("Valores Ausentes - Conjunto de Treino", fontsize=14)
plt.show()

msno.bar(test_data, color="green", figsize=(10, 6))
plt.title("Valores Ausentes - Conjunto de Teste", fontsize=14)
plt.show()

# 3. Estatísticas descritivas das variáveis numéricas
print("\nEstatísticas Descritivas - Treino:")
print(train_data.describe())

print("\nEstatísticas Descritivas - Teste:")
print(test_data.describe())

# 4. Distribuição de SurvivalTime e censura no treino
sns.histplot(train_data["SurvivalTime"], kde=True, bins=30, color="blue", label="Survival Time")
plt.title("Distribuição do Survival Time")
plt.xlabel("Tempo de Sobrevivência")
plt.legend()
plt.show()

sns.countplot(data=train_data, x="Censored", hue="Censored", palette="viridis", legend=False)
plt.title("Distribuição de Censored")
plt.xlabel("Censored (0 = Evento, 1 = Censurado)")
plt.ylabel("Contagem")
plt.show()

# Relação entre Censored e SurvivalTime
sns.boxplot(data=train_data, x="Censored", y="SurvivalTime", hue="Censored", palette="pastel", dodge=False)
plt.title("SurvivalTime vs Censored")
plt.xlabel("Censored (0 = Evento, 1 = Censurado)")
plt.ylabel("Tempo de Sobrevivência")
plt.show()

msno.matrix(train_data)
plt.title("Missing Data Matrix - Train Data")
plt.show()

msno.matrix(test_data)
plt.title("Missing Data Matrix - Test Data")
plt.show()

msno.dendrogram(train_data)
plt.title("Missing Data Dendrogram - Train Data")
plt.show()

msno.dendrogram(test_data)
plt.title("Missing Data Dendrogram - Test Data")
plt.show()


# Proporção de valores ausentes no Treino
missing_train_percent = train_data.isnull().mean() * 100
print(missing_train_percent)

# Proporção de valores ausentes no teste
missing_test_percent = test_data.isnull().mean() * 100
print(missing_test_percent)


def cMSE(y_hat, y, c):
    """
    Calcula o Censored Mean Squared Error (cMSE).

    :param y_hat: Valores previstos.
    :param y: Valores reais.
    :param c: Indicador de censura (1 = censurado, 0 = não censurado).
    :return: cMSE.
    """
    err = y - y_hat
    censored_error = (1 - c) * (err ** 2) + c * np.maximum(0, err) ** 2
    return np.sum(censored_error) / len(err)


y_real = np.array([5, 10, 15, 20])
y_pred = np.array([4, 12, 14, 19])
censored = np.array([0, 1, 0, 1])

print("\nTeste da Métrica cMSE:")
print("cMSE:", cMSE(y_pred, y_real, censored))

# Heatmap de correlação para o conjunto de Treino
corr_matrix = train_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de Correlação - Treino")
plt.show()

##########################################################################################
# Remover colunas
##########################################################################################
train_data_cleaned = train_data.drop(columns=["GeneticRisk", "ComorbidityIndex", "TreatmentResponse", "SurvivalTime", "Censored"], errors="ignore")
print("Colunas restantes no conjunto de Treino após remoção de colunas específicas:")
print(train_data_cleaned.columns)

# Remover colunas com valores null no conjunto de teste
test_data_cleaned = test_data.drop(columns=["GeneticRisk", "ComorbidityIndex", "TreatmentResponse"], errors="ignore")
print("\nColunas restantes no conjunto de teste após remoção de colunas específicas:")
print(test_data_cleaned.columns)

# Exibir quantidades de pontos restantes
print(f"Número de pontos restantes no Treino: {len(train_data_cleaned)}")
print(f"Número de pontos restantes no teste: {len(test_data_cleaned)}")

features_for_pairplot = ['id', 'Age', 'Gender', 'Stage', 'TreatmentType']
train_data_for_pairplot = train_data_cleaned[features_for_pairplot]

sns.pairplot(train_data_for_pairplot)
plt.suptitle("Pairplot entre as Variáveis e SurvivalTime", y=1.02)
plt.show()

