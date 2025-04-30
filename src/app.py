# %% [markdown]
# # Proyecto: Naive Bayes

# %% [markdown]
# ## Importar librerías

# %%
# Librerías estándar
import pickle
import pandas as pd

# Scikit-learn: preparación de datos
from sklearn.model_selection import train_test_split, GridSearchCV

# Scikit-learn: vectorización
from sklearn.feature_extraction.text import CountVectorizer

# Scikit-learn: modelos Naive Bayes
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# Scikit-learn: métricas
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score


# %% [markdown]
# ## Paso 1: Carga del conjunto de datos

# %%
# Enlace del dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"

# Cargar el dataset
df = pd.read_csv(url)

# Mostrar las primeras filas
df.head()

# %%
# Guardar el DataFrame en un archivo CSV en la carpeta raw
df.to_csv('/workspaces/efrainnalmeida-naive-bayes-project/data/raw/playstore_reviews.csv', index=False)

# %%
# Guardar el DataFrame en un archivo CSV en la carpeta interim
df.to_csv('/workspaces/efrainnalmeida-naive-bayes-project/data/interim/playstore_reviews.csv', index=False)

# %%
# Extraer el archivo CSV de la carpeta interim
df_interim = pd.read_csv("/workspaces/efrainnalmeida-naive-bayes-project/data/interim/playstore_reviews.csv")

# Mostrar las primeras filas
df_interim.head()

# %% [markdown]
# ## Paso 2: Procesamiento del texto

# %%
# Mostrar forma del DataFrame
df_interim.shape

# %%
# Mostrar información del DataFrame
df_interim.info()

# %%
# Mostrar estadísticas descriptivas
df_interim["polarity"].describe()

# %%
# Frecuencia absoluta y relativa de la variable "polarity"
print(df_interim["polarity"].value_counts())
print(df_interim["polarity"].value_counts(normalize=True))

# %%
# 1. Eliminar espacios y convertir a minúsculas
df_interim['review'] = df_interim['review'].str.strip().str.lower()

# %%
# 2. Eliminar columna que no aporta valor predictivo
df_interim = df_interim.drop("package_name", axis=1)

# %%
# 3. Separar variables predictoras y objetivo
X = df_interim["review"]
y = df_interim["polarity"]

# %%
# 4. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# 5. Vectorizar con CountVectorizer
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# %% [markdown]
# ## Paso 3: Construir el modelo Naive Bayes

# %%
# MultinomialNB
model_multi = MultinomialNB()
model_multi.fit(X_train_vec, y_train)
y_pred_multi = model_multi.predict(X_test_vec)
y_prob_multi = model_multi.predict_proba(X_test_vec)[:, 1]

print("🔵 MultinomialNB")
print(classification_report(y_test, y_pred_multi))
print("F1 Score (macro):", f1_score(y_test, y_pred_multi, average='macro'))
print("F1 Score (weighted):", f1_score(y_test, y_pred_multi, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_multi))

# GaussianNB
model_gauss = GaussianNB()
model_gauss.fit(X_train_vec, y_train)
y_pred_gauss = model_gauss.predict(X_test_vec)
y_prob_gauss = model_gauss.predict_proba(X_test_vec)[:, 1]

print("\n🟡 GaussianNB")
print(classification_report(y_test, y_pred_gauss))
print("F1 Score (macro):", f1_score(y_test, y_pred_gauss, average='macro'))
print("F1 Score (weighted):", f1_score(y_test, y_pred_gauss, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_gauss))

# BernoulliNB
model_bernoulli = BernoulliNB()
model_bernoulli.fit(X_train_vec, y_train)
y_pred_bernoulli = model_bernoulli.predict(X_test_vec)
y_prob_bernoulli = model_bernoulli.predict_proba(X_test_vec)[:, 1]

print("\n🟢 BernoulliNB")
print(classification_report(y_test, y_pred_bernoulli))
print("F1 Score (macro):", f1_score(y_test, y_pred_bernoulli, average='macro'))
print("F1 Score (weighted):", f1_score(y_test, y_pred_bernoulli, average='weighted'))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_bernoulli))

# %% [markdown]
# ## Paso 4: Optimizar el modelo anterior

# %%
# Definir el modelo base
nb = MultinomialNB()

# Definir la grilla
param_grid = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
    'fit_prior': [True, False]
}

# Configurar la búsqueda
grid_search = GridSearchCV(
    estimator=nb,
    param_grid=param_grid,
    scoring='f1_macro',  # Métrica a optimizar
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Ejecutar la búsqueda
grid_search.fit(X_train_vec, y_train)

# Mostrar mejores hiperparámetros y score
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor F1 macro (validación cruzada):", grid_search.best_score_)

# %%
# Obtener el mejor modelo entrenado automáticamente
best_model = grid_search.best_estimator_

# Predecir en el conjunto de prueba
y_pred_opt = best_model.predict(X_test_vec)
y_prob_opt = best_model.predict_proba(X_test_vec)[:, 1]

# Evaluación
print("🔵 Modelo optimizado MultinomialNB (desde best_estimator_)")
print(classification_report(y_test, y_pred_opt))
print("F1 macro:", f1_score(y_test, y_pred_opt, average='macro'))
print("F1 weighted:", f1_score(y_test, y_pred_opt, average='weighted'))
print("ROC AUC:", roc_auc_score(y_test, y_prob_opt))

# %% [markdown]
# ## Paso 5: Guardar el modelo

# %%
# Ruta de guardado
model_path = "/workspaces/efrainnalmeida-naive-bayes-project/models/multinomialnb_model.pkl"

# Guardar el modelo entrenado
with open(model_path, "wb") as file:
    pickle.dump(best_model, file)

print(f"Modelo guardado correctamente en: {model_path}")


