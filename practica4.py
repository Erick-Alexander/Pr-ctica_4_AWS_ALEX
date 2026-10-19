"""
PRÁCTICA 4: GENERACIÓN DE DATOS SINTÉTICOS CON FAKER Y DIVISIÓN STRATIFIED TRAIN-TEST
(Como en mi video loquendo XD)
"""

from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Aquí el faker se configura de modo que la localización chilanga siendo es_MX y la semilla puesta en 42
fake = Faker('es_MX')
Faker.seed(42)
np.random.seed(42)

# Aquí se genera todo el Dataset sintético, es decir los 1000 registros de datos.
registros = []
for i in range(1000):
    registros.append({
        'id_cliente': i + 1,
        'nombre': fake.name(),
        'email': fake.email(),
        'telefono': fake.phone_number(),           # campo extra de telefono crado por el faker
        'fecha_nacimiento': fake.date_of_birth(minimum_age=18, maximum_age=75), # campo de fecha creado por el faker.
        'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], p=[0.5, 0.4, 0.1]),
        'ingreso_anual': round(np.random.uniform(100000, 1000000), 2),
        'churn': np.random.choice([0, 1], p=[0.8, 0.2])  # tasa del churn que en realidad se vuelve del 0.19%
    })

df_sintetico = pd.DataFrame(registros)

print("=== 1. MUESTRA DEL DATASET SINTÉTICO GENERADO (FAKER es_MX) ===")
print(df_sintetico.head(5))
print(f"\nTotal Registros Generados: {len(df_sintetico)}")
print("\n")

# Aquí se codifica la categoría del One-Hot para el valor síntetico de estado civil
df_encoded = pd.get_dummies(df_sintetico, columns=['estado_civil'], drop_first=False)

# Aquí se selecciona la matriz de características en X y las variables de objetivo en Y en donde se elimina información vacía.
X = df_encoded.drop(columns=['id_cliente', 'nombre', 'email', 'telefono', 'fecha_nacimiento', 'churn'])
y = df_encoded['churn']

# Aquí es donde toda la data de 1000 registros se dividen en los 800 y 200 datos de entrenamiento y testeo.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("=== 2. DIMENSIONES DE LOS CONJUNTOS DE DATOS ===")
print(f"Dataset Completo X: {X.shape} | y: {y.shape}")
print(f"Conjunto de Entrenamiento (Train) X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"Conjunto de Prueba (Test) X_test: {X_test.shape} | y_test: {y_test.shape}")
print("\n")

# Aquí es donde se verifica y analiza la estratificación en base al tamaño del churn.
tabla_proporciones = pd.DataFrame({
    'Conjunto': ['Dataset Completo', 'Entrenamiento (Train 80%)', 'Prueba (Test 20%)'],
    'Total Registros': [len(y), len(y_train), len(y_test)],
    'Casos Churn = 1': [y.sum(), y_train.sum(), y_test.sum()],
    'Casos No Churn = 0': [len(y) - y.sum(), len(y_train) - y_train.sum(), len(y_test) - y_test.sum()],
    'Tasa de Churn (Proporción)': [f"{y.mean():.4f} ({y.mean()*100:.2f}%)", 
                                  f"{y_train.mean():.4f} ({y_train.mean()*100:.2f}%)", 
                                  f"{y_test.mean():.4f} ({y_test.mean()*100:.2f}%)"]
})

print("=== 3. TABLA COMPARATIVA DE ESTRATIFICACIÓN DE LA CLASE CHURN ===")
print(tabla_proporciones.to_string(index=False))
