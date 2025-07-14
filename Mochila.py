"""
Problema de la Mochila 0-1 resuelto mediante Algoritmos Genéticos

Este script implementa un algoritmo genético para resolver el clásico problema de la mochila,
donde se debe maximizar el valor total de los ítems seleccionados sin exceder la capacidad máxima.

Características:
- Dataset ampliado de ítems (generado aleatoriamente)
- Soporte para múltiples ejecuciones
- Análisis estadístico (media, desviación estándar, test t)
- Visualización interactiva de la evolución del modelo
- Generación de reporte final de soluciones

Autor: Yerard (2025)
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import silhouette_score
from IPython.display import display

# Configuración global
np.random.seed(42)
random.seed(42)

# -----------------------------
# 1. Generación del dataset de ítems
# -----------------------------

def generar_dataset(n_items=30, max_valor=200, max_peso=30, capacidad_mochila=None):
    """
    Genera un dataset sintético de ítems con valor y peso aleatorio.
    
    Parámetros:
        n_items (int): Número de ítems a generar
        max_valor (int): Valor máximo por ítem
        max_peso (int): Peso máximo por ítem
        capacidad_mochila (int): Capacidad de la mochila (si no se da, se calcula como 60% del peso total)
    
    Retorna:
        DataFrame con columnas ['item', 'valor', 'peso']
        capacidad (int): Capacidad máxima de la mochila
    """
    items = [f'Item_{i+1}' for i in range(n_items)]
    valores = np.random.randint(1, max_valor + 1, size=n_items)
    pesos = np.random.randint(1, max_peso + 1, size=n_items)
    
    df_items = pd.DataFrame({
        'item': items,
        'valor': valores,
        'peso': pesos
    })
    
    if capacidad_mochila is None:
        # Establecer capacidad en función del total
        capacidad = int(df_items['peso'].sum() * 0.6)
    else:
        capacidad = capacidad_mochila
    
    return df_items, capacidad

# Generar dataset
df_items, capacidad = generar_dataset(n_items=30)  # Cambiar n_items según necesidad
print("📦 Dataset de ítems:")
display(df_items.head(8))  # Mostrar solo algunos
print(f"\nCapacidad de la mochila: {capacidad} kg")

# -----------------------------
# 2. Funciones del algoritmo genético
# -----------------------------

def crear_individuo(n_genes):
    """
    Crea un individuo binario aleatorio (representa una posible solución).
    Cada bit indica si se incluye (1) o no (0) un ítem.
    """
    return [random.randint(0, 1) for _ in range(n_genes)]

def evaluar(ind, df, capacidad):
    """
    Evalúa un individuo: calcula su valor total si cumple con la capacidad de la mochila.
    Si excede el peso, devuelve penalización (0).
    """
    peso_total = sum(ind[i] * df.iloc[i]['peso'] for i in range(len(ind)))
    valor_total = sum(ind[i] * df.iloc[i]['valor'] for i in range(len(ind)))
    return valor_total if peso_total <= capacidad else 0

def seleccion_torneo(poblacion, df, capacidad, k=3):
    """
    Selección por torneo: elige k individuos aleatorios y retorna el mejor.
    """
    competidores = random.sample(poblacion, k)
    return max(competidores, key=lambda ind: evaluar(ind, df, capacidad))

def cruce_un_punto(p1, p2):
    """
    Cruce de un punto entre dos padres para generar un hijo.
    """
    punto = random.randint(1, len(p1) - 1)
    return p1[:punto] + p2[punto:]

def mutacion(ind, tasa_mutacion=0.05):
    """
    Aplica mutación a cada gen del individuo con probabilidad `tasa_mutacion`.
    """
    return [1 - gen if random.random() < tasa_mutacion else gen for gen in ind]

def algoritmo_genetico(
    df,
    capacidad,
    generaciones=100,
    tamaño_poblacion=50,
    tasa_mutacion=0.05
):
    """
    Ejecuta un algoritmo genético para resolver el problema de la mochila.
    
    Retorna:
        mejor_solución (list): Representación binaria del mejor individuo
        mejor_valor (float): Mejor valor obtenido
        historia (list): Historial del mejor valor por generación
    """
    n = len(df)
    poblacion = [crear_individuo(n) for _ in range(tamaño_poblacion)]
    mejores_valores = []

    for _ in range(generaciones):
        nueva_poblacion = []
        for _ in range(tamaño_poblacion):
            p1 = seleccion_torneo(poblacion, df, capacidad)
            p2 = seleccion_torneo(poblacion, df, capacidad)
            hijo = cruce_un_punto(p1, p2)
            hijo = mutacion(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion
        mejor_actual = max(poblacion, key=lambda ind: evaluar(ind, df, capacidad))
        mejores_valores.append(evaluar(mejor_actual, df, capacidad))

    mejor_final = max(poblacion, key=lambda ind: evaluar(ind, df, capacidad))
    mejor_valor = evaluar(mejor_final, df, capacidad)
    return mejor_final, mejor_valor, mejores_valores

# -----------------------------
# 3. Ejecutar múltiples corridas
# -----------------------------

# Parámetros de simulación
rondas = 20
resultados = []

for i in range(rondas):
    print(f"\rEjecutando ronda {i+1}/{rondas}", end="")
    solucion, valor, historia = algoritmo_genetico(
        df=df_items,
        capacidad=capacidad,
        generaciones=100,
        tamaño_poblacion=50,
        tasa_mutacion=0.05
    )
    resultados.append({
        'ronda': i + 1,
        'solucion': solucion,
        'valor_total': valor,
        'historia': historia
    })

# Convertir resultados en DataFrame
df_resultados = pd.DataFrame(resultados)

# Calcular estadísticas adicionales
df_resultados['media_historia'] = df_resultados['historia'].apply(lambda x: np.mean(x))
df_resultados['desviacion_std'] = df_resultados['historia'].apply(lambda x: np.std(x))

# -----------------------------
# 4. Análisis estadístico
# -----------------------------

# Media y desviación estándar de los valores totales obtenidos
media_valores = df_resultados['valor_total'].mean()
std_valores = df_resultados['valor_total'].std()

# Test t contra un benchmark (ej.: 1000)
t_stat, p_valor = stats.ttest_1samp(df_resultados['valor_total'], popmean=1000)

# -----------------------------
# 5. Visualización de resultados
# -----------------------------

# Histograma de valores totales obtenidos
plt.figure(figsize=(12, 6))
sns.histplot(df_resultados['valor_total'], kde=True, color='steelblue', bins=10)
plt.title("Distribución de Valores Totales Obtenidos")
plt.xlabel("Valor Total")
plt.ylabel("Frecuencia")
plt.axvline(media_valores, color='red', linestyle='--', label=f'Media: {media_valores:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evolución promedio del fitness durante las generaciones
plt.figure(figsize=(10, 5))
for i, row in df_resultados.iterrows():
    plt.plot(row['historia'], alpha=0.2, color='gray')
plt.plot(np.mean([r['historia'] for r in resultados], axis=0), label="Media Generacional", linewidth=2, color='navy')
plt.title("Evolución del Fitness Promedio Durante Generaciones")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Resumen final e impresión de resultados
# -----------------------------

print("\n\n📊 RESUMEN FINAL DEL ANÁLISIS")
print(f"{'-'*40}")
print(f"Media del valor total: {media_valores:.2f}")
print(f"Desviación estándar: {std_valores:.2f}")
print(f"T-test vs Benchmark (H₀: media = 1000): t = {t_stat:.2f}, p = {p_valor:.4f}")
print("\nMejores soluciones por ronda:")
print(df_resultados[['ronda', 'valor_total']].sort_values(by='valor_total', ascending=False).head(10))

# Mostrar detalles de la mejor solución
mejor_idx = df_resultados['valor_total'].idxmax()
mejor_solucion = df_resultados.loc[mejor_idx, 'solucion']
items_seleccionados = df_items.iloc[[i for i, val in enumerate(mejor_solucion) if val == 1]]
print("\n🎯 MEJOR SOLUCIÓN ENCONTRADA:")
print(items_seleccionados)
print(f"\nPeso total: {items_seleccionados['peso'].sum()} / {capacidad}")
print(f"Valor total: {items_seleccionados['valor'].sum()}")