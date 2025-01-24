# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:13:54 2025

@author: Danilo
"""

# Importando bibliotecas
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    roc_auc_score
)


# Função para calcular especificidade
def calcular_especificidade_multiclasse(y_true, y_pred, classe):
    """Calcula a especificidade para uma classe específica com base na matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm.sum() - (cm[:, classe].sum() + cm[classe, :].sum() - cm[classe, classe])
    fp = cm[:, classe].sum() - cm[classe, classe]
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
    return especificidade


# Função para plotar a matriz de confusão
def plotar_matriz_confusao(y_true, y_pred, classes):
    """Plota a matriz de confusão normalizada."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalizado = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(cm_normalizado, index=classes, columns=classes)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Matriz de Confusão Normalizada')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plotar_curva_roc_multiclasse(y_true, y_scores, classes):
    """Plota a curva ROC para cada classe e calcula AUC para múltiplas classes."""
    plt.figure(figsize=(8, 8))
    for i, classe in enumerate(classes):
        # Converte y_true para binário para a classe atual
        y_true_bin = (np.array(y_true) == classe).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Classe {classe} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Curva ROC Multiclasse')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()



# Função para calcular e exibir métricas de classificação
def avaliar_metrica_separada(nome, valor):
    print(f"{nome}: {valor:.2f}")


# Função principal para avaliar classificação
def avaliar_classificacao(y_true, y_pred, y_scores, classes):
    print("Métricas de Classificação:")
    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    avaliar_metrica_separada("Acurácia", acuracia)
    avaliar_metrica_separada("Precisão", precisao)
    avaliar_metrica_separada("Sensibilidade (Recall)", recall)
    avaliar_metrica_separada("F1-Score", f1)

    # Especificidade para cada classe
    print("\nEspecificidade por classe:")
    for i, classe in enumerate(classes):
        especificidade = calcular_especificidade_multiclasse(y_true, y_pred, i)
        print(f"Classe {classe}: {especificidade:.2f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))

    plotar_matriz_confusao(y_true, y_pred, classes)
    plotar_curva_roc_multiclasse(y_true, y_scores, classes)


# Código de exemplo em Google Colab
if __name__ == "__main__":
    # Dados fictícios para exemplo de classificação
    y_true_clf = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    y_pred_clf = [0, 1, 2, 3, 0, 0, 1, 2, 3, 4, 4, 1, 2, 3, 4]
    y_scores_clf = np.random.rand(len(y_true_clf), 5)  # Probabilidades preditivas para 5 classes
    classes = [0, 1, 2, 3, 4]

    # Avaliação do modelo
    avaliar_classificacao(y_true_clf, y_pred_clf, y_scores_clf, classes)
