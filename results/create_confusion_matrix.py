import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Dados
referencia = [-546.2643823, 45.22279725, -11.96941717, 6.679883187, -1.929589934,
              -8.305748951, -6.708250001, -10.07031429, -9.691383692, -2.572011707,
              -4.286557595, -4.864034418, -2.808991769]

saida = [-420.53622, 42.652294, -7.537948, 16.34475, -13.077563,
         -18.679827, -26.082201, -21.402866, -12.958301, -8.205008,
         -4.88414, -1.7932868, -5.979645]

labels = ['feliz', 'triste', 'neutro', 'raiva', 'surpresa', 'medo']
bins = [-1000, -10, -5, 0, 5, 10, 1000]  # Ajuste conforme seu domínio

# Converter valores em classes
y_true = pd.cut(referencia, bins=bins, labels=labels).astype(str)
y_pred = pd.cut(saida, bins=bins, labels=labels).astype(str)

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Plotar
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()   