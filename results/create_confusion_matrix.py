from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Exemplo de dados
referencia = [-546.2643823,45.22279725,-11.96941717,6.679883187,-1.929589934,-8.305748951,-6.708250001,-10.07031429,-9.691383692,-2.572011707,-4.286557595,-4.864034418,-2.808991769]
saida = [-420.53622  ,   42.652294 ,   -7.537948 ,   16.34475   , -13.077563,
  -18.679827 ,  -26.082201   ,-21.402866  , -12.958301 ,   -8.205008,
   -4.88414 ,    -1.7932868 ,  -5.979645 ]



# Definir bins e rótulos
bins = [-np.inf, -5, 5, np.inf]
labels = ['Negativo', 'Neutro', 'Positivo']

# Discretizar
ref_cat = pd.cut(referencia, bins=bins, labels=labels)
pred_cat = pd.cut(saida, bins=bins, labels=labels)

# Matriz de confusão
cm = confusion_matrix(ref_cat, pred_cat)
# Visualizar
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Speaker_0000_00046.wav')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()