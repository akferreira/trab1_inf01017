from random import shuffle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_folds(k, dataset):
  #obtem os indices das instancias de spam e não-spam
  spam_index = list(dataset.index[dataset['spam'] == True])
  notspam_index = list(dataset.index[dataset['spam'] == False])

  #embaralha os indices aleatoriamente para garantir que os folds sejam aleatorios
  shuffle(spam_index)
  shuffle(notspam_index)

  #divide os arrays de indices em k partes
  spam_index_folds = np.array_split(spam_index,k)
  notspam_index_folds = np.array_split(notspam_index,k)

  folds_index = []
  folds = []

  #cria k folds de indices combinando um elemento dos indices de spam e um dos indices de não-spam
  for spam,notspam in zip(spam_index_folds,notspam_index_folds):
    folds_index.append(list(spam) + list(notspam))

  #por meio das listas de índices busca os elementos da tabela do dataset
  for kfold_index in folds_index:
    folds.append( dataset.iloc[kfold_index] )

  return folds
    
def combine_folds_training_testing(folds):
  K = len(folds)

  train_test_sets = []

  for i in range(K):
    print()
    test_fold_index = i
    training_folds_index = [j for j in range(K) if j != i]

    X_test = folds[test_fold_index].drop('spam', axis=1)
    y_test = folds[test_fold_index]['spam']

    #combina os folds do pandadataframe que serão usados para o conjunto de treinamento
    df_merged = pd.concat( [folds[index] for index in training_folds_index], ignore_index = False, sort=False)
    X_train = df_merged.drop('spam', axis=1)
    y_train = df_merged['spam']

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_test_sets.append({'X_train': X_train,'X_test': X_test, 'y_train': y_train,'y_test':y_test})

  return train_test_sets
