import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import stdev,mean
from random import shuffle
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,ConfusionMatrixDisplay



def generate_folds(k,dataset,spam,notspam,total):
  spam_k = spam/k
  notspam_k = notspam/k

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


class ConfusionMatrix():
    def __init__(self,y_test,y_pred):
      self.matrix = [[0,0],[0,0]]

      for pred,actual in zip(y_pred,y_test):
        self.matrix[pred][actual]+=1

      self.pred = y_pred
      self.actual = y_test
      self.total = len(y_pred)
      self.TN = self.matrix[0][0]
      self.FN = self.matrix[0][1]
      self.FP = self.matrix[1][0]
      self.TP = self.matrix[1][1]
      self.fmeasure_beta = 2  #para o caso de detecção de spam, julgou-se mais importante minimizar a quantidade de spam não detectado do que a taxa de emails classifcados erroneamente como spam

    def __str__(self):
      string = ""
      string += f"\n \t\t Classe verdadeira \t \n \t\t 0 \t 1"
      string += f"\n Classe | 0\t {self.matrix[0][0]}\t{self.matrix[0][1]}"
      string += f"\n Predita| 1\t {self.matrix[1][0]}\t{self.matrix[1][1]}"
      return string

    def accuracy(self):
      return (self.TN + self.TP)/self.total

    def recall(self):
      return (self.TP/ (self.TP + self.FN))

    def precision(self):
      return (self.TP/ (self.TP + self.FP))

    def f1_measure(self):
      return  (1 + self.fmeasure_beta**2) * ((self.precision() * self.recall())/ ((self.fmeasure_beta**2 * self.precision()) + self.recall()) )

    def metrics_text(self):
      return ( f"\n Acurácia: {self.accuracy()} \n Recall : {self.recall()}\n precisão : {self.precision()} \n f1 score : {self.f1_measure()}\n")


    def metrics(self):
      return {'acc': self.accuracy(),'recall': self.recall(),'precision': self.precision(),'f1': self.f1_measure()}



def knn_plotting(knn_metrics,knn_stdev,knn_sizes):
  plt.title('Métricas KNN')
  plt.figure("Média")
  plt.plot(knn_sizes,knn_metrics['acc'],marker='o', linestyle='--', color='r', label='Acurácia')
  plt.plot(knn_sizes,knn_metrics['recall'],marker='o', linestyle='--', color='b', label='Recall')
  plt.plot(knn_sizes,knn_metrics['precision'],marker='o', linestyle='--', color='g', label='Precisão')
  plt.plot(knn_sizes,knn_metrics['f1'],marker='o', linestyle='--', color='m', label='F1 score')
  plt.legend()
  plt.xticks(knn_sizes,knn_sizes)
  plt.savefig('metrics_avg.png')

  
  plt.figure("Acurácia Desvio padrão")
  plt.xticks(knn_sizes,knn_sizes)
  plt.errorbar(knn_sizes,knn_metrics['acc'],yerr = knn_stdev['acc'],marker='o', linestyle='--', color='r', label='Acurácia')
  plt.legend()
  plt.savefig('metrics1_stdev.png')

  plt.figure("Recall Desvio padrão")
  plt.xticks(knn_sizes,knn_sizes)
  plt.errorbar(knn_sizes,knn_metrics['recall'],yerr = knn_stdev['recall'],marker='o', linestyle='--', color='b', label='Recall')
  plt.legend()
  plt.savefig('metrics2_stdev.png')

  plt.figure("Precisão Desvio padrão")
  plt.xticks(knn_sizes,knn_sizes)
  plt.errorbar(knn_sizes,knn_metrics['precision'],yerr = knn_stdev['precision'],marker='o', linestyle='--', color='g', label='Precisão')
  plt.legend()
  plt.savefig('metrics3_stdev.png')

  plt.figure("f1 Desvio padrão")
  plt.xticks(knn_sizes,knn_sizes)
  plt.errorbar(knn_sizes,knn_metrics['f1'],yerr = knn_stdev['f1'],marker='o', linestyle='--', color='m', label='F1 score')
  plt.legend()
  plt.savefig('metrics4_stdev.png')

  
  
  
  




names = ["word_freq_make" ,"word_freq_address","word_freq_all"          ,"word_freq_3d"           ,"word_freq_our"          ,"word_freq_over"         ,"word_freq_remove"       ,"word_freq_internet"     ,"word_freq_order"        ,"word_freq_mail"         ,"word_freq_receive"      ,"word_freq_will"         ,"word_freq_people"       ,"word_freq_report"       ,"word_freq_addresses"    ,"word_freq_free"         ,"word_freq_business"     ,"word_freq_email"        ,"word_freq_you"          ,"word_freq_credit"       ,"word_freq_your"         ,"word_freq_font"         ,"word_freq_000"          ,"word_freq_money"        ,"word_freq_hp"           ,"word_freq_hpl"          ,"word_freq_george"       ,"word_freq_650"          ,"word_freq_lab"          ,"word_freq_labs"         ,"word_freq_telnet"       ,"word_freq_857"          ,"word_freq_data"         ,"word_freq_415"          ,"word_freq_85"           ,"word_freq_technology"   ,"word_freq_1999"         ,"word_freq_parts"        ,"word_freq_pm"           ,"word_freq_direct"       ,"word_freq_cs"           ,"word_freq_meeting"      ,"word_freq_original"     ,"word_freq_project" ,"word_freq_re"  ,"word_freq_edu" ,"word_freq_table"   ,"word_freq_conference"   ,"char_freq_;" ,"char_freq_("  ,"char_freq_[" ,"char_freq_!"
           ,"char_freq_$" ,"char_freq_#" ,"capital_run_length_average" ,"capital_run_length_longest" ,"capital_run_length_total"   ,'spam']

url = 'https://raw.githubusercontent.com/akferreira/trab1_inf01017/main/spambase.data'


df = pd.read_csv(url, names=names)

total_instances = len(df)
spam_count = df['spam'].value_counts()[1]
notspam_count = df['spam'].value_counts()[1]


X = df.drop('spam', axis=1)
y = df['spam']

K = 5
folds = generate_folds(K,df,spam_count,notspam_count,total_instances)
train_test_sets = combine_folds_training_testing(folds)

knn_metrics = {'acc': [],'recall': [],'precision': [],'f1': []}
knn_stdev = {'acc': [],'recall': [],'precision': [],'f1': []}
knn_sizes = [3,5,7,9,11,13]
for knn_size in knn_sizes:
  knn = KNeighborsClassifier(n_neighbors=knn_size)
  print(f"{knn_size=}")
  
  metricsList = {'acc': [],'recall': [],'precision': [],'f1': [],'matrix': []}

  for train_test_set in train_test_sets:
    knn.fit(train_test_set['X_train'], train_test_set['y_train'])

    y_pred = knn.predict(train_test_set['X_test'])

    cm = ConfusionMatrix(train_test_set['y_test'], y_pred)
    cm_metrics = cm.metrics()

    metricsList['acc'].append(cm_metrics['acc'])
    metricsList['recall'].append(cm_metrics['recall'])
    metricsList['precision'].append(cm_metrics['precision'])
    metricsList['f1'].append(cm_metrics['f1'])

  acc_avg = mean(metricsList['acc'])
  recall_avg = mean(metricsList['recall'])
  precision_avg = mean(metricsList['precision'])
  f1_avg = mean(metricsList['f1'])

  acc_stdev = stdev(metricsList['acc'])
  recall_stdev = stdev(metricsList['recall'])
  precision_stdev = stdev(metricsList['precision'])
  f1_stdev = stdev(metricsList['f1'])

  knn_metrics['acc'].append(acc_avg)
  knn_metrics['recall'].append(recall_avg)
  knn_metrics['precision'].append(precision_avg)
  knn_metrics['f1'].append(f1_avg)

  knn_stdev['acc'].append(acc_stdev)
  knn_stdev['recall'].append(recall_stdev)
  knn_stdev['precision'].append(precision_stdev)
  knn_stdev['f1'].append(f1_stdev)


  print(f"Acurácia: {acc_avg:.4}, desvio padrão: {acc_stdev:.4}")
  print(f"Recall: {recall_avg:.4}, desvio padrão: {recall_stdev:.4}")
  print(f"Precisão: {precision_avg:.4}, desvio padrão: {precision_stdev:.4}")
  print(f"F1 score: {f1_avg:.4}, desvio padrão: {f1_stdev:.4}\n")

knn_plotting(knn_metrics,knn_stdev,knn_sizes)



    
