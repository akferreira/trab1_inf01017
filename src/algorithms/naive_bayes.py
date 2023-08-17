from statistics import stdev,mean

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from utils import plotting, ConfusionMatrix


def naive_bayes(train_test_sets):
  print("=============== Inicio da Execução do Algoritmo Naive Bayes ===============")
  var_smoothing_values = [ 1e-09*(i/100) for i in range(100)]

  nb_metrics = {'acc': [],'recall': [],'precision': [],'f1': [],'matrix': []}
  nb_stdev = {'acc': [],'recall': [],'precision': [],'f1': [],'matrix': []}
  metricsList = {'acc': [],'recall': [],'precision': [],'f1': [],'matrix': []}

  for var_smoothing_current in var_smoothing_values:
    NB = GaussianNB(var_smoothing = var_smoothing_current)

    for train_test_set in train_test_sets:
      NB.fit(train_test_set['X_train'], train_test_set['y_train'])
      y_pred = NB.predict(train_test_set['X_test'])

      cm = ConfusionMatrix(train_test_set['y_test'], y_pred)
      cm_metrics = cm.metrics()

      metricsList['acc'].append(cm_metrics['acc'])
      metricsList['recall'].append(cm_metrics['recall'])
      metricsList['precision'].append(cm_metrics['precision'])
      metricsList['f1'].append(cm_metrics['f1'])
      metricsList['matrix'].append(cm_metrics['matrix'])

    acc_avg = mean(metricsList['acc'])
    recall_avg = mean(metricsList['recall'])
    precision_avg = mean(metricsList['precision'])
    f1_avg = mean(metricsList['f1'])

    acc_stdev = stdev(metricsList['acc'])
    recall_stdev = stdev(metricsList['recall'])
    precision_stdev = stdev(metricsList['precision'])
    f1_stdev = stdev(metricsList['f1'])

    nb_metrics['acc'].append(acc_avg)
    nb_metrics['recall'].append(recall_avg)
    nb_metrics['precision'].append(precision_avg)
    nb_metrics['f1'].append(f1_avg)
    nb_metrics['matrix'] = [ConfusionMatrix.average_matrix(metricsList['matrix'])]
    nb_metrics['matrix_index'] = ''

    nb_stdev['acc'].append(acc_stdev)
    nb_stdev['recall'].append(recall_stdev)
    nb_stdev['precision'].append(precision_stdev)
    nb_stdev['f1'].append(f1_stdev)

  plotting(nb_metrics,nb_stdev,[i for i in range(100)],'NB')
  print("=============== Fim da Execução do Algoritmo Naive Bayes ===============")