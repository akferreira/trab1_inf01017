# Importações necessárias
from statistics import stdev, mean
from utils import ConfusionMatrix, plotting
from sklearn.neighbors import KNeighborsClassifier

# Função que executa o algoritmo KNN
def knn(train_test_sets):
    print("=============== Inicio da Execução do Algoritmo KNN ===============")

    # Dicionários para armazenar métricas e desvios padrão
    knn_metrics = {'acc': [], 'recall': [], 'precision': [], 'f1': [], 'matrix': []}
    knn_stdev = {'acc': [], 'recall': [], 'precision': [], 'f1': []}
    
    # Lista de tamanhos de vizinhos a serem testados
    knn_sizes = [3, 5, 7, 9, 11, 13]

    # Loop sobre diferentes tamanhos de vizinhos
    for knn_size in knn_sizes:
        # Criação do classificador KNN com o tamanho de vizinhos atual
        knn = KNeighborsClassifier(n_neighbors=knn_size)
        print(f"{knn_size=}")
        
        # Dicionário para armazenar métricas para cada conjunto de treinamento/teste
        metricsList = {'acc': [], 'recall': [], 'precision': [], 'f1': [], 'matrix': []}

        # Loop sobre diferentes conjuntos de treinamento/teste
        for train_test_set in train_test_sets:
            # Treinamento do classificador KNN
            knn.fit(train_test_set['X_train'], train_test_set['y_train'])

            # Previsões usando o classificador treinado
            y_pred = knn.predict(train_test_set['X_test'])

            # Cálculo da matriz de confusão e suas métricas
            cm = ConfusionMatrix(train_test_set['y_test'], y_pred)
            cm_metrics = cm.metrics()

            # Armazenamento das métricas calculadas
            metricsList['acc'].append(cm_metrics['acc'])
            metricsList['recall'].append(cm_metrics['recall'])
            metricsList['precision'].append(cm_metrics['precision'])
            metricsList['f1'].append(cm_metrics['f1'])
            metricsList['matrix'].append(cm_metrics['matrix'])

        # Cálculo das médias e desvios padrão das métricas
        acc_avg = mean(metricsList['acc'])
        recall_avg = mean(metricsList['recall'])
        precision_avg = mean(metricsList['precision'])
        f1_avg = mean(metricsList['f1'])
        acc_stdev = stdev(metricsList['acc'])
        recall_stdev = stdev(metricsList['recall'])
        precision_stdev = stdev(metricsList['precision'])
        f1_stdev = stdev(metricsList['f1'])

        # Armazenamento das métricas médias e desvios padrão
        knn_metrics['acc'].append(acc_avg)
        knn_metrics['recall'].append(recall_avg)
        knn_metrics['precision'].append(precision_avg)
        knn_metrics['f1'].append(f1_avg)
        knn_metrics['matrix'].append(ConfusionMatrix.average_matrix(metricsList['matrix']))
        knn_metrics['matrix_index'] = knn_sizes
        knn_stdev['acc'].append(acc_stdev)
        knn_stdev['recall'].append(recall_stdev)
        knn_stdev['precision'].append(precision_stdev)
        knn_stdev['f1'].append(f1_stdev)

        # Impressão das métricas médias e desvios padrão para o tamanho de vizinho atual
        print(f"Acurácia: {acc_avg:.4}, desvio padrão: {acc_stdev:.4}")
        print(f"Recall: {recall_avg:.4}, desvio padrão: {recall_stdev:.4}")
        print(f"Precisão: {precision_avg:.4}, desvio padrão: {precision_stdev:.4}")
        print(f"F1 score: {f1_avg:.4}, desvio padrão: {f1_stdev:.4}\n")

    # Chamada à função para plotagem de gráficos
    plotting(knn_metrics, knn_stdev, knn_sizes, 'KNN')
    
    print("=============== Fim da Execução do Algoritmo KNN ===============")