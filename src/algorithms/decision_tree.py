# Importando classes e funções necessárias
from sklearn.tree import DecisionTreeClassifier
from statistics import mean, stdev
from utils import plotting, names, ConfusionMatrix

# Função que executa o algoritmo Decision Tree
def decision_tree(train_test_sets):
    print("=============== Inicio da Execução do Algoritmo Decision Tree ===============")
    
    # Dicionários para armazenar métricas de desempenho e desvio padrão
    dtree_metrics = {'acc': [], 'recall': [], 'precision': [], 'f1': [], 'matrix': [], 'matrix_index': []}
    dtree_stdev = {'acc': [], 'recall': [], 'precision': [], 'f1': []}
    
    # Definindo uma faixa de profundidade para a Decision Tree (ignorando os primeiros 2 valores)
    depth_range = [depth for depth in range(len(names)) if depth > 2]

    # Iterando sobre diferentes profundidades da Decision Tree
    for current_depth in depth_range: 
        # Dicionário para armazenar métricas para cada conjunto de treinamento/teste
        metricsList = {'acc': [], 'recall': [], 'precision': [], 'f1': [], 'matrix': []}

        # Iterando sobre os conjuntos de treinamento/teste fornecidos
        for train_test_set in train_test_sets:
            # Criando um classificador Decision Tree com a profundidade atual
            dtree = DecisionTreeClassifier(random_state=0, max_depth=current_depth)
            dtree.fit(train_test_set['X_train'], train_test_set['y_train'])
            y_pred = dtree.predict(train_test_set['X_test'])
            cm = ConfusionMatrix(train_test_set['y_test'], y_pred)
            cm_metrics = cm.metrics()

            # Armazenando métricas para cada conjunto de treinamento/teste
            metricsList['acc'].append(cm_metrics['acc'])
            metricsList['recall'].append(cm_metrics['recall'])
            metricsList['precision'].append(cm_metrics['precision'])
            metricsList['f1'].append(cm_metrics['f1'])
            metricsList['matrix'].append(cm_metrics['matrix'])

        # Calculando médias das métricas para a profundidade atual
        acc_avg = mean(metricsList['acc'])
        recall_avg = mean(metricsList['recall'])
        precision_avg = mean(metricsList['precision'])
        f1_avg = mean(metricsList['f1'])

        # Calculando desvios padrão das métricas para a profundidade atual
        acc_stdev = stdev(metricsList['acc'])
        recall_stdev = stdev(metricsList['recall'])
        precision_stdev = stdev(metricsList['precision'])
        f1_stdev = stdev(metricsList['f1'])

        # Armazenando médias e desvios padrão no dicionário dtree_metrics e dtree_stdev
        dtree_metrics['acc'].append(acc_avg)
        dtree_metrics['recall'].append(recall_avg)
        dtree_metrics['precision'].append(precision_avg)
        dtree_metrics['f1'].append(f1_avg)
        if (current_depth % 10 == 0 or current_depth == 3):
            dtree_metrics['matrix'].append(ConfusionMatrix.average_matrix(metricsList['matrix']))
            dtree_metrics['matrix_index'].append(current_depth)

        dtree_stdev['acc'].append(acc_stdev)
        dtree_stdev['recall'].append(recall_stdev)
        dtree_stdev['precision'].append(precision_stdev)
        dtree_stdev['f1'].append(f1_stdev)

    # Chama a função de plotagem com as métricas e desvios padrão calculados
    plotting(dtree_metrics, dtree_stdev, depth_range, 'dtree')
    
    print("=============== Fim da Execução do Algoritmo Decision Tree ===============")