import os
import matplotlib.pyplot as plt

# Funcao que cria a pasta para armazenar os plots
def create_algorithm_plot_dir(plot_dir_name):
  main_folder = "plots"

  if not os.path.exists(main_folder):
    raise ValueError("Pasta plots não existe")

  path = os.path.join(main_folder, plot_dir_name)

  os.mkdir(path)

  return path


def plotting(metrics,stdev,sizes,classifier_type):
  xlabelsize = 10
  ylabelsize = 15
  xlabel = ""
  path_plot = ""

  if(classifier_type.upper() == 'KNN'):
    xlabel = 'Valor de K'
    xlabelsize = 18
    path_plot = os.path.join(create_algorithm_plot_dir("knn"), classifier_type)

  elif (classifier_type.upper() == 'DTREE'):
    xlabel = 'Profundidade'
    xlabelsize = 8
    path_plot = os.path.join(create_algorithm_plot_dir("dtree"), classifier_type) 

  else:
    xlabel = 'Var smoothing. % de 1E-09'
    xlabelsize = 8
    path_plot = os.path.join(create_algorithm_plot_dir("naive_bayes"), classifier_type)

  if path_plot == "":
    raise ValueError("Erro ao criar o diretório de destino!")

  plt.figure(f"{classifier_type} Média")
  plt.xlabel(xlabel)
  plt.title(f"Métricas {classifier_type} | Média")
  plt.plot(sizes,metrics['acc'],marker='o', linestyle='--', color='r', label='Acurácia')
  plt.plot(sizes,metrics['recall'],marker='o', linestyle='--', color='b', label='Recall')
  plt.plot(sizes,metrics['precision'],marker='o', linestyle='--', color='g', label='Precisão')
  plt.plot(sizes,metrics['f1'],marker='o', linestyle='--', color='m', label='F1 score')
  plt.legend()
  plt.grid()
  plt.xticks( [size for size in sizes if ((len(sizes) < 10 or size % 5 == 0 )) ]) 
  plt.tick_params(axis='x', which='major', labelsize=xlabelsize)
  plt.tick_params(axis='y', which='major', labelsize=ylabelsize)
  plt.savefig(f'{path_plot}_metrics_avg.png')

  for matrix,index in zip(metrics['matrix'],metrics['matrix_index']):
    fig, ax = plt.subplots()
    ax.matshow(matrix,cmap=plt.cm.Blues)
    plt.title(f'{xlabel} {index=}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    ax.text(0, 0, f"TN = {round(matrix[0][0])}", va='center', ha='center',size = 20)
    ax.text(0, 1, f"FN = {round(matrix[0][1])}", va='center', ha='center',size = 20)
    ax.text(1, 0, f"FP = {round(matrix[1][0])}", va='center', ha='center',size = 20)
    ax.text(1, 1, f"TP = {round(matrix[1][1])}", va='center', ha='center',size = 20)
    plt.savefig(f'{path_plot}_{index}_confusion_matrix.png')

  plt.figure(f"{classifier_type} Acurácia Desvio padrão")
  plt.xlabel(xlabel)
  plt.title(f'{classifier_type} | Acurácia c/ Desvio padrão')
  plt.xticks( [size for size in sizes if ((len(sizes) < 10 or size % 5 == 0 )) ]) 
  plt.tick_params(axis='x', which='major', labelsize=xlabelsize)
  plt.tick_params(axis='y', which='major', labelsize=ylabelsize)
  plt.errorbar(sizes,metrics['acc'],yerr = stdev['acc'],marker='o', linestyle='--', color='r', label='Acurácia')
  plt.legend()
  plt.grid()
  plt.savefig(f'{path_plot}_metrics1_stdev.png')

  plt.figure(f"{classifier_type} Recall Desvio padrão")
  plt.xlabel(xlabel)
  plt.title(f'{classifier_type} | Recall c/ Desvio padrão')
  plt.xticks( [size for size in sizes if ((len(sizes) < 10 or size % 5 == 0 )) ]) 
  plt.tick_params(axis='x', which='major', labelsize=xlabelsize)
  plt.tick_params(axis='y', which='major', labelsize=ylabelsize)
  plt.errorbar(sizes,metrics['recall'],yerr = stdev['recall'],marker='o', linestyle='--', color='b', label='Recall')
  plt.legend()
  plt.grid()
  plt.savefig(f'{path_plot}_metrics2_stdev.png')

  plt.figure(f"{classifier_type} Precisão Desvio padrão")
  plt.xlabel(xlabel)
  plt.title(f'{classifier_type} | Precisão c/ Desvio padrão')
  plt.xticks( [size for size in sizes if ((len(sizes) < 10 or size % 5 == 0 )) ]) 
  plt.tick_params(axis='x', which='major', labelsize=xlabelsize)
  plt.tick_params(axis='y', which='major', labelsize=ylabelsize)
  plt.errorbar(sizes,metrics['precision'],yerr = stdev['precision'],marker='o', linestyle='--', color='g', label='Precisão')
  plt.legend()
  plt.grid()
  plt.savefig(f'{path_plot}_metrics3_stdev.png')

  plt.figure(f"{classifier_type} f1 Desvio padrão")
  plt.xlabel(xlabel)
  plt.title(f'{classifier_type} | F1_Score c/ Desvio padrão')
  plt.xticks( [size for size in sizes if ((len(sizes) < 10 or size % 5 == 0 )) ]) 
  plt.tick_params(axis='x', which='major', labelsize=xlabelsize)
  plt.tick_params(axis='y', which='major', labelsize=ylabelsize)
  plt.errorbar(sizes,metrics['f1'],yerr = stdev['f1'],marker='o', linestyle='--', color='m', label='F1 score')
  plt.legend()
  plt.grid()
  plt.savefig(f'{path_plot}_metrics4_stdev.png')
