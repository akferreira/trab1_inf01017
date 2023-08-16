import os
import shutil
import time

from utils.pre_processing import get_data
from algorithms import knn, decision_tree, naive_bayes, generate_folds, combine_folds_training_testing

def remove_and_create_plot_directory():
    folder_name = "plots"

    if os.path.exists(folder_name):        
        shutil.rmtree(folder_name)

    os.mkdir(folder_name)

# Inicialmente limpa a pasta que ficara os plots
remove_and_create_plot_directory()

#Obtem os dados 
df = get_data()

# Geração do K-folds
K = 5
folds = generate_folds(K,df)
train_test_sets = combine_folds_training_testing(folds)

# KNN
knn(train_test_sets)

# Decision Tree
decision_tree(train_test_sets)

# Naive_bayes
#naive_bayes(train_test_sets)





