from statistics import stdev, mean

class ConfusionMatrix:
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
      return {'acc': self.accuracy(),'recall': self.recall(),'precision': self.precision(),'f1': self.f1_measure(),'matrix': self}

    @classmethod
    def average_matrix(cls,ConfusionMatrixList):
      TN_avg = mean([matrix.TN for matrix in ConfusionMatrixList])
      FN_avg = mean([matrix.FN for matrix in ConfusionMatrixList])
      TP_avg = mean([matrix.TP for matrix in ConfusionMatrixList])
      FP_avg = mean([matrix.FP for matrix in ConfusionMatrixList])

      return [[TN_avg,FN_avg],[FP_avg,TP_avg]]