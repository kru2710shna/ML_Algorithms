import Data

Test_data_file = '/Users/krushna/ML_Algorithms/KNN/Data/t10k-images.idx3-ubyte'
Rest_Lable_file = '/Users/krushna/ML_Algorithms/KNN/Data/t10k-labels.idx1-ubyte'
Train_data_file = '/Users/krushna/ML_Algorithms/KNN/Data/train-images.idx3-ubyte'
Train_Label_file = '/Users/krushna/ML_Algorithms/KNN/Data/train-labels.idx1-ubyte'

def read_files(indi_file):
    with open (indi_file ,'rb') as f:
        f.read(4)
    