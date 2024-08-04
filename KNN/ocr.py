import Data

Test_data_file = '/Users/krushna/ML_Algorithms/KNN/Data/t10k-images.idx3-ubyte'
Test_Lable_file = '/Users/krushna/ML_Algorithms/KNN/Data/t10k-labels.idx1-ubyte'
Train_data_file = '/Users/krushna/ML_Algorithms/KNN/Data/train-images.idx3-ubyte'
Train_Label_file = '/Users/krushna/ML_Algorithms/KNN/Data/train-labels.idx1-ubyte'

def bytes_to_int(bytes_data):
    return int.from_bytes(bytes_data , 'big')

def read_files(indi_file, n_max=None):
    images = []
    with open (indi_file ,'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max:
            n_images = n_max
        n_rows  = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))
        for i in range(n_images):
            image= []
            for row in range(n_rows):
                rows = []
                for col in range(n_cols): 
                    pixel = f.read(1)
                    rows.append(pixel)
                image.append(rows)
            images.append(image)
    return images

def read_lables(indi_file, n_max_lables=None):
    lables = []
    with open (indi_file ,'rb') as f:
        _ = f.read(4)
        n_lables = bytes_to_int(f.read(4))
        if n_max_lables:
            n_lables = n_max_lables
        for i in range(n_lables):
            label =  f.read(1)
            lables.append(label)
    return lables


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

    
def extract_features(X):
    return [flatten_list(sample) for sample in X]
    

def main():
    X_train = read_files(Train_data_file,10000)
    Y_train = read_lables(Train_Label_file)
    x_test = read_files(Train_data_file,10000)
    y_test = read_lables(Test_Lable_file)
    X_train = extract_features(Train_data_file)
    x_test = extract_features(Train_data_file)
    
    
def knn(X_train, X_test, k=3):
    
    

if __name__ == "__main__":
    main()
    