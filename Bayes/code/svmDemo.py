from SVM import svmutil
import pickle
from code import loadData

# 训练集文件
train_images_idx3_ubyte_file = '../data/train-images-idx3-ubyte/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'


prefix = "../model/"
suffix = ".pkl"

train_x=[]
train_y=loadData.load_train_labels()
test_x=[]
test_y=loadData.load_test_labels()
filename="img_aft_2bw_source_svm"
f_pw = prefix + filename + "-train_set" + suffix
f_pjw = prefix + filename + "-test_set" + suffix

with open(f_pw, 'rb') as f__pw:
    with open(f_pjw, 'rb') as f__pjw:
        train_x = pickle.load(f__pw)
        test_x = pickle.load(f__pjw)


print("File has loaded .")

svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.1 -m 1000")

svmutil.svm_save_model("../model/svm_model_rbf_0.1.model",svm_model)

# svm_model=svmutil.svm_load_model("../model/svm_model_linear.model")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)