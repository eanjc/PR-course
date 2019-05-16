# encoding:utf-8
from SVM import *
from SVM import svmutil
import pickle
from code import loadData
from code import  imgProcess
import numpy as np

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
filename="img_aft_2bw_midfilter_svm"
f_pw = prefix + filename + "-train_set" + suffix
f_pjw = prefix + filename + "-test_set" + suffix

with open(f_pw, 'rb') as f__pw:
    with open(f_pjw, 'rb') as f__pjw:
        train_x = pickle.load(f__pw)
        test_x = pickle.load(f__pjw)


print("File has loaded .")


# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -m 2560")

# svmutil.svm_save_model("../model/svm_model_rbf_default.model",svm_model)

# svm_model=svmutil.svm_load_model("../model/svm_model_rbf_default.model")
# print("g=0.001")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.001 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.002")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.002 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=default")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.005")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.005 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.01")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.01 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.02")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.05")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.05 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.03")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.03 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.04")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.04 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.06")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.06 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.07")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.07 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.08")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.08 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.1")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.1 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.2")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.2 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.5")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2  -g 0.5 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("linear")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 0   -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)
# print("g=0.02 N=100")
# svm_model=svmutil.svm_train(train_y[:100],train_x[:100],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:100],train_x[:100],svm_model)
# print("g=0.02 N=200")
# svm_model=svmutil.svm_train(train_y[:200],train_x[:200],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:200],train_x[:200],svm_model)
# print("g=0.02 N=500")
# svm_model=svmutil.svm_train(train_y[:500],train_x[:500],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:500],train_x[:500],svm_model)
# print("g=0.02 N=1000")
# svm_model=svmutil.svm_train(train_y[:1000],train_x[:1000],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:1000],train_x[:1000],svm_model)
# print("g=0.02 N=2000")
# svm_model=svmutil.svm_train(train_y[:2000],train_x[:2000],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:2000],train_x[:2000],svm_model)
# print("g=0.02 N=5000")
# svm_model=svmutil.svm_train(train_y[:5000],train_x[:5000],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:5000],train_x[:5000],svm_model)
# print("g=0.02 N=10000")
# svm_model=svmutil.svm_train(train_y[:10000],train_x[:10000],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:10000],train_x[:10000],svm_model)
# print("g=0.02 N=300")
# svm_model=svmutil.svm_train(train_y[:300],train_x[:300],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:300],train_x[:300],svm_model)
# print("g=0.02 N=400")
# svm_model=svmutil.svm_train(train_y[:400],train_x[:400],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:400],train_x[:400],svm_model)
# print("g=0.02 N=600")
# svm_model=svmutil.svm_train(train_y[:600],train_x[:600],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:600],train_x[:600],svm_model)
# print("g=0.02 N=700")
# svm_model=svmutil.svm_train(train_y[:700],train_x[:700],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:700],train_x[:700],svm_model)
# print("g=0.02 N=800")
# svm_model=svmutil.svm_train(train_y[:800],train_x[:800],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:800],train_x[:800],svm_model)
# print("g=0.02 N=900")
# svm_model=svmutil.svm_train(train_y[:900],train_x[:900],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:900],train_x[:900],svm_model)

# print("g=0.02 N=50")
# svm_model=svmutil.svm_train(train_y[:50],train_x[:50],"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y[:50],train_x[:50],svm_model)

# print("Sobel feature g=0.02 FullSize")
# svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.02 -m 2560")
# svmutil.svm_predict(test_y,test_x,svm_model)
# svmutil.svm_predict(train_y,train_x,svm_model)

# 中值滤波
print("Midfilter feature g=0.02 FullSize")
svm_model=svmutil.svm_train(train_y,train_x,"-c 1 -t 2 -g 0.02 -m 2560")
svmutil.svm_predict(test_y,test_x,svm_model)
svmutil.svm_predict(train_y,train_x,svm_model)
