from code import loadData
from code import imgProcess
from code import  bayesModel
import numpy as np
import pickle

# 训练集文件
train_images_idx3_ubyte_file = '../data/train-images-idx3-ubyte/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'

train_images = loadData.load_train_images()
train_labels = loadData.load_train_labels()

test_images= loadData.load_test_images()
test_labels=loadData.load_test_labels()

train_images_step1=np.empty((len(train_images), 28, 28))
test_images_step1=np.empty((len(test_images), 28, 28))

# dd = imgProcess.ImgProcess()
# dd.num_cols = 26
# dd.num_rows = 26
for i in range(len(train_images)):
    img_step1=imgProcess.ImgProcess().gray2bw_standard(train_images[i])
    # img_step1=dd.gray2bw(imgProcess.ImgProcess().sobel(train_images[i]),T=180)
    train_images_step1[i]=img_step1
    if(i%500==0):
        print("Has translated %d training pictures."%i)

# for i in range(len(test_images)):
#     img_step1=imgProcess.ImgProcess().gray2bw_standard(test_images[i])
#     # img_step1 = dd.gray2bw(imgProcess.ImgProcess().sobel(test_images[i]), T=180)
#     test_images_step1[i]=img_step1
#     if(i%500==0):
#         print("Has translated %d test pictures."%i)

# trainsetF=open("../model/img_aft_sobel-train_set.pkl",'rb')
# testsetF=open("../model/img_aft_sobel-test_set.pkl",'rb')
# train_images_step1=pickle.load(trainsetF)
# test_images_step1=pickle.load(testsetF)


bayesDemo=bayesModel.BayesModel(train_images_step1,train_labels,28*28)

bayesDemo.train_model()
# bayesDemo.saveModelToFile("full_dataset_train")
# bayesDemo.loadModelFromFile("full_dataset_train")
ACC=bayesDemo.predict(train_images_step1,train_labels)
print("Predict acc is %f ."%ACC)
