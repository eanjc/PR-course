from code import loadData
from code import imgProcess
from code import  bayesModel
import numpy as np

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

train_images_step1=np.empty((len(train_images), 26, 26))
test_images_step1=np.empty((len(test_images), 26, 26))

ip_26=imgProcess.ImgProcess()
ip_26.num_rows=26
ip_26.num_cols=26

for i in range(len(train_images)):
    img_step1=imgProcess.ImgProcess().midFilter(train_images[i],3)
    img_step2=ip_26.gray2bw_standard(img_step1)
    train_images_step1[i]=img_step2
    if(i%500==0):
        print("Has translated %d training pictures."%i)

for i in range(len(test_images)):
    img_step1=imgProcess.ImgProcess().midFilter(test_images[i],3)
    img_step2=ip_26.gray2bw_standard(img_step1)
    test_images_step1[i]=img_step2
    if(i%500==0):
        print("Has translated %d test pictures."%i)

bayesDemo=bayesModel.BayesModel(train_images_step1,train_labels,26*26)
bayesDemo.train_model()
# bayesDemo.saveModelToFile("full_dataset_train")
ACC=bayesDemo.predict(train_images_step1,train_labels)
print("Predict acc on train_set is %f ."%ACC)

ACC=bayesDemo.predict(test_images_step1,test_labels)
print("Predict acc on test_set is %f ."%ACC)