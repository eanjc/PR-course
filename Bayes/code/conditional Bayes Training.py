import tensorflow as tf
import pickle
import  math
from code import loadData
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
prefix = "../model/"
suffix = ".pkl"

filename='full_dataset_train'
f_pw = prefix + filename + "-pw" + suffix
f_pjw = prefix + filename + "-pjw" + suffix

with open(f_pw, 'rb') as f__pw:
    with open(f_pjw, 'rb') as f__pjw:
        PW = pickle.load(f__pw)
        PJW = pickle.load(f__pjw)

print("Model has loaded .")

filename="img_aft_2bw_source_svm"
f_pw = prefix + filename + "-train_set" + suffix
f_pjw = prefix + filename + "-test_set" + suffix

with open(f_pw, 'rb') as f__pw:
    with open(f_pjw, 'rb') as f__pjw:
        train_x = pickle.load(f__pw)
        test_x = pickle.load(f__pjw)

print("File has loaded .")

X_vector=[]
for i in range(len(train_x[:10000])):
    XT=[]
    for c in range(10):
        # XTC=[]
        for d in range(len(train_x[i])):
            if train_x[i][d]==1:
                XT.append(math.log( PJW[c][d]))
            else:
                XT.append(math.log( 1-PJW[c][d]))
        # XT.append(XTC)
    X_vector.append(XT)
    if i%500==0:
        print("Has trans %d X vectors. "%i)

Y_vector=[]
def y2vec(v):
    if v==0:
        return [1,0,0,0,0,0,0,0,0,0]
    if v==1:
        return [0,1,0,0,0,0,0,0,0,0]
    if v==2:
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    if v==3:
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    if v==4:
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    if v==5:
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    if v==6:
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    if v==7:
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    if v==8:
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    if v==9:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

for i in range(len(train_labels[:10000])):
    v=int(train_labels[i])
    Y_vector.append(y2vec(v))
    if i%500==0:
        print("Has trans %d Y vectors. "%i)

XTest_Vector=[]

for i in range(len(test_x[:2000])):
    XT=[]
    for c in range(10):
        # XTC=[]
        for d in range(len(test_x[i])):
            if train_x[i][d]==1:
                XT.append(math.log( PJW[c][d]))
            else:
                XT.append(math.log( 1-PJW[c][d]))
        # XT.append(XTC)
    XTest_Vector.append(XT)
    if i%500==0:
        print("Has trans %d XTest vectors. "%i)

YTest_Vector=[]
for i in range(len(test_labels[:2000])):
    v=int(test_labels[i])
    YTest_Vector.append(y2vec(v))
    if i%500==0:
        print("Has trans %d YTest vectors. "%i)

filename='train_bayes'
f_pw = prefix + filename + "-XV-10000" + suffix
f_pjw = prefix + filename + "-YV-10000" + suffix

# with open(f_pw, 'wb') as f__pw:
#     with open(f_pjw, 'wb') as f__pjw:
#         pickle.dump(X_vector,f__pw)
#         pickle.dump(Y_vector,f__pjw)
# print("File has saved")

# with open(f_pw, 'rb') as f__pw:
#     with open(f_pjw, 'rb') as f__pjw:
#         X_vector = pickle.load(f__pw)
#         Y_vector = pickle.load(f__pjw)
#
# print("Vector File has loaded .")



x=tf.placeholder(tf.float32,[None,10*28*28])
y_=tf.placeholder(tf.float32,[None,10])
t_l=tf.placeholder(tf.int64,[None])



w=tf.Variable(tf.random_normal((10*28*28,10),stddev=1,seed=1))
r=tf.matmul(x,w)
b = tf.Variable(tf.zeros([10]))
yy=tf.nn.softmax(r+b)

cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(yy,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-yy,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


correct_predition=tf.equal(tf.argmax(yy,1),t_l)
eval_step=tf.reduce_mean(tf.cast(correct_predition,tf.float32))

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    SETPS=50000
    SIZE=10000
    for i in range(SETPS):

        # for j in range(SIZE):
        sess.run(train_step,feed_dict={x:X_vector,y_:Y_vector})
        if i %50 ==0:
            loss_current=sess.run(cross_entropy,feed_dict={x:X_vector,y_:Y_vector})
            print ("After %d training steps,current loss is %f ."%(i,loss_current))
            acc=sess.run(eval_step,feed_dict={x:XTest_Vector,t_l:test_labels[:2000]})
            print ("After %d training steps,current acc on test_set is %f ."%(i,acc))



