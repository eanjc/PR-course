import pickle
from code import loadData
import tensorflow as tf

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

train_y=loadData.load_train_labels()
test_y=loadData.load_test_labels()
filename="img_aft_2bw_source_svm"
f_pw = prefix + filename + "-train_set" + suffix
f_pjw = prefix + filename + "-test_set" + suffix

with open(f_pw, 'rb') as f__pw:
    with open(f_pjw, 'rb') as f__pjw:
        train_x = pickle.load(f__pw)
        test_x = pickle.load(f__pjw)

print("File has loaded .")

train_y_one_hot=[]
test_y_one_hot=[]
def y2vec(v):
    if v==0:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if v==1:
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
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

for i in range(len(train_y)):
    v=int(train_y[i])
    train_y_one_hot.append(y2vec(v))
    if i%500==0:
        print("Has trans %d Y vectors. "%i)

for i in range(len(test_y)):
    v=int(test_y[i])
    test_y_one_hot.append(y2vec(v))
    if i%500==0:
        print("Has trans %d Y vectors. "%i)

# Linear  Classify Model
BATCH_SIZE=2000
x=tf.placeholder(tf.float32,[None,28*28])
y_=tf.placeholder(tf.float32,[None,10])
y_label=tf.placeholder(tf.int64,[None])

w1=tf.Variable(tf.random_normal([28*28,10],stddev=1,seed=1),name="w1")
b1=tf.Variable(tf.random_normal([10],stddev=1,seed=1),name="b1")

t1=tf.nn.tanh(tf.matmul(x,w1)+b1)

bs=tf.Variable(tf.random_normal([10],stddev=1,seed=1),name="bs")
yy=tf.nn.softmax(t1+bs)

reg=tf.contrib.layers.l2_regularizer(0.0001)
reg_loss=reg(w1)

cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(yy,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-yy,1e-10,1.0)))
loss=cross_entropy+reg_loss

global_step=tf.Variable(0,trainable=False)
learning_rate=tf.train.exponential_decay(0.5,global_step,60000/BATCH_SIZE,0.99,staircase=True)


train_step=tf.train.AdamOptimizer(0.0005).minimize(loss,global_step=global_step)


is_correct_train=tf.equal(tf.argmax(yy,1),tf.argmax(train_y_one_hot,1))
is_correct_test=tf.equal(tf.argmax(yy,1),tf.argmax(test_y_one_hot,1))
acc_calc_test_step=tf.reduce_mean(tf.cast(is_correct_test,tf.float32))
acc_calc_train_step=tf.reduce_mean(tf.cast(is_correct_train,tf.float32))

saver=tf.train.Saver()
save_path="../model/tf_model_2.ckpl"


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    STEPS=100000
    for i in range(STEPS):
        # acc_avg=(acc1+acc2)/2
        # T=0
        # if acc_avg<0.92:
        #     sess.run(train_step,feed_dict={x:train_x,y_:train_y_one_hot})
        # else:
        #     sess.run(train_step_slow,feed_dict={x:train_x,y_:train_y_one_hot})
        #     T=1
        s=(i*BATCH_SIZE)%60000
        e=(i*BATCH_SIZE)%60000+BATCH_SIZE
        sess.run(train_step,feed_dict={x:train_x[s:e],y_:train_y_one_hot[s:e]})
        
        
        if i % 100 ==0:
            loss_current=sess.run(loss,feed_dict={x:train_x,y_:train_y_one_hot})
            print ("After %d training steps,current loss is %f ."%(i,loss_current))
            acc1=sess.run(acc_calc_train_step,feed_dict={x:train_x,y_:train_y_one_hot})
            print ("After %d training steps,current acc on train_set is %f ."%(i,acc1))
            acc2=sess.run(acc_calc_test_step,feed_dict={x:test_x,y_:test_y_one_hot})
            print("After %d training steps,current acc on test_set is %f ." % (i, acc2))

            
        if i%1000==0:
            saver.save(sess,save_path,global_step=global_step)
    



