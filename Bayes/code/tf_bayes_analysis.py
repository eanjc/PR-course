import pickle
import numpy as np
import tensorflow as tf
import math
import codecs

filename="full_dataset_train"
prefix = "../model/"
suffix = ".pkl"

f_pw = prefix + filename + "-pw" + suffix
f_pjw = prefix + filename + "-pjw" + suffix

with open(f_pw, 'rb') as f__pw:
    with open(f_pjw, 'rb') as f__pjw:
        PW = pickle.load(f__pw)
        PJW = pickle.load(f__pjw)

print("Model has loaded .")

for i in range(10):
    v=math.log(PW[i])
    PW[i]=v

for i in range(10):
    for j in range(784):
        v=math.log(PJW[i][j])
        PJW[i][j]=v


x=tf.Variable(tf.random_normal([1,784],stddev=1,seed=1),dtype=tf.float32,name="x-Toargmax")
w=tf.Variable(tf.constant(np.array(PJW,dtype='float32').T),dtype=tf.float32,trainable=False)
b=tf.Variable(tf.constant(np.array(PW,dtype='float32').T),dtype=tf.float32,trainable=False)
yy=tf.matmul(x,w)+b
# yy=tf.nn.softmax(tf.matmul(x,w)+b)

ak=yy[0][0]
loss=-ak

get_argmax=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
clip_op = tf.assign(x, tf.clip_by_value(x, 1e-10, 1))

fo=codecs.open("../res/bayes_weight_test_argmax_0.txt",'w')
foi=codecs.open("../res/bayes_weight_test_argmax_0_img_format.txt",'w')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(clip_op)

    STEPS=100000
    for i in range(STEPS):
        cl=sess.run(loss)
        sr=sess.run(yy)
        print(sr)
        print("%d : loss = %f"%(i,cl))
        sess.run(get_argmax)
        sess.run(clip_op)
        if i%5000==0:
            xx=sess.run(x)
            print(xx)
            for v in xx:
                s = ""
                for n in v:
                    s = s + str(n) + "\t"
                s = s + "\n"
                fo.write(s)
                fo.write("\n")
                fo.flush()
            index=1
            s=1
            img_format="["
            for v in xx[0]:
                if s==1:
                    line="["
                    s=0
                if index%28!=0:
                    line=line+str(int(v*255))+","
                else:
                    line=line+str(int(v*255))+"],"
                    img_format+=line
                    s=1
                index=index+1
            img_format+="]\n"
            foi.write(img_format)
            foi.flush()