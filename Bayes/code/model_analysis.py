import tensorflow as tf
import numpy as np
import codecs

# ckpt = tf.train.get_checkpoint_state('../model/')       # 通过检查点文件锁定最新的模型
saver = tf.train.import_meta_graph("../model/tf_model_1.ckpl-99001.meta") # 载入图结构，保存在.meta文件中
reader = tf.train.NewCheckpointReader('../model/tf_model_1.ckpl-99001')
# saver = tf.train.import_meta_graph("../model/tf_model_2.ckpl-50001.meta") # 载入图结构，保存在.meta文件中
# reader = tf.train.NewCheckpointReader('../model/tf_model_2.ckpl-50001')
m = reader.get_variable_to_dtype_map()
s= reader.get_variable_to_shape_map()
# w=reader.get_tensor("Variable")
print(m)
print(s)
print(reader.get_tensor("Variable"))
print("--")
print(reader.get_tensor("Variable/Adam"))
print("--")
print(reader.get_tensor("Variable/Adam_1"))

x=tf.Variable(tf.random_normal([1,784],stddev=1,seed=1),name="x_argmax")
w1=reader.get_tensor("Variable")
b1=reader.get_tensor("Variable_1")
w1=tf.Variable(tf.constant(w1,dtype=tf.float32),trainable=False)
b1=tf.Variable(tf.constant(b1,dtype=tf.float32),trainable=False)
t1=tf.nn.relu(tf.matmul(x,w1)+b1)

w2=reader.get_tensor("Variable_2")
b2=reader.get_tensor("Variable_3")
w2=tf.Variable(tf.constant(w2,dtype=tf.float32),trainable=False)
b2=tf.Variable(tf.constant(b2,dtype=tf.float32),trainable=False)

t2=tf.nn.relu(tf.matmul(t1,w2)+b2)

w3=reader.get_tensor("Variable_4")
w3=tf.Variable(tf.constant(w3,dtype=tf.float32),trainable=False)
b3=reader.get_tensor("Variable_5")
b3=tf.Variable(tf.constant(b3,dtype=tf.float32),trainable=False)

t3=tf.nn.relu(tf.matmul(t2,w3)+b3)

bs=reader.get_tensor("Variable_6")
bs=tf.Variable(tf.constant(bs,dtype=tf.float32),trainable=False)

# w1=reader.get_tensor("Variable/Adam")
# b1=reader.get_tensor("Variable_1/Adam")
# w1=tf.Variable(tf.constant(w1,dtype=tf.float32),trainable=False)
# b1=tf.Variable(tf.constant(b1,dtype=tf.float32),trainable=False)
# t1=tf.nn.relu(tf.matmul(x,w1)+b1)
#
# w2=reader.get_tensor("Variable_2/Adam")
# b2=reader.get_tensor("Variable_3/Adam")
# w2=tf.Variable(tf.constant(w2,dtype=tf.float32),trainable=False)
# b2=tf.Variable(tf.constant(b2,dtype=tf.float32),trainable=False)
#
# t2=tf.nn.relu(tf.matmul(t1,w2)+b2)
#
# w3=reader.get_tensor("Variable_4/Adam")
# w3=tf.Variable(tf.constant(w3,dtype=tf.float32),trainable=False)
# b3=reader.get_tensor("Variable_5/Adam")
# b3=tf.Variable(tf.constant(b3,dtype=tf.float32),trainable=False)
#
# t3=tf.nn.relu(tf.matmul(t2,w3)+b3)
#
# bs=reader.get_tensor("Variable_6/Adam")
# bs=tf.Variable(tf.constant(bs,dtype=tf.float32),trainable=False)
# yy=t3+bs
yy=tf.nn.softmax(t3+bs)


a0=yy[0][9]
print(a0)
loss=-a0
loss2=-a0+tf.reduce_sum(tf.abs(x-0.5))

clip_op = tf.assign(x, tf.clip_by_value(x, 1e-10, 1))
get_argmax=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# print(w)
# print(type(w))
#
fo=codecs.open("../res/weight_test_argmax_9_conf1.txt",'w')
foi=codecs.open("../res/weight_test_argmax_9_img_format_conf1.txt",'w')

with tf.Session() as sess:
    tf.global_variables_initializer().run()

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



