import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

xy=np.loadtxt('img_pixels2.csv',delimiter=',',dtype=np.float32)
xy_test=np.loadtxt('img_test2.csv',delimiter=',',dtype=np.float32)
size=64**2
x_data=xy[:,0:-1]*10**-3
y_data=xy[:,[-1]]
test_data_x=xy_test[0:40,0:-1]*10**-3
test_data_y=xy_test[0:40,[-1]]
#print(x_data)
#print(y_data)
#print(test_data_x)
X = tf.placeholder(tf.float32, [None, size])

Y = tf.placeholder(tf.int32, [None, 1])

nb_classes = 4
Y_one_hot=tf.one_hot(Y,nb_classes)
Y_one_hot=tf.reshape(Y_one_hot,[-1,nb_classes])
W = tf.Variable(tf.random_normal([size, nb_classes]), name='weight')

b = tf.Variable(tf.random_normal([nb_classes]), name='bias')



# tf.nn.softmax computes softmax activations
logits=tf.matmul(X,W)+b
hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

cost_i=tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                               labels=Y_one_hot)
#cost=tf.reduce_mean(-tf.reduce_sum(Y_one_hot*tf.log(hypothesis),axis=1))
cost=tf.reduce_mean(cost_i)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction=tf.arg_max(hypothesis,1)
is_corrent=tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y_one_hot,1))
accuracy=tf.reduce_mean(tf.cast(is_corrent,tf.float32))
data_size=440
training_epochs=4000
batch_size=40
sta=0
step_ac=[]
step_cost=[]
step=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(Y_one_hot))
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(data_size/batch_size)
        for i in range(total_batch):
            if sta >= data_size:
                sta = 0
            batch_xs=x_data[int(sta):int(sta+batch_size),:]
            batch_ys=y_data[int(sta):int(sta+batch_size),:]
            c,_=sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost+=c/total_batch
            sta = sta + batch_size
           # print("Working...",sta)
       # if tf.is_nan(c)==True :
        print('Epoch:', '%04d' % (epoch + 1),'cost ={:.9f}'.format(avg_cost))
        step_ac.append(accuracy.eval(session=sess, feed_dict={X: test_data_x, Y: test_data_y}))
        step_cost.append(avg_cost)
        step.append(epoch)
    print("Learning finished")

    # Test the model using test sets

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={

        X: test_data_x, Y: test_data_y}))
    #pred=sess.run(prediction,feed_dict={X:test_data_x})
    a=sess.run(hypothesis,feed_dict={X:test_data_x})
    #print(a,"\n",sess.run(tf.arg_max(a,1)))
    print(sess.run(tf.arg_max(a,1)))
    print(test_data_y)
    plt.subplot(211)
    plt.plot(step,step_cost)
    plt.subplot(212)
    plt.plot(step,step_ac)
    plt.show()
#def next_batch():
