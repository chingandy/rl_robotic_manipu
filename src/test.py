import tensorflow as tf
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
import parser
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

from keras import backend as K
K.set_session(sess)


# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

from keras.layers import Dense

# Keras layers can be called on TensorFlow tensors:
with tf.device('/GPU:0 '):
#with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):
    x = Dense(128, activation='relu')(img)
    x = Dense(128, activation='relu')(x)
    preds = Dense(10, activation='softmax')(x)

# the placeholder for the labels and the loss function
labels = tf.placeholder(tf.float32, shape=(None, 10))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0], labels: batch[1], K.learning_phase(): 1})

# Evalutate the model
from keras.metrics import categorical_accuracy as accuracy 

acc_value = accuracy(labels, preds)
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels}))




