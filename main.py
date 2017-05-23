'''
Created by friedrich12 
you must use python2 for this python3 doesn't have midi
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import midi_manipulation
from rbm_chords import gibbs_sample, sample, songs

lowest_note = midi_manipulation.lowerBound
highest_note = midi_manipulation.upperBound
note_range = highest_note-lowest_note

# timesteps that will created at a time
num_timesteps = 15
#  size of the visible layer
n_visable = 2*note_range*num_timesteps
# Size of hidden layer
n_hidden = 50
num_epochs = 200
batch_size = 100

# This is our learning rate feel free to change
lr = tf.constant(0.005, tf.float32)

# data holder
x = tf.placeholder(tf.float32, [None, n_visable], name='x')
# our weights matrix
W = \
    tf.Variable(tf.random_normal([n_visable, n_hidden],
                                 0.001), name='w')
# our bias vector
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name='bh'))
# bias vector for visiable layer
bv = tf.Variable(tf.zeros([1, n_visable], tf.float32, name='bv'))

# Samples
x_sample = gibbs_sample(1)

# stating from x
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
#from x_sample
h_sample =sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

#Updates
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder = tf.mul(lr/size_bt, tf.sub(tf.matmul(tf.transpose(x), h),
                                    tf.matmul(tf.transpose(x_sample),h_sample)))
bv_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(x, x_sample), 0, True))
bh_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(h, h_sample), 0 , True))

# Let tensorflow run update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder),
        bh.assign_add(bh_adder)]

# Run the graph
with tf.Session() as sess:
    #Initalize the model
    init = tf.initialize_all_variables()
    sess.run(init)

    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            song = np.array(song)
            # train each RBM
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i+batch_size]
                sess.run(updt, feed_dict={x: tr_x})

    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visable))})

    for i in range(sample.shape[0]):
        if not any(sample[i,:]):
            continue
        # save junk to midi file
        S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
        midi_manipulation.noteStateMatrixToMidi(S,"generated_chord_{}".format(i))