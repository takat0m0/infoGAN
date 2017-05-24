#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator

from tf_util import flatten

class Model(object):
    def __init__(self, class_num, z_dim, batch_size):

        self.input_size = 32
        self.class_num = class_num
        self.z_dim = z_dim
        self.batch_size = batch_size

        self.Lambda = 10
        
        self.lr = 0.001
        
        # generator config
        gen_layer = [512, 256, 128, 1]
        gen_in_dim = int(self.input_size/2**(len(gen_layer) - 1))

        #discriminato config
        disc_layer = [1, 64, 128, 256]

        # -- generator -----
        self.gen = Generator([u'gen_reshape', u'gen_deconv'],
                             gen_in_dim, gen_layer)

        # -- discriminator --
        self.disc = Discriminator([u'disc_conv', u'disc_fc'], disc_layer)

        # -- q ---------------
        self.Q_value = Discriminator([u'Q_val_conv', u'Q_val_fc'],
                                     disc_layer, class_num)
        
    def set_model(self):

        # -- define place holder -------
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.c = tf.placeholder(tf.float32, [self.batch_size, self.class_num])
        self.figs= tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 1])
        #figs_ = flatten(self.figs)
        
        # -- generator -----------------
        gen_figs = self.gen.set_model(self.c, self.z, self.batch_size, True, False)
        g_logits = self.disc.set_model(gen_figs, True, False)

        self.g_obj =  tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = g_logits,
                labels = tf.ones_like(g_logits)
            )
        )

        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- q loss ------------------
        q_logits = self.Q_value.set_model(gen_figs, True, False)
        self.q_obj = self.Lambda * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits = q_logits,
                labels = self.c
            )
        )
        train_var = self.gen.get_variables() + self.Q_value.get_variables()
        self.train_q  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = train_var)

        # -- discriminator --------
        d_logits = self.disc.set_model(self.figs, True, True)

        d_obj_true = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = d_logits,
                labels = tf.ones_like(d_logits)
            )
        )
        d_obj_false = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = g_logits,
                labels = tf.zeros_like(g_logits)
            )
        )
        self.d_obj = d_obj_true + d_obj_false
        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())
        
        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.c, self.z, self.batch_size, False, True)
        
    def training_gen(self, sess, c_list, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.c:c_list,
                                         self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, c_list, z_list, figs):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {self.c:c_list,
                                         self.z: z_list,
                                         self.figs:figs})
        return d_obj
    
    def training_q(self, sess, c_list, z_list):
        _, d_obj = sess.run([self.train_q, self.q_obj],
                            feed_dict = {self.c:c_list,
                                         self.z: z_list,
                                         })
        return d_obj
    
    def gen_fig(self, sess, c, z):
        ret_ = sess.run(self.gen_figs,
                       feed_dict = {self.c:c, self.z: z})
        ret = []
        for fig in ret_:
            ret.append(np.reshape(fig, [32, 32, 1]))
        return ret

if __name__ == u'__main__':
    model = Model(class_num = 10, z_dim = 30, batch_size = 100)
    model.set_model()
    
