from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.misc
from six.moves import xrange
from scipy.misc import imresize
from subpixel import PS
from ops import *
from utils import *

def doresize(x, shape):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, shape)
    return y

class DCGAN(object):
    def __init__(self, sess, image_size=128, is_crop=True,
                 batch_size=64, image_shape=[128, 128, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_size = 32
        self.sample_size = batch_size
        self.image_shape = image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = 3

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_size, self.input_size, 3], name='real_images')
        # self.inputs = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3], name='real_images')

        try:
            self.up_inputs = tf.image.resize_images(self.inputs, self.image_shape[0], self.image_shape[1], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError:
            # newer versions of tensorflow
            self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        # self.images = tf.placeholder(tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape, name='sample_images')
        # self.sample_images = tf.placeholder(tf.float32, [None] + self.image_shape, name='sample_images')

        self.G = self.generator(self.inputs)
        self.G_sum = tf.image_summary("G", self.G)
        self.g_loss = tf.reduce_mean(tf.square(self.images-self.G))
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        # first setup validation data
        data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.jpg")))

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.merge_summary([self.G_sum, self.g_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_inputs = [doresize(xx, [self.input_size,]*2) for xx in sample]
        sample_images = np.array(sample).astype(np.float32)
        sample_input_images = np.array(sample_inputs).astype(np.float32)

        save_images(sample_input_images, [8, 8], './samples/inputs_small.jpg')
        '''
        for i in range(len(sample_input_images)):
            imsave2(sample_input_images[i],'./samples/input_small_%d.jpg' % (i,))
        '''
        save_images(sample_images, [8, 8], './samples/reference.jpg')
        '''
        for i in range(len(sample_images)):
            imsave2(sample_images[i],'./samples/reference_%d.jpg' % (i,))
        '''
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # we only save the validation inputs once
        have_saved_inputs = False

        for epoch in xrange(config.epoch):
            data = sorted(glob(os.path.join("./data", config.dataset, "train", "*.jpg")))
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                input_batch = [doresize(xx, [self.input_size,]*2) for xx in batch]
                batch_images = np.array(batch).astype(np.float32)
                batch_inputs = np.array(input_batch).astype(np.float32)

                # Update G network
                _, summary_str, errG = self.sess.run([g_optim, self.g_sum, self.g_loss],
                    feed_dict={ self.inputs: batch_inputs, self.images: batch_images })
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errG))

                if np.mod(counter, 100) == 1:
                    samples, g_loss, up_inputs = self.sess.run(
                        [self.G, self.g_loss, self.up_inputs],
                        feed_dict={self.inputs: sample_input_images, self.images: sample_images}
                    )
                    if not have_saved_inputs:
                        save_images(up_inputs, [8, 8], './samples/inputs.jpg')
                        have_saved_inputs = True

                    '''
                    for i in range(len(samples)):
                        print samples[i].shape
                        imsave2(samples[i],'./samples/valid_%s_%s_%d.jpg' % (epoch, idx,i))
                        # save_images(samples,[1,1])
                    '''
                    save_images(samples, [8, 8],
                                './samples/valid_%s_%s.jpg' % (epoch, idx))
                    print("[Sample] g_loss: %.8f" % (g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def generator(self, z):
        # project `z` and reshape
        self.h0, self.h0_w, self.h0_b = deconv2d(z, [self.batch_size, 32, 32, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0', with_w=True)
        h0 = lrelu(self.h0)

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, 32, 32, self.gf_dim], name='g_h1', d_h=1, d_w=1, with_w=True)
        h1 = lrelu(self.h1)

        h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, 32, 32, 3*16], d_h=1, d_w=1, name='g_h2', with_w=True)
        h2 = PS(h2, 4, color=True)

        return tf.nn.tanh(h2)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def batch_test(self, checkpoint_dir):
        if self.load(checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")
            return

        data = sorted(glob(os.path.join("./data", self.dataset_name, "test", "*.jpg")))
        batch_idxs = len(data) // self.batch_size
        batch_remain = len(data) % self.batch_size
        print "Test data length: %d" % (len(data))
        print "Batch size: %d" % (self.batch_size,)
        print "Batch idxs: %d" % (batch_idxs,)
        print "Batch remain: %d" % (batch_remain,)

        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
            input_batch = [doresize(xx, [self.input_size,]*2) for xx in batch]
            batch_images = np.array(batch).astype(np.float32)
            batch_inputs = np.array(input_batch).astype(np.float32)

            save_images(batch_inputs, [8, 8], './samples/batch_%d_small_inputs.jpg'%(idx+1,))
            '''
            for i in range(len(batch_inputs)):
                imsave2(batch_inputs[i],'./samples/batch_%d_small_inputs_%d.jpg' % (idx+1,i))
            '''
            save_images(batch_images, [8, 8], './samples/batch_%d_reference.jpg'%(idx+1,))
            '''
            for i in range(len(batch_images)):
                imsave2(batch_images[i],'./samples/batch_%d_reference_%d.jpg' % (idx+1,i))
            '''

            samples, g_loss, up_inputs = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={ self.inputs: batch_inputs, self.images: batch_images }
            )

            save_images(up_inputs, [8, 8], './samples/batch_%d_scale_straight.jpg'%(idx+1,))
            save_images(samples, [8, 8], './samples/batch_%d_test_out.jpg' % (idx+1,))

            for i in range(len(samples)):
                # print samples[i].shape
                imsave2(samples[i],'./samples/batch_%d_test_out_%d.jpg' % (idx+1, i))

            print("[Test batch %d] g_loss: %.8f" % (idx+1, g_loss))

        if batch_remain > 0:
            batch_files = data[batch_idxs * self.batch_size: len(data)]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
            img_zero = np.zeros((self.batch_size-batch_remain, self.image_size, self.image_size, 3))
            batch = np.concatenate((batch, img_zero))
            input_batch = [doresize(xx, [self.input_size, ] * 2) for xx in batch]

            batch_images = np.array(batch).astype(np.float32)
            batch_inputs = np.array(input_batch).astype(np.float32)

            save_images(batch_inputs, [8, 8], './samples/batch_remain_small_inputs.jpg')
            '''
            for i in range(len(batch_inputs)):
                imsave2(batch_inputs[i],'./samples/batch_remain_small_inputs_%d.jpg' % (i,))
            '''
            save_images(batch_images, [8, 8], './samples/batch_remain_reference.jpg')
            '''
            for i in range(len(batch_images)):
                imsave2(batch_images[i],'./samples/batch_remain_reference_%d.jpg' % (i,))
            '''

            samples, g_loss, up_inputs = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={self.inputs: batch_inputs, self.images: batch_images}
            )

            save_images(up_inputs, [8, 8], './samples/batch_remain_scale_straight.jpg')
            save_images(samples, [8, 8], './samples/batch_remain_test_out.jpg')

            for i in range(batch_remain):
                # print samples[i].shape
                imsave2(samples[i], './samples/batch_remain_test_out_%d.jpg' % (i,))

            print("[Batch remain] g_loss: %.8f" % (g_loss,))


    def batch_test2(self, checkpoint_dir):
        if self.load(checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")
            return

        data = sorted(glob(os.path.join("./data", self.dataset_name, "test", "*.jpg")))
        batch_idxs = len(data) // self.batch_size
        batch_remain = len(data) % self.batch_size
        print "Test data length: %d" % (len(data))
        print "Batch size: %d" % (self.batch_size,)
        print "Batch idxs: %d" % (batch_idxs,)
        print "Batch remain: %d" % (batch_remain,)

        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            batch = [get_image2(batch_file) for batch_file in batch_files]
            batch_inputs = np.array(batch).astype(np.float32)

            save_images(batch_inputs, [8, 8], './samples/batch_%d_small_inputs.jpg' % (idx + 1,))

            samples, up_inputs = self.sess.run(
                [self.G, self.up_inputs],
                feed_dict={ self.inputs: batch_inputs}
            )

            save_images(up_inputs, [8, 8], './samples/batch_%d_scale_straight.jpg'%(idx+1,))
            save_images(samples, [8, 8], './samples/batch_%d_test_out.jpg' % (idx+1,))

            for i in range(len(samples)):
                # print samples[i].shape
                imsave2(samples[i],'./samples/batch_%d_test_out_%d.jpg' % (idx+1, i))

        if batch_remain > 0:
            batch_files = data[batch_idxs * self.batch_size: len(data)]
            batch = [get_image2(batch_file) for batch_file in batch_files]
            img_zero = np.zeros((self.batch_size-batch_remain, self.input_size, self.input_size, 3))
            batch = np.concatenate((batch, img_zero))
            batch_inputs = np.array(batch).astype(np.float32)

            save_images(batch_inputs, [8, 8], './samples/batch_remain_small_inputs.jpg')

            samples, up_inputs = self.sess.run(
                [self.G, self.up_inputs],
                feed_dict={self.inputs: batch_inputs}
            )

            save_images(up_inputs, [8, 8], './samples/batch_remain_scale_straight.jpg')
            save_images(samples, [8, 8], './samples/batch_remain_test_out.jpg')

            for i in range(batch_remain):
                # print samples[i].shape
                imsave2(samples[i], './samples/batch_remain_test_out_%d.jpg' % (i,))


    """
    def single_test(self, checkpoint_dir):
        if self.load(checkpoint_dir):
            print(" [*] Load ckeckpoint successfully!!!")
        else:
            print(" [!] Load checkpoint failed...")
            return

        data = sorted(glob(os.path.join("./data", self.dataset_name, "test", "*.jpg")))
        idxs = len(data)
        print "Test data length: %d" % (idxs)

        for idx in xrange(0, idxs):
            files = data[idx:idx+1]

            image = [get_image(file, self.image_size, is_crop=self.is_crop) for file in files]
            input_image = [doresize(xx, [self.input_size, ] * 2) for xx in image]

            image = np.array(image).astype(np.float32)
            input_image = np.array(input_image).astype(np.float32)

            imsave2(input_image[0], './samples/input_%d.jpg' % (idx,))
            imsave2(image[0], './samples/reference_%d.jpg' % (idx,))

            sample, g_loss, up_input = self.sess.run(
                [self.G, self.g_loss, self.up_inputs],
                feed_dict={ self.inputs: input_image, self.images: image }
            )

            imsave2(up_input[0], './samples/scale_straight_%d.jpg' % (idx,))
            imsave2(sample[0], './samples/test_out_%d.jpg' % (idx,))

            print("[Test batch %d] g_loss: %.8f" % (idx+1, g_loss))
    """
