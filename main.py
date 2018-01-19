import tensorflow as tf
from structure import CycleGAN
from datetime import datetime
import os

batch_size = 1
image_size = 256
lambda1 = 10.0  # weight for forward cycle loss (X->Y->X)
lambda2 = 10.0  # weight for backward cycle loss (Y->X->Y)
learning_rate = 2e-4  # learning rate for Adam
beta1 = 0.5  # momentum term of Adam
pool_size = 50  # size of image buffer that stores previously generated images
ngf = 64  # number of gen filters in first conv layer
norm = 'instance'  # [instance, batch] use instance norm or batch norm
use_lsgan = True  # use lsgan (mean squared error) or cross entropy loss
X = 'data/tfrecords/photo.tfrecords'  # X tfrecords file for training
Y = 'data/tfrecords/ukiyoe.tfrecords'  # Y tfrecords file for training

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('pre_trained', None, 'for continue training')

class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []
    
    def query(self, image):
        if self.pool_size == 0:
            return image
        
        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image

def main():
    if FLAGS.pre_trained is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.pre_trained.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            print("Unable to make checkpoints direction: %s" % checkpoints_dir)
            pass
    
    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(X, Y, batch_size, image_size, use_lsgan, norm, lambda1, lambda2, learning_rate, beta1, ngf)
        G_loss, DY_loss, F_loss, DX_loss, fake_y, fake_x = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, DY_loss, F_loss, DX_loss)
        
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as sess:
        if FLAGS.pre_trained is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            fake_Y_pool = ImagePool(pool_size)
            fake_X_pool = ImagePool(pool_size)
            
            while not coord.should_stop():
                fake_y_val, fake_x_val = sess.run([fake_y, fake_x])
                
                _, G_loss_val, DY_loss_val, F_loss_val, DX_loss_val, summary = (
                    sess.run([optimizers, G_loss, DY_loss, F_loss, DX_loss, summary_op],
                        feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                                   cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
                    )
                )
                
                train_writer.add_summary(summary, step)
                train_writer.flush()
                
                if step % 100 == 0:
                    print('Step %d:' % step)
                    print('  G_loss   : {}'.format(G_loss_val))
                    print('  DY_loss : {}'.format(DY_loss_val))
                    print('  F_loss   : {}'.format(F_loss_val))
                    print('  DX_loss : {}'.format(DX_loss_val))
                
                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                
                step += 1
        
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            print("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
