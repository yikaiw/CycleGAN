import tensorflow as tf
import layers
from reader import Reader

REAL_LABEL = 0.9

def batch_convert2int(images):
    return tf.map_fn(convert2int, images, dtype=tf.uint8)

class Generator:
    def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size
    
    def __call__(self, input):
        with tf.variable_scope(self.name):
            # The generator architecture is:
            # The network with 6 blocks consists of: c7s1-32,d64,d128,R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3
            # The network with 9 blocks consists of: c7s1-32,d64,d128,R128,R128,R128,R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3
            c7s1_32 = layers.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                                 reuse=self.reuse, name='c7s1_32')  # (?, w, h, 32)
            d64 = layers.dk(c7s1_32, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='d64')  # (?, w/2, h/2, 64)
            d128 = layers.dk(d64, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='d128')  # (?, w/4, h/4, 128)
            
            if self.image_size <= 128:
                res_output = layers.n_res_blocks(d128, reuse=self.reuse, n=6)  # (?, w/4, h/4, 128)
            else:
                res_output = layers.n_res_blocks(d128, reuse=self.reuse, n=9)  # (?, w/4, h/4, 128)
            
            u64 = layers.uk(res_output, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u64')  # (?, w/2, h/2, 64)
            u32 = layers.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u32', output_size=self.image_size)  # (?, w, h, 32)
            
            output = layers.c7s1_k(u32, 3, norm=None,
                                activation='tanh', reuse=self.reuse, name='output')  # (?, w, h, 3)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        return output
    
    def sample(self, input):
        image = batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image


class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid
    
    def __call__(self, input):
        with tf.variable_scope(self.name):
            # The discriminator architecture is: C64-C128-C256-C512
            C64 = layers.Ck(input, 64, reuse=self.reuse, norm=None,
                         is_training=self.is_training, name='C64')  # (?, w/2, h/2, 64)
            C128 = layers.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C128')  # (?, w/4, h/4, 128)
            C256 = layers.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C256')  # (?, w/8, h/8, 256)
            C512 = layers.Ck(C256, 512, reuse=self.reuse, norm=self.norm,
                          is_training=self.is_training, name='C512')  # (?, w/16, h/16, 512)
            
            output = layers.last_conv(C512, reuse=self.reuse,
                                   use_sigmoid=self.use_sigmoid, name='output')  # (?, w/16, h/16, 1)
        
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        return output


class CycleGAN:
    def __init__(self, X_train_file='', Y_train_file='', batch_size=1, image_size=256,
                 use_lsgan=True, norm='instance', lambda1=10.0, lambda2=10.0, 
                 learning_rate=2e-4, beta1=0.5, ngf=64):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        
        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
        self.D_Y = Discriminator('D_Y', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
        self.D_X = Discriminator('D_X', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.fake_x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.fake_y = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
    
    def model(self):
        X_reader = Reader(self.X_train_file, name='X',
                          image_size=self.image_size, batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file, name='Y',
                          image_size=self.image_size, batch_size=self.batch_size)
        x = X_reader.feed()
        y = Y_reader.feed()
        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)
        
        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)
        
        # Y -> X
        fake_x = self.F(y)
        F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)
        
        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))
        
        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        
        tf.summary.image('X/generated', batch_convert2int(self.G(x)))
        tf.summary.image('X/reconstruction', batch_convert2int(self.F(self.G(x))))
        tf.summary.image('Y/generated', batch_convert2int(self.F(y)))
        tf.summary.image('Y/reconstruction', batch_convert2int(self.G(self.F(y))))
        
        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x
    
    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
            
            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step
        
        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')
        
        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')
    
    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        if use_lsgan:
            error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            error_real = -tf.reduce_mean(layers.safe_log(D(y)))
            error_fake = -tf.reduce_mean(layers.safe_log(1 - D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss
    
    def generator_loss(self, D, fake_y, use_lsgan=True):
        if use_lsgan:
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
        else:
            loss = -tf.reduce_mean(layers.safe_log(D(fake_y))) / 2
        return loss
    
    def cycle_consistency_loss(self, G, F, x, y):
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss
