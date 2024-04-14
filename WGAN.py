import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        #[b, 100] ==> [b, 3*3*512] ==> [b, 3, 3, 512] ==>[b, 64, 64, 3]
        self.fc = layers.Dense(3*3*512)
        self.conv1 = layers.Conv2DTranspose(256, kernel_size=3, strides=3, padding='valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='valid')#kernel_size取1~7 #strides选择要使得结果为64*64
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, kernel_size=4, strides=3, padding='valid')# rgb 3个通道
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        #[z, 100] => [z, 3*3*512]
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.tanh(x)#-1~1神经网络生成 想人为观察/2 +1 *255

        return x

class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        #[b, 64, 64, 3] ==> [b, 1]

        self.conv1 = layers.Conv2D(64, kernel_size=5, strides=3, padding='valid')

        self.conv2 = layers.Conv2D(128, kernel_size=5, strides=3, padding='valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, kernel_size=5, strides=3, padding='valid')
        self.bn3 = layers.BatchNormalization()

        #[b, w, h, 3] ==> [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)#输出分类

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        #[b, w, h, 3] ==> [b, -1]
        x = self.flatten(x)
        #[b, -1]  ==> [b, 1]
        logits = self.fc(x)

        return logits

def main():

    d = Discriminator()
    g = Generator()
    #测试输入
    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])

    prob = d(x)
    print(prob.shape)
    x_hat = g(z)
    print(x_hat.shape)


if __name__ == '__main__':

    #测试
    main()



















