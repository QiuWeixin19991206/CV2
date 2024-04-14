import tensorflow as tf

def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]
    #[b, h, w, c]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [b, h, w, c]==>[b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)
    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])   #varible类型就不用跟踪，tensor需要跟踪
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads: [b, h, w, c] =>[b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) #[b] 二范数
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp




