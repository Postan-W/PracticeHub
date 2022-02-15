import tensorflow as tf
input_tensor = tf.constant(value=[[[1,1,1],[2,2,2],[3,3,3],[4,4,4]],[[1,1,1],[2,2,2],[3,3,3],[4,4,4]],[[1,1,1],[2,2,2],[3,3,3],[4,4,4]],[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]])
with tf.Session() as sess:
    print(sess.run(tf.size(input_tensor))/sess.run(tf.shape(input_tensor))[0])
