import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

def plot(samples):

    x_dim=samples.shape[1]
    color=samples.shape[3]
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if samples.shape[3] == 3:
            sample = sample.reshape(x_dim, x_dim, color)
            plt.imshow(sample)
        else:
            sample = sample.reshape(x_dim, x_dim)
            plt.imshow(sample, cmap='Greys_r')
         
    return fig


def generateSamples(out_dir, z_dim=100):
#     fileNames=[]
    if not os.path.exists(out_dir+'/generated/'):
            os.makedirs(out_dir+'/generated/')

    for root, dirs, files in os.walk(out_dir+"/model/"):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() =='.meta':
                model=root+os.path.splitext(filename)[0]
                imageName=os.path.splitext(filename)[0]
                print(model)
#                 fileNames.append(root+os.path.splitext(filename)[0])

                tf.reset_default_graph()
                with tf.Session() as sess:
                #     z = tf.placeholder(tf.float32, shape=[None, z_dim])
                #     saver = tf.train.Saver()
                    saver=tf.train.import_meta_graph(model+'.meta')
                    saver.restore(sess, model)
                    graph=tf.get_default_graph()

                    tName1=graph.get_operation_by_name('z').name+':0'
                    z=graph.get_tensor_by_name(tName1)

                    tName2=graph.get_operation_by_name('generator/final_gen').name+':0'
                    gen=graph.get_tensor_by_name(tName2)

                    np.random.seed(42)

                    batch_z = np.random.normal(-1.0, 1.0, size=[16, z_dim]).astype(np.float32)

                    samples = sess.run(gen, feed_dict={z: batch_z})

                    fig = plot(samples)

                    plt.savefig(out_dir+'/generated/{}.png'
                                .format(imageName), bbox_inches='tight')
                    
                    plt.show()
                    plt.close()
                    
                    
                