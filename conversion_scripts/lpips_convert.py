import nobuco
from nobuco import ChannelOrder
import numpy as np
import lpips
import tensorflow as tf
import os, sys

tf.config.set_visible_devices([], 'GPU')

def convert_to_keras(
    spatial=True,           # Return a spatial map of perceptual distance.
    backbone='squeeze',     # net = 'squeeze', 'vgg', 'alex'
    device='cpu',
    trace_html=False):

    model = lpips.LPIPS(net=backbone, spatial=spatial).to(device)
    # model = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

    ex_ref = lpips.im2tensor(lpips.load_image('./imgs/ex_ref.png'))
    ex_p0 = lpips.im2tensor(lpips.load_image('./imgs/ex_p0.png'))
    ex_p1 = lpips.im2tensor(lpips.load_image('./imgs/ex_p1.png'))

    model.forward(ex_ref, ex_p0)

    keras_model = nobuco.pytorch_to_keras(
        model,
        args=[ex_ref, ex_p0], kwargs=None,
        inputs_channel_order=ChannelOrder.TENSORFLOW,
        trace_shape=True,
        save_trace_html=trace_html
    )

    # Testing conversion
    ex_ref_tf = tf.convert_to_tensor(
            np.moveaxis(ex_ref.numpy(), 1, -1)
        )
    ex_p0_tf = tf.convert_to_tensor(
            np.moveaxis(ex_p0.numpy(), 1, -1)
        )
    ex_p1_tf = tf.convert_to_tensor(
            np.moveaxis(ex_p1.numpy(), 1, -1)
        )

    print("Tensorflow:", ex_ref_tf.shape)
    print("Pytorch:", ex_ref.shape)


    # Test outputs
    dist_1 = model.forward(ex_ref, ex_p0)
    dist_2 = model.forward(ex_ref, ex_p1)

    dist_1_tf = keras_model([ex_ref_tf, ex_p0_tf])
    dist_2_tf = keras_model([ex_ref_tf, ex_p1_tf])

    print('\n\tBackbone:\t%s'%backbone, '\n\tSpatial:\t%d'%spatial, '\n')
    if not spatial:
        print('Distances (torch): (%.3f, %.3f)'%(dist_1, dist_2))
        print('Distances (keras): (%.3f, %.3f)'%(dist_1_tf, dist_2_tf))
    else:
        print('Distances (torch): (%.3f, %.3f)'%(dist_1.mean(), dist_2.mean()))
        print('Distances (keras): (%.3f, %.3f)'%(dist_1_tf.numpy().mean(), dist_2_tf.numpy().mean()))
        
    return keras_model
        
def main():
    print_summary=False
    mdl_dir = os.path.join(os.path.expanduser('~'), 'compiled')
    if not os.path.exists(mdl_dir):
        os.mkdir(mdl_dir)
    print("Generating keras implementations for each Lpips option")
    for net in ['squeeze', 'alex', 'vgg']:
        tmp = convert_to_keras(spatial=True, backbone=net)
        tmp.save(os.path.join(mdl_dir, 'lpips_%s_spatial.keras'%(net)))
        if print_summary: print(tmp.summary())
        tmp = convert_to_keras(spatial=False, backbone=net)
        tmp.save(os.path.join(mdl_dir, 'lpips_%s.keras'%(net)))
        if print_summary: print(tmp.summary())
    print("Done")
    
if __name__ == '__main__':
    sys.exit(main())