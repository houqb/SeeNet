import cv2
import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

class CReLULayer(caffe.Layer):
    def setup(self, bottom, top):
        ##config 
        params = eval(self.param_str)
        self.maxt = params['maxt']
        self.mint = params['mint']
        if len(bottom) != 3:
            raise Exception('Need to define three bottoms')
        if len(top) != 1:
            raise Exception('Only Need to define one top');
    
    def reshape(self, bottom, top):
        self.N = bottom[0].data.shape[0]
        top[0].reshape(*bottom[0].shape)
    
    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data
        label = bottom[2].data.reshape(20)
        #cl = np.where(label == 1)[0]

        mask = bottom[1].data[0].copy()
        mask[mask < 0] = 0
        for i in range(20):
            if int(label[i]) == 1:
                ma = np.max(mask[i])
                mi = np.min(mask[i])
                mask[i] = (mask[i] - mi) / (ma - mi + 1e-8)
            else:  # 0
                mask[i,...] = 0
        mask = mask.max(axis=0)
        self.pos = np.where(mask > self.maxt)
        top[0].data[:, :, self.pos[0], self.pos[1]] = 0
        self.pos1 = np.where(mask < self.mint)
        top[0].data[:, :, self.pos1[0], self.pos1[1]] = top[0].data[:,:,self.pos1[0], self.pos1[1]] * -1

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = top[0].diff
            bottom[0].diff[...][0, 0, self.pos[0], self.pos[1]] = 0
            #bottom[0].diff[...][0, 0, self.pos1[0], self.pos1[1]] *= -1              
            bottom[0].diff[:, :, self.pos1[0], self.pos1[1]] = bottom[0].diff[:,:,self.pos1[0], self.pos1[1]] * -1
