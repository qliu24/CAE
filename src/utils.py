import pickle
import numpy as np
import cv2
import os

class Data_loader_cifar10:
    def __init__(self, data_file_reg, idx_ls):
        # img_mean=np.array([125.3, 122.9, 113.9])/255
        img_mean=np.array([125.3, 122.9, 113.9])
        self.img_mean = img_mean.reshape(-1,1).repeat(1024, axis=1).reshape(1,-1)
        
        self.data_mat = np.zeros((0,3*32*32)).astype('uint8')
        for ii in idx_ls:
            data_file = data_file_reg.format(ii)
            with open(data_file, 'rb') as fh:
                data_dict = pickle.load(fh, encoding='bytes')

            self.data_mat = np.concatenate([self.data_mat, data_dict[bytes('data', encoding='UTF-8')].astype('uint8')], axis=0)

        self.totalN = self.data_mat.shape[0]
        print('Total data loaded: {0}'.format(self.totalN))
        
        permuted = np.random.permutation(self.totalN)
        self.eval_idx = permuted[0:int(self.totalN/5)]  # %20 for evaluation
        self.train_idx = permuted[int(self.totalN/5):]  # %80 for training
        
        self.data_mat = (self.data_mat-self.img_mean)/255  # zero centered
        
    def get_batch(self, batch_size=128, tp='train'):
        if tp=='train':
            idx_s = np.random.choice(self.train_idx, size=[batch_size,])
        else:
            idx_s = np.random.choice(self.eval_idx, size=[batch_size,])
            
        img_s = self.data_mat[idx_s].reshape(batch_size,3,32,32).transpose(0,2,3,1)
        return img_s
    
    
class Data_loader_imagenet:
    def __init__(self, file_list):
        self.scale_size = 227
        self.img_mean=np.float32([[[104., 117., 124.]]])
        
        with open(file_list, 'r') as fh:
            content = fh.readlines()
        
        self.img_list = [x.strip() for x in content]
        self.totalN = len(self.img_list)
        print('total number of images: {0}'.format(self.totalN))
        
        permuted = np.random.permutation(self.totalN)
        self.eval_idx = permuted[0:int(self.totalN/5)]  # %20 for evaluation
        self.train_idx = permuted[int(self.totalN/5):]  # %80 for training
        
    def get_batch(self, batch_size=128, tp='train'):
        if tp=='train':
            idx_s = np.random.choice(self.train_idx, size=[batch_size,])
        else:
            idx_s = np.random.choice(self.eval_idx, size=[batch_size,])
            
        img_s = np.zeros((batch_size, self.scale_size, self.scale_size, 3))
        for ib,ii in enumerate(idx_s):
            fname = self.img_list[ii]
            assert(os.path.isfile(fname))
            img = cv2.imread(fname)
            img_s[ib] = (cv2.resize(img, (self.scale_size, self.scale_size))-self.img_mean)/255
            
        return img_s
        
        
        
    