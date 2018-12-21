import os
import torch

class BaseModel(object):
    '''Base model for gans_detection, and define some base function 
    '''
    def __init__(self, opt):
        self.opt = opt
        self.mode = opt.mode
        # self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.Tensor = torch.FloatTensor
        self.save_dir = os.path.join(opt.save_dir, opt.name)

    def draft_data(self, input):
        self.input = input

    def backward_D(self):
        pass
    
    def backward_G(self):
        pass
    
    def backward_L(self):
        pass

    def pre_train(self):
        pass
    
    def train(self):
        pass

    def test(self):
        pass
    
    def save_network(self):
        pass

    def save_image(self):
        pass

    def save(self):
        pass

    def load_network(self):
        pass

    def visual(self):
        pass

    def name(self):
        return 'BaseModel'
    
    def info(self):
        '''print the infomation of this model, contain the network arch,
        ...networh input
        '''
        pass