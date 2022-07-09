import os
from datetime import datetime

import torch

from config_files import configs_ODA, configs_rot_disk
from models.SSConv import TrainSSConv


#Specify configuration file (each dataset has its own file)
par = configs_rot_disk.configs()


#Initialize new weights if required 
if par.new_weights:
    #Create name of weights 
    weights_name = 'SSConvWeights_alpha_{alpha}_vth_{vth}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_t_{time}'.format(alpha = par.SSConv_alpha, vth = par.SSConv_v_th, lambda_X = par.SSConv_lambda_X, lambda_v = par.SSConv_lambda_v, time = datetime.now())
    
    #Create directory of new weights 
    weights_name = os.path.join(par.SSConv_weights_dir, weights_name)
    
    #Initialize new weights 
    weights = par.SSConv_w_init*torch.ones(*par.SSConv_weight_shape)
 
else:
    #Get directory of trained weights 
    weights_name = os.path.join(par.SSConv_weights_dir, par.SSConv_weights_name)
    
    #Load trained weights 
    weights = torch.load(weights_name)


#Checking if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")

else:
    device = torch.device("cpu")
    print("Running on the CPU")


#Perform training
train = TrainSSConv(device, par, weights_name, SSConv_weights= weights) 
s_STDP = train.forward(device = device)

#Save result
torch.save(s_STDP.weights, weights_name)


