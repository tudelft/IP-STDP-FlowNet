import os
from datetime import datetime

import torch

from config_files import configs_ODA, configs_rot_disk
from models.MSConv import TrainMSConv



#Specify which parameters to use 
par = configs_rot_disk.configs()
# par = configs_ODA.configs()


#Get directory of trained weights 
weights_name_SSConv = os.path.join(par.SSConv_weights_dir, par.SSConv_weights_name)

#Load trained weights 
weights_SSConv = torch.load(weights_name_SSConv)


#Initialize new weights if required 
if par.new_weights:
    weights_name_exc = 'MSConvWeights_exc_alpha_{alpha}_vth_{vth}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_tau_max_{tau_max}_t_{time}'.format(alpha = par.MSConv_alpha, vth = par.MSConv_v_th, lambda_X = par.MSConv_lambda_X, lambda_v = par.MSConv_lambda_v, tau_max = par.MSConv_tau_max, time = datetime.now())
    weights_name_exc = os.path.join(par.MSConv_weights_dir, weights_name_exc)

    weights_name_inh = 'MSConvWeights_inh_alpha_{alpha}_vth_{vth}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_tau_max_{tau_max}_t_{time}'.format(alpha = par.MSConv_alpha, vth = par.MSConv_v_th, lambda_X = par.MSConv_lambda_X, lambda_v = par.MSConv_lambda_v, tau_max = par.MSConv_tau_max, time = datetime.now())
    weights_name = os.path.join(par.MSConv_weights_dir, weights_name_inh)
    
    weights_exc = par.MSConv_w_init_exc*torch.ones(*par.MSConv_weight_shape)
    weights_inh = torch.zeros(*par.MSConv_weight_shape)


else:
    #Get directory of trained weights 
    weights_name_exc = os.path.join(par.MSConv_weights_dir, par.MSConv_weights_name +'_exc.pt')
    weights_name_inh = os.path.join(par.MSConv_weights_dir, par.MSConv_weights_name +'_inh.pt')
    
    #Load trained weights 
    weights_exc = torch.load(weights_name_exc)
    weights_inh = torch.load(weights_name_inh)

    
#Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")

else:
    device = torch.device("cpu")
    print("Running on the CPU")

train = TrainMSConv(device, par = par, weights_name_exc = weights_name_exc, weights_name_inh = weights_name_inh, SSConvWeights = weights_SSConv, MSConvWeights_exc = weights_exc, MSConvWeights_inh = weights_inh) 
s_STDP_exc, s_STDP_inh = train.forward(device = device)

#Saving results
torch.save(s_STDP_exc.weights, weights_name_exc)
torch.save(s_STDP_inh.weights,  weights_name_inh)






