
import bz2
import os
from typing import NamedTuple, Tuple

import _pickle as cPickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tools.plotting as plot
import torch
from functional.STDP_federico2 import STDPFedericoState
from module.STDP_federico import (STDPFedericoFeedForwardCell,
                                  STDPFedericoParameters)

from models.layer_definitions import Input, SSConv


class TrainSSConv(torch.nn.Module):
    """ This class can be used to train the SSConv layer in Federicos SNN.

    Parameters: 
        device (torch.device): device on which training should be performed (CPU/GPU)
        par (object): SNN parameters
        height (int): number of pixels in input images in vertical direction 
        width (int): number of pixels in input image in horizontal direction 
        method (string): 
        alpha (int): 
        dt(float): time step of the simulation 
    """
    
    def __init__(
        self, device, par, weights_name, method = "super", alpha = 100, SSConv_weights = "SSConvWeights.pt"
    ):
        super(TrainSSConv, self).__init__()
 
        
        self.par = par

        #Initialize network layers 
        self.input = Input(self.par)
        self.noDyn_input = self.input.noDyn_input

        self.SSConv = SSConv(self.par)
        self.lif_SSConv = self.SSConv.lif_SSConv

        #Load weights 
        self.SSConv_weights = SSConv_weights.to(device).to(torch.float32)
 
        self.weights_name = weights_name

        #Setting up STDP rule 
        self.STDP_params = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.SSConv_w_init, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP = STDPFedericoFeedForwardCell(self.par.SSConv_weight_shape, (self.par.SSConv_out_dim, self.SSConv.conv_SSConv_dim[0], self.SSConv.conv_SSConv_dim[1]), p = self.STDP_params, dt = self.par.dt)


    def forward(self,  
        device: torch.device 
    ) -> Tuple[torch.Tensor, STDPFedericoState]:

        """ Compute STDP updates for SSConv layer 

        Parameters: 
            device (torch.device): device on which computations shall be performed
        """
     
        #Initialize STDP states 
        s_STDP = self.STDP.initial_state(self.par.batch_size, weights = self.SSConv_weights, device = device, dtype = torch.float32 )

        #Initialize Input states
        s_input = self.noDyn_input.initial_state(self.par.batch_size, device = device, dtype = torch.float32)
        
        #Initialize SSConv states 
        s_SSConv = self.lif_SSConv.initial_state_WTA(self.par.batch_size, device = device, dtype = torch.float32)
         
     
        for sequence in range(self.par.iterations):
            
            #Randomly select sequence from directory 
            random_files = np.random.choice(os.listdir(self.par.directory), self.par.batch_size)
            
            #Load first sequence
            data = bz2.BZ2File(self.par.directory + '/{}'.format(random_files[0]), 'rb')
            x = cPickle.load(data)
            
            #Determine sequence length
            seq_length = x.shape[0]
        
            #Add remaining sequences to batch 
            for sequence in range(1,len(random_files)):
                file_name = self.par.directory + '/{}'.format(random_files[sequence])
                load = torch.load(file_name)[:int(seq_length/2)]
                x = torch.cat((x, load),dim = 1)

            #Randomly decide whether or not to flip image in x and y direction, polarity and time
            flip_w, flip_h, flip_p = tuple(np.random.randint(0,2,3))

            #Aplly flips
            x = (1 - flip_w) * x + flip_w * x[:,:, :,:,  torch.arange(x.shape[4] -1, -1, -1)]
            x = (1 - flip_h) * x + flip_h * x[:, :, :, torch.arange(x.shape[3] -1, -1, -1)]
            x = (1 - flip_p) * x + flip_p * x[:,:, torch.arange(x.shape[2] -1, -1, -1)]
       
            #Initialize buffers for plotting 
            input_buffer = torch.zeros(30, self.par.batch_size, self.par.input_out_dim, self.par.SSConv_m, *self.input.conv_input_dim[0:2]).to(device)
            SSConv_buffer = torch.zeros(75, self.par.batch_size, self.par.SSConv_out_dim, self.par.merge_m, *self.SSConv.conv_SSConv_dim[0:2]).to(device)
            
            for ts in range(seq_length):
                
                #Only put input at current time step on GPU
                x_ts = x[ts, :].to(device)     

                #Compute total time
                time = sequence*seq_length + ts

                
               
                ###################Input layer###################
                # #Plot input before downsampling
                # plot.plot2D_discrete(x_ts, self.par.height, self.par.width, 0, 'input', (600, 720), (450, 250), 1, dt = self.par.dt) 

                #Compute Input layer output spikes and new state
                z, s_input = self.input.forward(x_ts, s_input, self.par.batch_size, (self.par.input_kernel, self.par.input_stride, self.par.input_padding))    
            
                #Buffer output spikes 
                input_buffer, input_buffer_sum = plot.rotate_buffer(z, input_buffer)

                #Plot output spikes of input layer
                plot.plot2D_discrete(input_buffer_sum[:, :,0], self.input.conv_input_dim[0], self.input.conv_input_dim[1], 0, 'downsampled input0', (1, 300), (450, 250), 1, dt = self.par.dt)  
              
                
                
                ###################SSConv layer###################
                #Compute SSConv layer output spikes and new state
                z, s_SSConv, plot_vth_SSConv, boxes, box_indices = self.SSConv.forward(z, s_SSConv, self.par, self.SSConv_weights, s_input.X, device, 'SSConv', False, self.par.training)
                
                #Buffer output spikes 
                SSConv_buffer, SSConv_buffer_sum = plot.rotate_buffer(z, SSConv_buffer)

                # #Plot presynaptic traces of SSConv layer
                # image_pst = plot.plot_presynaptic_traces(s_input.X, self.par.map_colours, 1, 2, boxes, box_indices, self.par.MSConv_padding3D, True,  0, 'pst_test', (0, 400), (960, 500), 1, device)

                #Plot output spikes of SSConv layer in one window
                image_output = plot.plot_output_spikes_together(SSConv_buffer_sum, self.par.map_colours, 0, 0, 'SSConv_output_spikes_together0', (700, 600), (450,250), 1, device)
            
                ###################Training###################
                if self.par.training:
                    #Performing STDP rule
                    s_STDP, X =  self.STDP.forward1(s_input.X, z, torch.tensor(self.par.SSConv_kernel3D).to(device), torch.tensor(self.par.SSConv_stride3D).to(device) , torch.tensor(self.par.SSConv_padding3D).to(device), device, s_STDP)

                #Plotting SSConv weights
                image_weights = plot.plot_weights_SSConv(s_STDP.weights, 'SSConv_weights_after', (1,1),(1050, 190), True, 1, device )

                # #Plotting colour legend for SSConv layer 
                # plot.plot_SSConv_colour_legend(s_STDP.weights, 'SSConv_colour_legend', (700,400),(1050, 190), True, 1, device, self.par.map_colours)
                
                if time%100 == 0:
                    print('--------------------')
                    print('Sequence: ', sequence+1, '/', self.par.iterations)
                    print('Time: ', time)
                    print('--------------------\n')

            #Save weights
            torch.save(s_STDP.weights, self.weights_name)

        return s_STDP



 

    


 