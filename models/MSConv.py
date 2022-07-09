
import bz2
import os
import random
from typing import NamedTuple, Tuple

import _pickle as cPickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools.plotting as plot
import torch
from functional.lif_mod2 import linspace
from functional.STDP_federico2 import STDPFedericoState
from module.STDP_federico import (STDPFedericoFeedForwardCell,
                                  STDPFedericoParameters)
from tools.OF_vectors import compute_OF, flow_viz_np, plot_OF

from models.layer_definitions import Input, Merge, MSConv, SSConv


class TrainMSConv(torch.nn.Module):
    """ Perform training/inference with MSConv layer

    Parameters: 
        device (torch.device): device on which training should be performed (CPU/GPU)
        par (SNN_param): parameters of the SNN
        weights_name_exc (str): name of excitatory MSConv weights
        weights_name_inh (str): name of inhibitory MSConv weights
        SSConv_weights (torch.Tensor): trained SSConv weights 
        MSConvWeights_exc (torch.Tensor): excitatory MSConv weights
        MSConvWeights_ing (torch.Tensor): inhibitory MSConv weights
        
    """
    
    def __init__(
        self, device, par, weights_name_exc, weights_name_inh, SSConvWeights , MSConvWeights_exc, MSConvWeights_inh
    ):
        super(TrainMSConv, self).__init__()

   
        #Load network parameters
        self.par = par

        #Load weights  
        self.SSConv_weights = SSConvWeights.to(device).to(torch.float32)
        self.MSConv_weights_exc = MSConvWeights_exc.to(device).to(torch.float32)
        self.MSConv_weights_inh = MSConvWeights_inh.to(device).to(torch.float32)
        self.weights_name_exc = weights_name_exc
        self.weights_name_inh = weights_name_inh

       
        #Initialize network layers 
        self.input = Input(self.par)
        self.noDyn_input = self.input.noDyn_input

        self.SSConv = SSConv(self.par)
        self.lif_SSConv = self.SSConv.lif_SSConv

        self.merge = Merge(self.par)
        self.lif_merge = self.merge.lif_merge

        self.MSConv = MSConv(self.par)
        self.lif_MSConv = self.MSConv.lif_MSConv

    
        #Set up STDP rule 
        self.STDP_params_exc = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.MSConv_w_init_exc, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP_exc = STDPFedericoFeedForwardCell(self.par.MSConv_weight_shape, (self.par.MSConv_out_dim, self.MSConv.conv_MSConv_dim[0], self.MSConv.conv_MSConv_dim[1]), p = self.STDP_params_exc, dt = self.par.dt)

        self.STDP_params_inh = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.MSConv_w_init_inh, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP_inh = STDPFedericoFeedForwardCell(self.par.MSConv_weight_shape, (self.par.MSConv_out_dim, self.MSConv.conv_MSConv_dim[0], self.MSConv.conv_MSConv_dim[1]), p = self.STDP_params_inh, dt = self.par.dt)


    def forward(self,  
        device: torch.device 
    ) -> Tuple[torch.Tensor, STDPFedericoState]:

        """ Compute forward step of SNN

        Parameters: 
            device (torch.device): device on which computations shall be performed
        """
      
        #Initialize STDP states
        s_STDP_exc = self.STDP_exc.initial_state(self.par.batch_size, self.MSConv_weights_exc, device = device, dtype = torch.float32 )
        s_STDP_inh = self.STDP_inh.initial_state(self.par.batch_size, self.MSConv_weights_inh, device = device, dtype = torch.float32 )

        #Initialize SNN states
        s_input = self.noDyn_input.initial_state(self.par.batch_size, device = device, dtype = torch.float32)
        s_SSConv = self.lif_SSConv.initial_state_WTA(self.par.batch_size, device = device, dtype = torch.float32)
        s_merge = self.lif_merge.initial_state_NT(self.par.batch_size, device = device, dtype = torch.float32)
        s_MSConv = self.lif_MSConv.initial_state_WTA(self.par.batch_size, device = device, dtype = torch.float32)
        
        #Initialize output spikes for postsynaptic tau_max update 
        z_MSConv = torch.zeros_like(s_MSConv.lifRefrac.lif.v)


        #To plot the MSConv output, optic flow vectors and colors need to be assigned to each output map. Since this is computationally expensive, we only do this during inference when the weights do not change anymore
        if self.par.training == False:
            #Assign optic flow vector to each output map 
            OF_class = compute_OF(self.par, self.MSConv_weights_exc, self.MSConv_weights_inh, device, 0.3)
            OF = OF_class.compute_OF()[0]
            
            #Assign color to each optic flow vector 
            map_colours = np.squeeze(flow_viz_np(np.expand_dims(OF[:,0],1), np.expand_dims(OF[:,1],1)))
      
            #Determine length of optical flow vectors 
            OF_length = torch.tensor((OF[:, 0]**2 + OF[:, 1]**2)**0.5).to(device)
            
            #Compute mean OF length 
            OF_mean = torch.mean(OF_length)
            
            #Extract sorted indices
            OF_sorted_idx = torch.argsort(OF_length)
            
            #Sort OF magnitudes
            OF_length_sorted = torch.round(100* OF_length[OF_sorted_idx])/100
        
        else: 
            #Make dummy data
            OF_mean = 1
            OF_length = 1
       

        
        #Initialize containers to collect results
        stiffness_ave_buffer_SSConv = []
        threshold_buffer_SSConv = []
        stiffness_ave_buffer_MSConv = []
        threshold_buffer_MSConv = []
        tau_max_mean_buffer = []
        activity_mean_buffer = []
    
    
        for sequence in range(self.par.iterations):
            
            #Randomly selecting sequence from directory 
            random_files = random.sample(os.listdir(self.par.directory), self.par.batch_size)
                       
            # Load first sequence
            data = bz2.BZ2File(self.par.directory + '/{}'.format(random_files[0]), 'rb')
            x = cPickle.load(data)

            #Determine sequence length
            seq_length = x.shape[0]
            
            #Adding remaining sequences in batch 
            for seq in range(1,len(random_files)):
                file_name = self.par.directory + '/{}'.format(random_files[seq])
                load = torch.load(file_name)
                x = torch.cat((x, load),dim = 1)
    
            #Randomly decide whether or not to flip image in x and y direction and polarity
            flip_w, flip_h, flip_p = tuple(np.random.randint(0,2,3))
            
            # #Aplly flips, (comment this part out when performing several test runs that are supposed to be compared)
            # x = (1 - flip_w) * x + flip_w * x[:,:, :,:,  torch.arange(x.shape[4] -1, -1, -1)]
            # x = (1 - flip_h) * x + flip_h * x[:, :, :, torch.arange(x.shape[3] -1, -1, -1)]
            # x = (1 - flip_p) * x + flip_p * x[:,:, torch.arange(x.shape[2] -1, -1, -1)]

            #Initializing buffers for plotting 
            input_buffer = torch.zeros(30, self.par.batch_size, self.par.input_out_dim, self.par.SSConv_m, *self.input.conv_input_dim[0:2]).to(device)
            SSConv_buffer = torch.zeros(30, self.par.batch_size, self.par.SSConv_out_dim, self.par.merge_m, *self.SSConv.conv_SSConv_dim[0:2]).to(device)
            merge_buffer = torch.zeros(30, self.par.batch_size, self.par.merge_out_dim, self.par.MSConv_m, *self.merge.conv_merge_dim[0:2]).to(device)
            MSConv_buffer = torch.zeros(30, self.par.batch_size, self.par.MSConv_out_dim, self.par.MSConv_m_after, *self.MSConv.conv_MSConv_dim[0:2]).to(device)
            
          
            if self.par.save_videos:
                # Save videos under results/videos
                directory = 'results/videos/'
                
                #Obtain name of sequence 
                run = random_files[0][:-5]
                
                #Initialize video writers
                input_video = cv2.VideoWriter( directory +'input_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1000, (self.input.conv_input_dim[1],self.input.conv_input_dim[0]))
            
                SSConv_video = cv2.VideoWriter(directory +'SSConv_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.SSConv.conv_SSConv_dim[1],self.SSConv.conv_SSConv_dim[0]) )
                vth_SSConv_video = cv2.VideoWriter(directory +'vth_SSConv_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.SSConv.conv_SSConv_dim[1], self.SSConv.conv_SSConv_dim[0]) )
                
                merge_video = cv2.VideoWriter(directory +'merge_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.merge.conv_merge_dim[1], self.merge.conv_merge_dim[0]))
                A_video = cv2.VideoWriter(directory +'A_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.merge.conv_merge_dim[1], self.merge.conv_merge_dim[0]))
                tau_max_video = cv2.VideoWriter(directory +'tau_max_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.merge.conv_merge_dim[1], self.merge.conv_merge_dim[0]))
                
                MSConv_video = cv2.VideoWriter(directory +'MSConv_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.MSConv.conv_MSConv_dim[1], self.MSConv.conv_MSConv_dim[0] ))
                vth_MSConv_video = cv2.VideoWriter(directory +'vth_MSConv_' + run +'.avi', cv2.VideoWriter_fourcc(*'MJPG'),1000, (self.MSConv.conv_MSConv_dim[1], self.MSConv.conv_MSConv_dim[0]))
        
            
            for ts in range(0, seq_length):
                
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
                
                #Plott output spikes of input layer
                plot_input = plot.plot2D_discrete(input_buffer_sum[:,:,0], self.input.conv_input_dim[0], self.input.conv_input_dim[1], 0, 'downsampled input', (1, 720), (450, 250), 1, dt = self.par.dt)          
                
                               
                
                
                ###################SSConv layer###################
                #Compute SSConv layer output spikes and new state
                z, s_SSConv, plot_vth_SSConv, boxes, box_indices = self.SSConv.forward(z, s_SSConv, self.par, self.SSConv_weights, s_input.X, device, 'SSConv', True, False)
                
                #Buffer output spikes 
                SSConv_buffer, SSConv_buffer_sum = plot.rotate_buffer(z, SSConv_buffer)

                # #Plot presynaptic traces of SSConv layer
                # plot.plot_presynaptic_traces(s_input.X, self.par.map_colours, 1, 2, boxes, box_indices, self.par.MSConv_padding3D, False,  0, 'pst_SSConv', (1200, 200), (525, 250), 1, device)

                #  Plotting output spikes of SSConv layer in one window
                plot_SSConv = plot.plot_output_spikes_together(SSConv_buffer_sum, self.par.map_colours, 0, 0, 'SSConv_output_spikes_together', (600, 685), (300,300), 1, device)

              
               
               
                ###################Merge layer###################
                #Compute Merge layer output spikes and new state
                z, s_merge, plot_sumX, plot_tau_max = self.merge.forward(z, s_merge, z_MSConv, self.par.training, ts, self.par.batch_size, device, OF_mean, OF_length)

                #Buffer output spikes 
                merge_buffer, merge_buffer_sum = plot.rotate_buffer(z, merge_buffer)

                #Plot output spikes of merge layer 
                plot_merge= plot.plot_output_spikes_together(merge_buffer_sum, [(100, 100, 100)],0, 0,  'merge_output_spikes', (0, 0), (525,250), 1, device)

               
                

                ###################MSConv layer###################
                #Compute MSConv layer output spikes and new state
                z, s_MSConv, plot_vth_MSConv, boxes, box_indices = self.MSConv.forward(z, s_MSConv, self.par, s_STDP_exc.weights, s_STDP_inh.weights, s_merge.X, device, 'MSConv',  True, self.par.training)
                
                #Save a copy of z for postsynaptic tau_max update 
                z_MSConv = z.clone()
           
                #Buffer output spikes 
                MSConv_buffer, MSConv_buffer_sum = plot.rotate_buffer(z, MSConv_buffer)
               
                # #Plot presynaptic traces of MSConv layer
                # pst = plot.plot_presynaptic_traces(s_merge.X, map_colours, 2, 5, boxes, box_indices, self.par.MSConv_padding3D, False,  0, 'pst_MSConv', (600, 100), (1100, 600), 1, device)
                
                # Plot output spikes of MSConv layer in one window. CANNOT BE PLOTTED DURING TRAINING!
                plot_MSConv = plot.plot_output_spikes_together(MSConv_buffer_sum, map_colours, 0, 0, 'MSConv_output_spikes_together', (1400, 685), (525,250), 1, device)
               
                
                 
                ###################Training###################
                if self.par.training: 
                    s_STDP_exc, X =  self.STDP_exc.forward1(s_merge.X, z, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device) , torch.tensor(self.par.MSConv_padding3D).to(device), device, s_STDP_exc)
                    s_STDP_inh, X =  self.STDP_inh.forward1(s_merge.X, z, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device) , torch.tensor(self.par.MSConv_padding3D).to(device), device, s_STDP_inh)


                #Compute total MSConv weights and sort them in increasing order of OF magnitued 
                weights = (s_STDP_exc.weights + self.par.MSConv_beta * s_STDP_inh.weights)[OF_sorted_idx]
                # weights = (s_STDP_exc.weights + self.par.MSConv_beta * s_STDP_inh.weights)
                
                # #Plot all weights in one window 
                # plot.plot_weights_MSConv_sorted(weights[OF_sorted_idx], 'MSConv_weights_sorted', (1,1),(2000, 1000), True, 0, device)
                
                # # Plot distribution of OF vectors
                # plot_OF('colour_code.png', OF[np.array(OF_sorted_idx.cpu())])
               
                #These plots show the the MSConv weights during training, since they are quite slow, we recommend only plotting the first and last delay
                # #x-tau
                # plot.plot_weights_MSConv(weights[:,:,:,:,0], 8, 8, 'MSConv_weights_x_tau', (1,462),(1050, 190), False, 1, device)
                # #y-tau
                # plot.plot_weights_MSConv(weights[:,:,:,0], 8, 8, 'MSConv_weights_y_tau', (1,462),(1050, 190), False, 1, device)
                # x-y, tau = 0
                # plot.plot_weights_MSConv(weights[:,:,0], 8, 8, 'MSConv_weights_x_y_tau0', (1,1),(600, 600), True, 1, device, col_pad = 1)
                # #x-y, tau = 1
                # plot.plot_weights_MSConv(weights[:,:,1], 8, 8, 'MSConv_weights_x_y_tau1', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 2
                # plot.plot_weights_MSConv(weights[:,:,2], 8, 8, 'MSConv_weights_x_y_tau2', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 3
                # plot.plot_weights_MSConv(weights[:,:,3], 8, 8, 'MSConv_weights_x_y_tau3', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 4
                # plot.plot_weights_MSConv(weights[:,:,4], 8, 8, 'MSConv_weights_x_y_tau4', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 5
                # plot.plot_weights_MSConv(weights[:,:,5], 8, 8, 'MSConv_weights_x_y_tau5', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 6
                # plot.plot_weights_MSConv(weights[:,:,6], 8, 8, 'MSConv_weights_x_y_tau6', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 7
                # plot.plot_weights_MSConv(weights[:,:,7], 8, 8, 'MSConv_weights_x_y_tau7', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 8
                # plot.plot_weights_MSConv(weights[:,:,8], 8, 8, 'MSConv_weights_x_y_tau8', (1,462),(600, 600), True, 1, device)
                #x-y, tau = 9
                # plot.plot_weights_MSConv(weights[:,:,9], 8, 8, 'MSConv_weights_x_y_tau9', (620,1),(600, 600), True, 1, device, col_pad = 1)

                mean_SSConv_vth = np.array(torch.mean(s_SSConv.v_th).cpu())
                mean_SSConv_S = np.array(torch.mean(s_SSConv.S).cpu())
                mean_MSConv_vth = np.array(torch.mean(s_MSConv.v_th).cpu())
                mean_MSConv_S = np.array(torch.mean(s_MSConv.S).cpu())
                mean_A = np.array(torch.mean(s_merge.A[torch.where(s_merge.A>0)]).cpu())
                mean_tau_max = np.array(torch.mean(s_merge.tau_max[torch.where(s_merge.A>0)]).cpu())
                
                #Write status update 
                if time%100 == 0:
                    if self.par.training == False:
                        print('--------------------')
                        print('Inference sequence: ', sequence+1, '/', self.par.iterations)
                        print('Time: ', time, '\n')
                        print('Current mean SSConv voltage threshold: ', mean_SSConv_vth)
                        print('Current mean SSConv synaptic stiffness: ', mean_SSConv_S, ' (target = {target})'.format(target = self.par.SSConv_S_tar))
                        print('Current mean MSConv voltage threshold: ', mean_MSConv_vth)
                        print('Current mean MSConv synaptic stiffness: ', mean_MSConv_S, ' (target = {target})'.format(target = self.par.MSConv_S_tar))
                        print('Current mean maximum synaptic delay: ', mean_tau_max, 'ms')
                        print('Current mean relative optic flow magnitude: ', mean_A, ' (target = {target})'.format(target = self.par.MSConv_A_tar))
                        print('--------------------\n')
                    else:
                        print('--------------------')
                        print('Training sequence: ', sequence+1, '/', self.par.iterations)
                        print('Time: ', time, '\n')
                        print('--------------------\n')
                    
                #Write to videos
                if self.par.save_videos:
                    input_video.write(plot_input) 
                    vth_SSConv_video.write(plot_vth_SSConv)
                    SSConv_video.write(plot_SSConv)
                    merge_video.write(plot_merge)
                    A_video.write(np.uint8(plot_sumX))
                    tau_max_video.write(np.uint8(plot_tau_max))
                    vth_MSConv_video.write(plot_vth_MSConv)
                    MSConv_video.write(plot_MSConv)
                    
                #Write result to buffers
                if self.par.save_results:
               
                    threshold_buffer_SSConv.append(mean_SSConv_vth)
                    stiffness_ave_buffer_SSConv.append(mean_SSConv_S)
                   
                    threshold_buffer_MSConv.append(mean_MSConv_vth)
                    stiffness_ave_buffer_MSConv.append(mean_MSConv_S)
                    
                    activity_mean_buffer.append(mean_A)
                    tau_max_mean_buffer.append(mean_tau_max)
                    
                
            #Release video streams     
            if self.par.save_videos:   
                input_video.release() 
                SSConv_video.release()
                vth_SSConv_video.release()
                merge_video.release()
                A_video.release()
                tau_max_video.release()
                MSConv_video.release()
                vth_MSConv_video.release()
                
                cv2.destroyAllWindows()

            #Save results to files
            if self.par.save_results:
                
                #Define directory where results are saved 
                directory = 'results/data/'
                
                #Obtain name of sequence 
                run = random_files[0][:-5]
                
                with bz2.BZ2File(directory + 'v_th_hist_SSConv_' + run + '.pbz2', 'w') as f: 
                    cPickle.dump(threshold_buffer_SSConv, f)
                    
                with bz2.BZ2File(directory + 'stiffness_hist_SSConv_' + run  + '.pbz2', 'w') as f: 
                    cPickle.dump(stiffness_ave_buffer_SSConv, f)
                    
                with bz2.BZ2File(directory + 'v_th_hist_MSConv_' + run +  '.pbz2', 'w') as f: 
                    cPickle.dump(threshold_buffer_MSConv, f)
                
                with bz2.BZ2File(directory + 'stiffness_hist_MSConv_' + run +  '.pbz2', 'w') as f: 
                    cPickle.dump(stiffness_ave_buffer_MSConv, f)
                    
                with bz2.BZ2File(directory + 'tau_max_hist_' + run + '.pbz2', 'w') as f: 
                    cPickle.dump(tau_max_mean_buffer, f)
                
                with bz2.BZ2File(directory + 'A_hist_' + run +  '.pbz2', 'w') as f: 
                    cPickle.dump(activity_mean_buffer, f)
        
    
                
            #Save weights
            torch.save(s_STDP_exc.weights, self.weights_name_exc)
            torch.save(s_STDP_exc.weights, self.weights_name_inh)

           
        

        
        return  s_STDP_exc,s_STDP_inh

