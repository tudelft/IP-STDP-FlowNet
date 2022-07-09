from typing import NamedTuple, Tuple

import numpy as np
import tools.plotting as plot
import torch
from functional.lif_mod2 import (LIFModFeedForwardStateNT,
                                 LIFModFeedForwardStateWTA)
from module.lif_mod import (LIFModFeedForwardCell, LIFModParameters,
                            LIFModParametersNeuronTrace)
from module.noDyn import (noDynFeedForwardCell, noDynFeedForwardState,
                          noDynParameters)
from norse.torch.functional.lif import LIFFeedForwardState
from norse.torch.module.lif import LIFParameters
from norse.torch.module.lif_refrac import (LIFRefracCell,
                                           LIFRefracFeedForwardCell,
                                           LIFRefracParameters)


class dim():
    """ This classs computes the output dimensions of the various layers based on the used stride, kernel and padding values.

    Parameters: 
        par (SNN_par): network parameters
      
    """

    def __init__(
        self, par
    ):
        #Network parameters
        self.par= par 

        #Input layer
        self.conv_input_dim = self.get_after_conv_dim(self.par.height, self.par.width, self.par.input_kernel3D, self.par.input_stride3D, self.par.input_padding3D, self.par.input_m)

        #SS-Conv layer
        self.conv_SSConv_dim = self.get_after_conv_dim(self.conv_input_dim[0], self.conv_input_dim[1], self.par.SSConv_kernel3D, self.par.SSConv_stride3D, self.par.SSConv_padding3D, self.par.SSConv_m)

        #Merge layer
        self.conv_merge_dim = self.get_after_conv_dim(self.conv_SSConv_dim[0], self.conv_SSConv_dim[1], self.par.merge_kernel3D, self.par.merge_stride3D, self.par.merge_padding3D, self.par.merge_m)

        #MS-Conv layer
        self.conv_MSConv_dim = self.get_after_conv_dim(self.conv_merge_dim[0], self.conv_merge_dim[1], self.par.MSConv_kernel3D, self.par.MSConv_stride3D, self.par.MSConv_padding3D, self.par.MSConv_m)


    def get_after_conv_dim(self, H, W, k, s, p, D = 1, d = (1,1,1)):
        '''Computes the dimensions of a neural layer after convolution has been applied

            Parameters: 
                H: Height of input image 
                W: Width of input image 
                k: Kernel
                s: Stride
                p: Padding 
                d: dilation 
                D: depth of input image (number of synapses between two neurons)
        '''
        
        H_out = np.floor((H + 2*p[1] - d[1] * (k[1] -1) - 1)/s[1] + 1)
        W_out = np.floor((W + 2*p[2] - d[2] * (k[2] -1) - 1)/s[2] + 1)
        D_out = np.floor((D + 2*p[0] - d[0] * (k[0] -1) - 1)/s[0] + 1)

        return int(H_out), int(W_out), int(D_out)
    

class Input():
    """ Defines the input layer of the SNN in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"

    Parameters: 
        par (SNN_par): class defining network parameters
       
    """

    def __init__(
        self, par
    ):
        #Network parameters
        self.par = par 

        #Compute output dimension of inut layer 
        self.conv_input_dim = dim(self.par).conv_input_dim
        
        #Setting up noDyn parameters 
        self.params_input = noDynParameters(alpha_mod = self.par.SSConv_alpha, lambda_x = self.par.SSConv_lambda_X, delays = self.par.SSConv_delay, m = self.par.SSConv_m)
        
        #Setting up neuron feedforward cell 
        self.noDyn_input = noDynFeedForwardCell((self.par.input_out_dim, self.conv_input_dim[0], self.conv_input_dim[1]), p = self.params_input, dt = self.par.dt)

    def forward(
        self, 
        x_ts : torch.tensor, 
        s_input : noDynFeedForwardState,
        batch_size : int,
        downsample_parameters : tuple
    )-> Tuple[torch.Tensor, noDynFeedForwardState]:

        """ Compute output of input layer  

        Parameters: 
            x_ts (torch.Tensor): input sequence at current times step
            s_input (noDynFeedforwardState): state of the input layer
            batch_size (int): number of sequences to be processed at same time
            downsample_parameters (tuple): parameters for downsampling of input data (kernel_size, stride, padding)    
        """

        #Downsampling data
        x_ts = torch.nn.MaxPool2d(*downsample_parameters)(x_ts)

        #Computing output spikes and neuron states 
        z, s_input = self.noDyn_input(batch_size, x_ts, s_input)

        return z, s_input


class SSConv():
    """Defines the SSConv layer of the SNN 

    Parameters: 
        par (SNN_par): class defining network parameters
      
    """

    def __init__(
        self, par
    ):
        #Network parameters
        self.par = par 

        #Compute output dimension of SSConv layer 
        self.conv_SSConv_dim = dim(self.par).conv_SSConv_dim
        
        #Setting up LIF parameters 
        self.params_SSConv = LIFRefracParameters(LIFParameters( tau_mem_inv = 1/self.par.SSConv_lambda_v, v_th = self.par.SSConv_v_th), rho_reset = self.par.SSConv_delta_refr)
        
        #Setting up modified LIF parameters 
        self.modParams_SSConv = LIFModParameters(lifRefrac = self.params_SSConv, delays = self.par.merge_delay, m = self.par.merge_m, lambda_vth = self.par.SSConv_lambda_vth, vth_rest = self.par.SSConv_vth_rest, vth_conv_params = self.par.SSConv_vth_conv_params, S_tar = self.par.SSConv_S_tar, n_vth = self.par.SSConv_n_vth)
        
        #Setting up neuron feedforward cell 
        self.lif_SSConv = LIFModFeedForwardCell(
            (self.par.SSConv_out_dim, self.conv_SSConv_dim[0],  self.conv_SSConv_dim[1]), p= self.modParams_SSConv, dt = self.par.dt
        )

    def forward(
        self, 
        z : torch.tensor, 
        s_SSConv : LIFModFeedForwardStateWTA,
        par,
        s_STDP_weights : torch.tensor,
        X: torch.tensor,
        device: torch.device,
        layer: str,
        v_th_adaptive:bool,
        training: bool
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:

        """ Compute output of SSConv layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_SSConv (LIFModFeedforwardStateWTA): state of the SSConv layer
            par: SNN parameters
            s_STDP_weights (torch.tensor): current weights of the SSConv layer  
            X (torch.tensor): presynaptic trace (neuron trace of previous layer)
            device (torch.devide): device on which computations are performed
            layer (str)
            v_th_adaptive (bool): specifies whether an adaptive threshold shall be used
            training (bool): indicates whether or not layer is being trained  
        """

        #Convolving input spikes with weights 
        input_tensor = torch.nn.functional.conv3d(z, s_STDP_weights.to(device), None, self.par.SSConv_stride3D, self.par.SSConv_padding3D)

        #Convolving presynaptic traces for each neuron 
        X_tensor = torch.nn.functional.conv3d(X, self.par.SSConv_weights_pst.to(device), None, self.par.SSConv_stride3D, self.par.SSConv_padding3D)
        
        #Establish number of active presynapitc traces within presynaptic trace window of each neuron 
        X_tensor_bool = torch.nn.functional.conv3d(torch.gt(X, 0.001).float(), self.par.SSConv_weights_pst.to(device), None, self.par.SSConv_stride3D, self.par.SSConv_padding3D)
        
        #Determine maximum convolved spiketrain within neighborhood of each neuron 
        X_tensorm, max_indices = torch.nn.MaxPool2d(3, 1, 1, return_indices = True)(X_tensor.squeeze(2))
  
        #Computing output spikes and neuron states: Max PSTs
        z, s_SSConv, image_output, boxes, box_indices = self.lif_SSConv.forward_WTA(par, input_tensor - X_tensorm.unsqueeze(2), v_th_adaptive, s_SSConv, torch.tensor(self.par.SSConv_kernel3D).to(device), torch.tensor(self.par.SSConv_stride3D).to(device), layer, training = training)

    
        return z, s_SSConv, image_output, boxes, box_indices



class Merge():
    """ This classs defines the Merge layer of the SNN 
    Parameters: 
        par (SNN_par): class defining network parameters
  
    """

    def __init__(
        self, par
    ):
        #Network parameters
        self.par = par 

        #Compute output dimension of SSConv layer 
        self.conv_merge_dim = dim(self.par).conv_SSConv_dim
        
        #Setting up LIF parameters 
        self.params_merge = LIFModParameters(LIFRefracParameters(LIFParameters(tau_mem_inv = 1/self.par.merge_lambda_v, v_th = self.par.merge_v_th), rho_reset = self.par.merge_delta_refr), delays =self.par.MSConv_delay, m =self.par.MSConv_m, A_tar = self.par.MSConv_A_tar, tau_max_conv_params = self.par.MSConv_tau_max_conv_params, n_tau_max = self.par.MSConv_n_tau_max, c_i_th = self.par.MSConv_c_i_th, c_j_th = self.par.MSConv_c_j_th)

        #Setting up modified LIF parameters 
        self.modParams_merge = LIFModParametersNeuronTrace(lifMod = self.params_merge,  alpha_mod = self.par.MSConv_alpha, lambda_X = self.par.MSConv_lambda_X)
        
        #Setting up neuron feedforward cell 
        self.lif_merge = LIFModFeedForwardCell(
            (self.par.merge_out_dim,self.conv_merge_dim[0],  self.conv_merge_dim[1]),
            p= self.modParams_merge, dt = self.par.dt
        )

    def forward(
        self, 
        z : torch.tensor, 
        s_merge : LIFModFeedForwardStateNT,
        z_MSConv:torch.Tensor,
        training: bool, 
        t: int, 
        batch_size : int,
        device: torch.device,
        OF_mean: torch.tensor, 
        OF_magnitudes:torch.tensor,
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:
        
        """ Compute output of merge layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_merge (LIFModFeedforwardStateNT): state of the merge layer
            z_MSConv (torch.Tensor): output spikes of the MSConv layer
            training (bool): indicates whether or not layer is being trained  
            t (int): current time step
            batch_size (int): number of sequences to be processed at same time
            device (torch.devide): device on which computations are performed  
            OF_mean (torch.tensor): mean magnitude of all ouput map optic flow vectors
            OF_magnitudes (torch.tensor): magnitudes of all output map optic flow vectors
        """

        #Convolving input spikes with weights 
        input_tensor= torch.nn.functional.conv3d(z, self.par.merge_weights.to(device), None, self.par.merge_stride3D, self.par.merge_padding3D)
        
        #Compute output spikes and neuron states
        z, s_merge, image_output, image_output2 = self.lif_merge.forward_NT(batch_size, input_tensor, s_merge, z_MSConv, training, t, self.par, OF_mean, OF_magnitudes)

        return z, s_merge, image_output, image_output2



class MSConv():
    """ This classs defines the MSConv layer of the SNN

    Parameters: 
        par (SNN_par): class defining network parameters
    """

    def __init__(
        self, par
    ):
        #Network parameters
        self.par = par 

        #Compute output dimension of SSConv layer 
        self.conv_MSConv_dim = dim(self.par).conv_MSConv_dim
        
        #Setting up LIF parameters 
        self.params_MSConv = LIFRefracParameters(LIFParameters(tau_mem_inv = 1/self.par.MSConv_lambda_v, v_th = self.par.MSConv_v_th), rho_reset = self.par.MSConv_delta_refr)
        
        #Setting up modified LIF parameters 
        self.modParams_MSConv = LIFModParameters(lifRefrac = self.params_MSConv, delays = self.par.MSConv_delays_after, m = self.par.MSConv_m_after, lambda_vth = self.par.MSConv_lambda_vth, vth_rest = self.par.MSConv_vth_rest, vth_conv_params = self.par.MSConv_vth_conv_params, tau_max_conv_params = self.par.MSConv_tau_max_conv_params, S_tar = self.par.MSConv_S_tar, n_vth = self.par.MSConv_n_vth, n_tau_max = self.par.MSConv_n_tau_max, c_i_th = self.par.MSConv_c_i_th, c_j_th = self.par.MSConv_c_j_th, A_tar = self.par.MSConv_A_tar)
        
        #Setting up neuron feedforward cell 
        self.lif_MSConv = LIFModFeedForwardCell(
            (self.par.MSConv_out_dim,self.conv_MSConv_dim[0],  self.conv_MSConv_dim[1]),
            p=self.modParams_MSConv, dt = self.par.dt
        )

    def forward(
        self, 
        z : torch.tensor, 
        s_MSConv : LIFModFeedForwardStateWTA, 
        par,
        s_STDP_weights_exc : torch.tensor,
        s_STDP_weights_inh : torch.tensor,
        X: torch.tensor,
        device: torch.device,
        layer:str,
        v_th_adaptive:bool,
        training: bool,  
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:

        """ Compute output of SSConv layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_MSConv (LIFModFeedforwardStateWTA): state of the SSConv layer
            par: SNN parameters
            s_STDP_weights_exc (torch.tensor): current excitatory weights of the SSConv layer  
            s_STDP_weights_inh (torch.tensor): current excitatory weights of the SSConv layer  
            X (torch.tensor): presynaptic trace (neuron trace of previous layer)
            device (torch.devide): device on which computations are performed
            layer (str): string specyfying which layer is considered for plotting
            v_th_adaptive (bool): specifies whether an adaptive threshold shall be used
            training (bool): indicates whether or not layer is being trained  
        """

        #Convolving input spikes with weights for excitatory and inhibitory weigts
        input_tensor_ex = torch.nn.functional.conv3d(z, s_STDP_weights_exc.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        input_tensor_inh = torch.nn.functional.conv3d(z, s_STDP_weights_inh.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        
        #Combining exitatory and inhibitory weights 
        input_tensor = input_tensor_ex + self.par.MSConv_beta*input_tensor_inh
        
        #Convolving presynaptic traces for each neuron 
        X_tensor = torch.nn.functional.conv3d(X, self.par.MSConv_weights_pst.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)

        X_tensorm, max_indices = torch.nn.MaxPool2d(3, 1, 1, return_indices = True)(X_tensor.squeeze(2))
        
        #Computing output spikes and neuron states
        z, s_MSConv, image_output, boxes, box_indices = self.lif_MSConv.forward_WTA(par, input_tensor - X_tensorm.unsqueeze(2), v_th_adaptive, s_MSConv, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device), layer, training = training)


        return z, s_MSConv, image_output, boxes, box_indices




