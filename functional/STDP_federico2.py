

import torch
import cv2
import time 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt

from typing import NamedTuple, Tuple

import tools.plotting as plot


class STDPFedericoParameters(NamedTuple):
    '''Parameters of Federicos STDP learning rule.

    Parameters:
        n (torch.Tensor): Learning rate 
        w_init (torch.Tensor): Value of inital  weights
        a (torch.Tensor): Parameter determning spread of weights 
        L (torch.Tensor): Convergence threshold
        ma_len(torch.Tensor): Length of moving average window for stopping criterion
    '''

    n: torch.Tensor = torch.as_tensor(10**(-4))
    w_init: torch.Tensor = torch.as_tensor(0.5)
    a: torch.Tensor = torch.as_tensor(0)
    L: torch.Tensor = torch.as_tensor(5*10**(-2))
    ma_len: torch.Tensor = torch.as_tensor(10)


class STDPFedericoState(NamedTuple):
    """State Federicos STDP learning rule.

    Parameters:
        weights (torch.Tensor): Weighs of the neural layers
        Li_buffer (torch.Tensor): Buffer to compute moving average of stopping criterion
        nu_spikes (torch.Tensor): Counter to determine how often a neuron has spiked
    """

    weights: torch.Tensor
    Li_buffer: torch.Tensor
    nu_spikes: torch.Tensor 


def stdp_federico_step2(
    x: torch.Tensor,
    z: torch.Tensor,
    kernel: torch.Tensor,
    stride: torch.Tensor,
    pad: torch.Tensor, 
    device: torch.device, 
    state: STDPFedericoState,
    shape: tuple,
    p: STDPFedericoParameters = STDPFedericoParameters(),
    dt: float = 10**(-3),
    
) -> Tuple[torch.Tensor, STDPFedericoState]:
    """Federicos STDP learning rule.

    Parameters:
        x(torch.Tensor): presynaptic spike traces
        z(torch.Tensor): postsynaptic spikes
        kernel(int): kernel size used in layer to be trained
        stride(int): stride used in layer to be trained 
        pad(int): padding used in layer to be trained 
        device (torch.device): device on which computations shall be performed
        state (STDPFedericoState): state of the STDP 
        shape (tuple): shape of the weights of the current layer
        p (STDPFedericoParameters): STDP parameters
        dt (float): integration time step
    """

    ##Determine start time to check how long computations take
    #start_time = time.time()
    
    #Initializing states 
    W_new = state.weights
    W_mean = state.weights
    Li_cur = torch.tensor([0.])
    Li_buffer_new = state.Li_buffer
    nu_spikes_new = state.nu_spikes
    X = torch.zeros(1, x.shape[1], x.shape[2], kernel[1], kernel[2]).to(device)
    x_unfolded_max = x

    #Determine indices of spikes 
    spike_indices = torch.where(z == 1)
   

    if len(spike_indices[0])>0:
        #Apply padding to x 
        x = torch.nn.functional.pad(x, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]))

        #Unfold tensor containing spike history of previous layer to obtain presynaptic spike train of all neurons
        #Result is of dimension [shape of MSConv output [N, channels (1 since all channels have the same pst), number of delays (1 since they disappear in the convolution), h, w], dimensions of PST ([ch, m, k, k])]
        x_unfolded = x.unfold(1, x.shape[1], 1).unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1]).unfold(4, kernel[2], stride[2])

        #Expanding unfolded x-tensor to number of output channels in current layer (presynaptic traces are the same for all channels)
        x_unfolded = x_unfolded.expand(-1, shape[0], -1, -1, -1, -1, -1, -1, -1)

        #Extracting presynaptic spike traces of all spiking neurons 
        X = x_unfolded[spike_indices]

        #Normalizing pressyaptic spike train
        Xmax = torch.amax(X, dim = (1,2,3,4), keepdim=True)
        Xmin = torch.amin(X, dim = (1,2,3,4), keepdim=True)
        X = (X - Xmin)/(Xmax - Xmin)
        

        # #Plotting presynaptic trace windows
        # plot.plot_pst_windwos(X, spike_indices, 'pst_windows', (1300, 1),(900, 1000), 0, device)

        #Extracting current weights of all spiking neurons
        W = state.weights[spike_indices[1]]

        #Computing weight update
        LTP_X = torch.exp(X) - p.a
        LTP_W = torch.exp(-(W - p.w_init))
        LTP = LTP_W*LTP_X

        LTD_X = torch.exp(1 - X) - p.a
        LTD_W = -torch.exp(W - p.w_init)
        LTD = LTD_W*LTD_X

        #Distributing weights over the corresponding sequences and channels 
        dW = torch.zeros(len(spike_indices[0]), z.shape[0],*state.weights.shape).to(device)
        dW[tuple(torch.arange(dW.shape[0])), spike_indices[0], spike_indices[1]] = p.n*(LTP + LTD) 

        #Taking channel-wise average of weights 
        dW = torch.mean(dW, dim = 0)

        #Summing up contributions of different sequences 
        dW = torch.sum(dW, dim = 0)

        # #Plotting dW 
        # plot.plot_weights_SSConv(dW, 'dW SSConv', (1000,600), (400, 400), True, 1, device)

        #Computing update
        W_new = state.weights + dW

        #Computing weights in range from zero to one
        Wmin = torch.amin(W_new, dim = (1,2,3,4)).view(state.weights.shape[0], 1, 1, 1, 1).expand(-1, *state.weights.shape[1:])
        Wmax = torch.amax(W_new, dim = (1,2,3,4)).view(state.weights.shape[0], 1, 1, 1, 1).expand(-1, *state.weights.shape[1:])
        W_mean = (W_new - Wmin)/(Wmax - Wmin)
        
        #Set up current Li equal to moving average of previous Li 
        Li_cur = torch.mean(Li_buffer_new, dim = 0).clone().unsqueeze(0).expand(len(spike_indices[0]), -1, -1)
        #Assign new Li values to corresponding maps
        Li_cur[tuple(torch.arange(len(spike_indices[0]))), spike_indices[0], spike_indices[1]] = 1/(shape[1]*shape[2]*shape[3]**2)*torch.sum((X-W_mean[spike_indices[1]])**2, dim = (1, 2,3,4))
        #Average Li-values per map 
        Li_cur = torch.mean(Li_cur, dim = 0)


        #Shifting entries in buffer 
        Li_buffer_old = Li_buffer_new[:-1].clone()
        Li_buffer_new[1:] = Li_buffer_old
        Li_buffer_new[0] = Li_cur

        #Updating number of spikes per neuron 
        nu_spikes_new[tuple(spike_indices[0]), tuple(spike_indices[1]), tuple(spike_indices[3]), tuple(spike_indices[4])] += 1


    #print ("Learning took ", time.time() - start_time, "to run")

    return STDPFedericoState(W_new, Li_buffer_new, nu_spikes_new), X







