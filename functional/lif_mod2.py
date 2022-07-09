

from typing import NamedTuple, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tools.plotting as plot
import torch
import torchvision
from norse.torch.functional.lif import (LIFFeedForwardState,
                                        lif_feed_forward_step, lif_step)
from norse.torch.functional.lif_refrac import (LIFRefracFeedForwardState,
                                               LIFRefracParameters,
                                               LIFRefracState,
                                               compute_refractory_update,
                                               lif_feed_forward_step, lif_step)
from norse.torch.functional.threshold import threshold


class LIFModState(NamedTuple):
    """State of the LIFMod neuron with multisynaptic connections and delays applied to the output. The LIFMod neuron builds on top of the LIFRefreac neuron but in addition, allows for multisynaptic conenctions between two neurons with varying delays. 

    Parameters:
        lifRefrac (LIFRefracState): state of the LIFRefrac neuron integration
        buffer (torch.Tensor): buffer containing time delays of output
        v_th (torch.Tensor): current voltage threshold for each neuron 
        S (torch.Tensor): current synaptic stiffness at each neuron
    """

    lifRefrac: LIFRefracState
    buffer: torch.Tensor
    v_th: torch.Tensor
    S: torch.Tensor
  
  


class LIFModStateNeuronTrace(NamedTuple):
    """State of a modified LIF neuron with neuron trace.

    Parameters:
        lifMod (LIFModState): state of the LIFMod neuron integration
        X (torch.Tensor): tensor containing neuron trace of output spikes 
        tau_max (torch.Tensor): current maximum synaptic delay at each neuron
        delays (torch.Tensor): delays at each neuron 
        A (torch.Tensor): current relative optic flow magnitude at each neuron

    """

    lifMod:  LIFModState
    X: torch.Tensor
    tau_max: torch.Tensor
    delays: torch.Tensor
    A: torch.Tensor


class LIFModParameters(NamedTuple):
    """Parameters of the LIFMod neuron.

    Parameters:
        lifRefrac (LIFRefracParameters): parameters of the LIFRefrac neuron integration
        delays (torch.Tensor): delays applied to the output of the neuron, one delay per synaptic connection [ms]
        m (torch.Tensor): number of synapses per neuron 

        S_tar (torch.Tensor): target for the stiffness value during inference
        vth_conv_parmas (torch.Tensor): convolutional parameters defining window size for voltage threshold update
        n_vth (torch.tensor): learning rate of adaptive voltage threshold
        lambda_vth (torch.Tensor): time constant for the adaptive voltage threshold
        vth_rest (torch.Tensor): resting voltage threshold
       
        A_tar (torch.Tensor): Target value for estimated relative optic flow magntiude 
        tau_max_conv_parmas (torch.Tensor): convolutional parameters defining window size for max tau_max update
        n_tau_max (torch.tensor): gain for adaptive tau_max 
        c_i_th (torch.Tensor): activity threshold of individual conenctions within presynaptic trace
        c_j_th (torch.Tensor): activity treshold for entire presynaptic trace of postsynaptic neuron
        
    """

    # Modfied LIF neuron paramters
    lifRefrac: LIFRefracParameters = LIFRefracParameters()
    delays: torch.Tensor = torch.Tensor([1]).long()
    m: torch.Tensor = torch.as_tensor(1)

    # Parameters for adaptive threshold
    S_tar: torch.Tensor = torch.as_tensor(2),
    vth_conv_params: torch.Tensor = torch.Tensor([(1, 20, 27 ),(1, 1, 1),(0,0, 0)])
    n_vth: torch.Tensor = torch.as_tensor(0.05),
    lambda_vth: torch.Tensor = torch.as_tensor(0.005)
    vth_rest: torch.Tensor = torch.as_tensor(0.01)

    # Paramter for adaptive maximum delay
    A_tar: torch.Tensor = torch.as_tensor(7.8)
    tau_max_conv_params: torch.Tensor = torch.Tensor(((1, 43, 58),(1, 1, 1),(0,0,0)))
    n_tau_max:  torch.Tensor = torch.as_tensor(-0.5)
    c_i_th: torch.Tensor = torch.as_tensor(0)
    c_j_th: torch.Tensor = torch.as_tensor(0.02)
   

class LIFModParametersNeuronTrace(NamedTuple):
    """Parameters of a modified LIF neuron with neuron trace.

    Parameters:
        lifMod (LIFModParameters): parameters of the LIFMod neuron integration
        alpha_mod (torch.Tensor): scaling factor of neuron trace  
        lambda_x (torch.Tensor) : time constant of decay in neuron trace
    """

    lifMod: LIFModParameters = LIFModParameters()
    alpha_mod: torch.Tensor = torch.as_tensor(0.25)
    lambda_X: torch.Tensor = torch.as_tensor(5)


def linspace(
        start: torch.Tensor,
        stop: torch.Tensor,
        num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.

    Parameters: 
        start (torch.Tensor): start-values of linspaces
        stop (torch.Tensor): stop-values of linspaces
        num (int): number of steps 
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32,
                         device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]

    return out


def compute_presynaptic_trace_update(
    z_new: torch.Tensor,
    state: LIFModStateNeuronTrace,
    p: LIFModParametersNeuronTrace = LIFModParametersNeuronTrace(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor]:
    """Compute the neuron trace update according to the specified ODE: 
     .. math::
        \begin{align*}
            \dot{X} &= 1/\lambda_{\text{x}} (-X + \alpha z_new)\\
        \end{align*}

    Parameters
        z_new (torch.Tensor): output spikes of neural layer.
        state (LIFModStateNeuronTrace): initial state of the modified LIF neuron with neuron trace.
        p (torch.Tensor): paramters of the modified LIF neuron with neuron trace.
        dt (float): simulation time step 
    """

    # Compute presynaptic trace update
    dX = dt*1/p.lambda_X * (-state.X + p.alpha_mod*z_new)
    X = state.X + dX

    return X


def compute_wta(
    z: torch.Tensor,
    rho_new: torch.Tensor,
    v_decayed: torch.Tensor,
    v_new: torch.Tensor,
    p: LIFModParameters,
    k: tuple,
    s: tuple,
    training
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform WTA on output spikes.

    Parameters
        z (torch.Tensor): output spikes
        rho_new (torch.Tensor): refractory state
        v_decayed (torch.Tensor): voltages of neurons before they spikes
        v_new (torch.Tensor): voltages of neurons after they spiked
        p (LIFModParameters): parameters of the modified LIF neurons
        k (tuple): convolutional kernel size of the map for which WTA is applied
        s (tuple): convolutional stride of the map for which WTA is applied
        training (bool): flag indicating whether map is being trained
    """

    #start_time = time.time()

    # Extracting spike indices
    spike_indices = torch.vstack(torch.where(z == 1))

    # # Shuffling indices to prevent one layer from getting all spikes during first iteration
    # spike_indices = spike_indices[:, torch.randperm(spike_indices.size()[1])]

    # Defining boxes describing presynaptic traces
    boxlpst = s[1:]*torch.transpose(spike_indices[(3, 2), :], 0, 1).float()
    boxupst = (s[1:]*torch.transpose(spike_indices[(3, 2), :], 0, 1) + k[1:]).float()

    # Defining boxes describing direct neural neighbourhood of each neuron
    boxl = (torch.transpose(spike_indices[(3, 2), :], 0, 1) - 0).float()
    boxu = (torch.transpose(spike_indices[(3, 2), :], 0, 1) + 1).float()

    # Creating tensor specifying to which batch each box belongs
    idxs_batch = spike_indices[0, :]

    # Creating tensor specifying to which map each box belongs
    idxs_map = spike_indices[1, :]

    # Performing WTA across all maps during training and only within the maps otherwise

    # only differentiate between different batches (= competition across maps)
    idxs_batch = idxs_batch

    # differentiate between batches and different maps (= no competition across maps)
    idxs_maps = idxs_batch * z.shape[1] + idxs_map

    scores = v_decayed[tuple(spike_indices)]

    # Performing WTA with competition across maps
    winds_batch = torchvision.ops.batched_nms(
        torch.cat((boxl, boxu), 1), scores, idxs_batch, 0)

    # Performing WTA with no competition across maps
    winds_maps = torchvision.ops.batched_nms(
        torch.cat((boxl, boxu), 1), scores, idxs_maps, 0)

    # Determine spike indices after duplicates within windows have been removed
    spike_indices_maps = spike_indices[:, tuple(winds_maps)]
    spike_indices_batch = spike_indices[:, tuple(winds_batch)]

    # Create outspikes for WTA with only maps and only batch
    z_new_maps = torch.zeros_like(z)
    z_new_batch = torch.zeros_like(z)
    z_new_maps[tuple(spike_indices_maps)] = 1
    z_new_batch[tuple(spike_indices_batch)] = 1

    if training == False:
        winds = winds_maps
        z_new = z_new_maps
    else:
        winds = winds_batch
        z_new = z_new_batch
        
    # Determine winning boxes
    boxes = np.array(torch.cat((boxlpst, boxupst), 1)[tuple(winds), :].cpu())
    box_indices = np.array(spike_indices[:, winds].cpu())



    # Inhibit neurons in neighborhood of spiking neuron
    # dis = torch.floor((k - 1)/s).long()
    # dis = torch.tensor([0, 1, 1]).long()
    dis = torch.tensor([0,0,0]).long()

    for i in range(len(winds)):
        rho_new[spike_indices[0, winds[i]], :, spike_indices[2, winds[i]] - dis[1]: spike_indices[2, winds[i]] +
                dis[1] + 1, spike_indices[3, winds[i]] - dis[2]: spike_indices[3, winds[i]] + dis[2] + 1] = p.lifRefrac.rho_reset
        v_new[spike_indices[0, winds[i]], :, spike_indices[2, winds[i]] - dis[1]: spike_indices[2, winds[i]
                                                                                                ] + dis[1] + 1, spike_indices[3, winds[i]] - dis[2]: spike_indices[3, winds[i]] + dis[2] + 1] = 0

    #print ("WTA2 took ", time.time() - start_time, "to run")
    
    return z_new, z_new_batch, z_new_maps, boxes, box_indices


def delays(
    state: LIFModState,
    p: LIFModParameters,
    delays: torch.tensor,
    z_new: torch.tensor,
    dt: float
) -> Tuple[torch.Tensor]:
    """This function adds the latest spikes (z_new) to the buffer containing the delayed versions of the moodified LIF neuron spikes and removes the oldest ones.

    Parameters:
        state (LIFModState): state of the modified LIF neuron
        p (LIFModParameters): parameters of the modifed LIF neuron
        delays (torch.tensor): delays at each neuron
        z_new (torch.Tensor): new output spikes 
        dt (float): simulation time step
    """

    # Moving all entries by one index
    state.buffer.rotate(1)

    # Putting latest spikes into buffer
    state.buffer[0] = z_new

    # Computing factor of timestep [specified by dt] and delay[ms]
    factor = int(10**(-3)/dt)

    test = tuple(delays)

    # Set up indices
    idx = torch.tensor(np.indices(delays.shape))
    
    # Set indices of delays to computed indices
    idx[0] = factor*delays

    test = tuple(idx)

    z_new = torch.stack(tuple(state.buffer))[test]

    # Reshaping the tensor to the suitable format for 3D convolution (Batch size, number of channels, number of synapses, height, width)
    z_new = z_new.permute(1, 2, 0, 3, 4)

    return z_new



def compute_tau_max(
    A: torch.tensor, 
    tau_max_cur: torch.tensor,
    active: torch.tensor, 
    A_target: float,  
    n_tau_max: float, 
) -> Tuple[torch.Tensor]:
    """Compute tau_max update 

    Parameters:
        A (torch.tensor): relative optic flow magnitude at each neuron 
        tau_max_cur (torch.tensor): current value of tau_max at each neuron 
        active (torch.tensor): tensor specifying at which neuron location updates shall be performed
        A_target (float): target value of relative optic flow magnitude
        n_tau_max (float): learning rate of tau_max update 
    """
    
    #Compute difference between target and actual value of A
    error = A_target - A
   
    #Compute change in tau_max 
    d_tau_max = active*error*n_tau_max
    
    #Update tau_max, make sure all values are positive
    tau_max = torch.clamp(tau_max_cur + d_tau_max, 0)

    return tau_max 
    
    

def compute_tau_max_update_post_spikes(
    z: torch.tensor,
    state: torch.tensor,
    OF_mean: torch.tensor,
    OF_magnitudes: torch.tensor,
    p: LIFModParameters,
) -> Tuple[torch.Tensor]:
    """Compute tau_max update with postsynaptic rule considering only identity of spiking maps and now how often they spike

    Parameters:
        z (torch.Tensor): output spikes
        OF_mean (torch.Tensor): mean value of optic flow of output maps 
        OF_magnitudes (torch.Tensor): optic flow magnitudes of output maps 
        p (LIFModParameters): neuron parameters 
    """
    
    tau_max = state.tau_max
    magnitude_means = state.A
    
    if len(torch.nonzero(z) > 0):
        
        #Compute sum of spikes within window of interest in every map
        spikes_per_map = torch.nn.AvgPool3d(*p.tau_max_conv_params, divisor_override = 1)(z)
       
        #Check in which windows and maps spikes have occured 
        spiking_maps = torch.gt(spikes_per_map, 0)
        
        #Check how many maps have spiked 
        num_spikes = torch.sum(spiking_maps, dim = 1)
        
        #Check at which pixel locations spikes have occured 
        active = torch.gt(num_spikes ,0)
        
        #Change shape of OF magnitudes to match spiking maps 
        OF_magnitudes = OF_magnitudes.view(1, spiking_maps.shape[1], 1, 1,1).expand(spiking_maps.shape)
        
        #Multiply spiking maps by OF magnitudes
        spiking_magnitudes = spiking_maps*OF_magnitudes
        
        #Compute mean OF length of active maps 
        magnitude_means = torch.sum(spiking_magnitudes, dim = 1)/num_spikes
        
        #Upsample to match original dimensions 
        magnitude_means = torch.nn.Upsample(size = state.tau_max.shape[2:], mode = 'nearest')(magnitude_means)
        active = torch.gt(magnitude_means, 0)
       
        #Compute new tau_max
        tau_max = compute_tau_max(magnitude_means, state.tau_max, active, OF_mean, p.n_tau_max)
        
    #Plot tau_max and average optic flow magnitude of spiking neurons
    image_output = plot.plot_voltage_trace(magnitude_means, 0, 0, 'sum_nhb_1', (300, 300), (300, 300), 1)
    image_output2 = plot.plot_voltage_trace(tau_max, 0, 0, 'Tau max',(800, 100), (300, 300), 1)
        
    return tau_max, magnitude_means, image_output, image_output2



def compute_tau_max_update_max(
    state,
    A_tar: torch.tensor,
    p: LIFModParameters,
    z: torch.tensor,
    postsynaptic: bool
) -> Tuple[torch.Tensor]:
    """Compute tau_max update with presynaptic method considering average number of active delays 

    Parameters:
        state (torch.Tensor): state of the SNN layer
        A_tar (torch.Tensor): target value for number of active delays 
        p (LIFModParameters): neuron parameters 
        z (torch.Tensor): MSCOnv output spikes
        postsynaptic (bool): specifies if postsynaptic method should be used
        
    """
    
    #Normalize X to reduce influence of pst-parameters
    X = (state.X - torch.min(state.X))/(torch.max(state.X) - torch.min(state.X))
    
    #If only presynaptic trace of spiking neurons is considered, convolve presynaptic trace such that its dimensions match z 
    if postsynaptic:
        stride = p.MSConv_stride
        padding = p.MSConv_padding
    else:
        stride = 1
        padding = int(p.MSConv_kernel/2)
 
    #Compute sum of presynaptic trace within neighbourhood of each pixel to look at features rather than individual neurons
    sum_nbh0 = torch.nn.AvgPool3d((1, p.MSConv_kernel, p.MSConv_kernel), (1, stride, stride), (0, padding, padding), count_include_pad = False)(torch.gt(X, p.MSConv_c_j_th).double())
  
    #Check where features are present (more than one active neuron within window)
    X_nonzero = torch.gt(sum_nbh0, p.MSConv_c_i_th)
    
    #Compute number of delays with non-zero pst
    sum_X_nonzero = torch.sum(X_nonzero, dim = 2).float()
    
    
    #Only consider presynaptic trace of spiking neurons if postsynaptic rule is applied
    if postsynaptic: 
        
        #Compute indices of non-active neurons and flatten output map spikes to fit with presynaptic trace
        idx_non_active = torch.where(torch.sum(z, dim = 1) == 0)
        
        #Only consider presynapitc traces of spiking neurons 
        sum_X_nonzero[idx_non_active] = 0
            

    #Check which synapses are active
    active = torch.gt(sum_X_nonzero, 0)
    
    #Compute maximum pst within neighbourhood
    sum_nbh_1= torch.nn.MaxPool3d((1,10,10), 1, 0)(sum_X_nonzero)
    sum_nbh_1 = torch.nn.Upsample(size = sum_X_nonzero.shape[2:], mode = 'nearest')(sum_nbh_1)
    
    #Compute average of active maxima 
    sum_nbh = torch.nn.AvgPool3d(*p.MSConv_tau_max_conv_params)((sum_nbh_1*active).float())
    
    sum_active = torch.nn.AvgPool3d(*p.MSConv_tau_max_conv_params)((active).float())
    sum_active_bool = torch.gt(sum_active, 0)
    
    #Check where number of active delays is larger than zero 
    idx_active = torch.where(sum_active>0)
    
    # Initialize all activity level of all neurons as desired value such that delay is not changed for inactive synapses
    activity = torch.zeros_like(sum_nbh)
    
    #Compute average maximum number of active delays at active neurons
    activity[idx_active] = (sum_nbh/sum_active)[idx_active]
    # activity[: ,:, :, :] = torch.mean((sum_nbh/sum_active)[idx_active])
    
 
    #Upsample to match original dimensions 
    activity = torch.nn.Upsample(size = X.shape[3:], mode = 'nearest')(activity)
    sum_active_bool = torch.gt(activity, 0)
    
    
    #Compute tau_max_update
    tau_max = compute_tau_max(activity, state.tau_max, sum_active_bool, A_tar, p.MSConv_n_tau_max)
   
    # Plot tau_max and maximum number of active delays 
    image_output = plot.plot_voltage_trace(sum_nbh_1*active, 0, 0, 'sum_nhb_1', (300, 300), (300, 300), 1)
    image_output2 = plot.plot_voltage_trace(tau_max, 0, 0, 'Tau max',(800, 100), (300, 300), 1)
  
    return tau_max, activity, image_output, image_output2
  

def compute_tau_max_update_pre_ave(
    X: torch.tensor,
    A_tar: torch.tensor,
    p: LIFModParameters, 
    s: LIFModState
) -> Tuple[torch.Tensor]:
    """Compute tau_max update with presynaptic method considering average number of active delays 

    Parameters:
        X (torch.Tensor): presynaptic trace 
        A_tar (torch.Tensor): target value for number of active delays 
        p (LIFModParameters): neuron parameters 
        s (LIFModState): state of the layer
    """
    # 
    #Normalize X to reduce influence of pst-parameters
    X = (X - torch.min(X))/(torch.max(X) - torch.min(X))

    #Compute sum of presynaptic trace within neighbourhood of each pixel to look at features rather than individual neurons
    sum_nbh0 = torch.nn.AvgPool3d((1,5,5), 1,(0,2,2), count_include_pad = False)(torch.gt(X, p.c_j_th).double())

    #Check where features are present (more than one active neuron within window)
    X_nonzero = torch.gt(sum_nbh0, p.c_i_th)
  
    #Compute number of delays with non-zero pst
    sum_X_nonzero = torch.sum(X_nonzero, dim = 2).float()
    
    #Check which synapses are active
    active = torch.gt(sum_X_nonzero, 0)
    
    #Compute average pst within neighbourhood
    sum_nbh= torch.nn.AvgPool3d(*p.tau_max_conv_params)(sum_X_nonzero)
    # sum_nbh_1 = torch.nn.Upsample(size = sum_X_nonzero.shape[2:], mode = 'nearest')(sum_nbh_1)
   
    #Compute average number of active synapses
    sum_active = torch.nn.AvgPool3d(*p.tau_max_conv_params)(active.float())
    # sum_active = torch.nn.Upsample(size = sum_X_nonzero.shape[2:], mode = 'nearest')(sum_active)
   
    #Compute where number of active delays is larger zero 
    sum_active_bool = torch.gt(sum_active, 0)

    #Initialize activity levels neurons 
    activity = torch.zeros_like(sum_active).float()
   
    #Check where number of active delays is larger than zero 
    idx_active = torch.where(sum_active>0)
    
    #Compute average maximum number of active delays at active neurons
    activity[idx_active] = (sum_nbh/sum_active)[idx_active]
    # activity[: ,:, :, :] = torch.mean((sum_nbh/sum_active)[idx_active])
    
    #Upsample to match original dimensions 
    activity = torch.nn.Upsample(size = sum_X_nonzero.shape[2:], mode = 'nearest')(activity)
    sum_active_bool = torch.nn.Upsample(size = sum_X_nonzero.shape[2:], mode = 'nearest')(sum_active_bool.float())
    
    #Compute tau_max_update
    tau_max = compute_tau_max(activity, s.tau_max, sum_active_bool, A_tar, p.n_tau_max)
    
    # Plot tau_max and number of active delays in the folded presynaptic trace
    image_output =  plot.plot_voltage_trace(sum_X_nonzero, 0, 0, 'sum_X_nonzero', (300, 300), (300, 300), 1)
    image_output2 = plot.plot_voltage_trace(tau_max, 0, 0, 'Tau max',(800, 100), (300, 300), 1)

    return tau_max, activity, image_output, image_output2
  

def compute_vth_update(
    z_new_maps: torch.tensor,
    state: LIFModState,
    p: LIFModParameters,
    S_tar: torch.tensor,
    layer: str
) -> Tuple[torch.Tensor]:
    """Compute voltage threshold update

    Parameters:
        z_new_maps (torch.Tensor): output spikes before WTA across maps 
        state (LIFModState): state of the modified LIF neuron
        p (LIFModParameters): parameters of the modifed LIF neuron 
        S_tar (float): target stiffness
        layer (str): string specifying which layer is considered for plotting 
    """

    #Compute number of maps in which spikes occur for every pixel location
    num_maps = torch.sum(z_new_maps, dim = 1, keepdim = True)

    #Determine neuron locations at which spikes occur in any map
    active = torch.gt(num_maps, 0)
     
    #Compute average number of active maps within neighbourhood
    sum_nbh = torch.nn.AvgPool3d( p.vth_conv_params[0], p.vth_conv_params[1], p.vth_conv_params[2])(num_maps)
    
    #Compute how many neurons have at least one map active on average within neighbourhood
    sum_active = torch.nn.AvgPool3d(p.vth_conv_params[0], p.vth_conv_params[1], p.vth_conv_params[2])(active.float())
    
    #Compute for each neighbourhood if there is at least one neuron at which spikes occur in at least one map
    sum_active_bool = torch.gt(sum_active, 0)
    
    # #Initialize activity level of all neurons
    activity = torch.zeros_like(sum_active_bool).float()
   
    #Compute average number of spiking maps only for spiking neurons
    activity[sum_active_bool] = (sum_nbh/sum_active)[sum_active_bool]
   
    #Upsample to match original dimensions 
    activity = torch.nn.Upsample(size = z_new_maps.shape[2:], mode = 'nearest')(activity)
    sum_active_bool = torch.nn.Upsample(size = z_new_maps.shape[2:], mode = 'nearest')(sum_active_bool.float())
    
    # Compute error between overlap goal and actual overlap
    dS = (S_tar - activity)

  
    # Compute voltage update
    d_v_th = p.n_vth*(-p.lambda_vth*dS*state.v_th*sum_active_bool + (1- p.lambda_vth) * (p.vth_rest- state.v_th)*(1 - sum_active_bool.float()))
    v_th = state.v_th + d_v_th
    
  
    #Plot v_th
    image_output = plot.plot_voltage_trace(v_th, 0, 0, 'voltage threshold ' + layer,(1400, 100), (300, 300), 1)
   

    return v_th, activity, image_output


class LIFModFeedForwardStateNT(NamedTuple):
    """State of a modified feed forward LIF neuron with absolute refractory period.

    Parameters:
        lifRefrac (LIFRefracFeedForwardState): state of the feed forward LIFRefrac neuron integration
        X (torch.Tensor): presynaptic trace state 
        buffer (torch.Tensor): buffer containing delayed spiked
        tau_max (torch.Tensor): current maximum synaptic delay at each neuron
        delays (torch.Tensor): delays at each neuron 
        A (torch.Tensor): current relative optic flow magnitude at each neuron
        
    """

    lifRefrac: LIFRefracFeedForwardState
    X: torch.Tensor
    buffer: torch.Tensor
    tau_max: torch.Tensor
    delays: torch.Tensor
    A: torch.Tensor


def lif_mod_feed_forward_step_NT(
    batch_size: torch.Tensor,
    i_new: torch.Tensor,
    state: LIFModFeedForwardStateNT,
    z_MSConv:torch.tensor,
    training:bool,
    shape: tuple,
    p: LIFModParametersNeuronTrace,
    t:float,
    par, 
    OF_mean: torch.Tensor,
    OF_magnitudes: torch.Tensor,
    dt: float = 10**(-3),
) -> Tuple[torch.Tensor, torch.Tensor, LIFModFeedForwardStateNT]:
    """
    Computes a single euler-integration step of a modified feed forward
     LIF neuron-model with a refractory period. It takes as input the input current as generated by an arbitrary torch
    module or function and the sum of all the presynaptic traces for each neuron. More specifically it implements one integration
    step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}} - X
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes and X the sum of the neuron's presynaptic traces.

    Parameters:
        batch_size (torch.Tensor): batch size 
        i_new (torch.Tensor): input current 
        state (LIFModFeedForwardState): state at the current time step
        z_MSConv (torch.tensor): MSConv output spikes
        training (bool): specifies whether training is performed on current layer
        shape (tuple): shape of neuron layer (out_dim, height, width)
        p (LIFModParametersNeuronTrace): parameters of the lif neuron
        t (float): current simulation time step 
        par: SNN parameters
        OF_mean (torch.Tensor): mean value of optic flow of output maps 
        OF_magnitudes (torch.Tensor): optic flow magnitudes of output maps
        dt (float): Integration timestep to use
    """

    # compute voltage updatees
    #dv = dt * p.lifMod.lifRefrac.lif.tau_mem_inv * ((p.lifMod.lifRefrac.lif.v_leak - state.lifMod.lifRefrac.lif.v) + state.lifMod.lifRefrac.lif.i)
    dv = dt * p.lifMod.lifRefrac.lif.tau_mem_inv * \
        ((p.lifMod.lifRefrac.lif.v_leak - state.lifRefrac.lif.v) +
         i_new.view(batch_size, *shape))
    v_decayed = state.lifRefrac.lif.v + dv

    # compute new spikes
    z_new = threshold(v_decayed - p.lifMod.lifRefrac.lif.v_th,
                      p.lifMod.lifRefrac.lif.method, p.lifMod.lifRefrac.lif.alpha)

    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lifMod.lifRefrac.lif.v_reset

    # compute refractory update
    v_new, z_new, rho_new = compute_refractory_update(
        state.lifRefrac, z_new, v_new, p.lifMod.lifRefrac)

    # Applying delays
    z_new = delays(state, p.lifMod, state.delays, z_new, dt)

    # Compute presynaptic trace update
    X_new = compute_presynaptic_trace_update(z_new, state, p, dt)
    
    #Define values to pass before tau_max update starts
    tau_max = state.tau_max
    delays_new = state.delays
    activity = state.A
    
    image_output  = np.uint8(np.zeros((*z_new.shape[3:], 3)))
    image_output2  = np.uint8(np.zeros((*z_new.shape[3:], 3)))
    
    # Apply adaptive tau_max
    if (t)>torch.max(state.tau_max) + 50 and training== False:
        
        if par.pre_ave:
            #Presynaptic averaging approach
            tau_max, activity, image_output, image_output2 = compute_tau_max_update_pre_ave(state.X, p.lifMod.A_tar, p.lifMod, state)
        
        elif par.pre_max:
            #Presynaptic max approach 
            tau_max, activity, image_output, image_output2 = compute_tau_max_update_max( state, p.lifMod.A_tar, par, None, False)
        
        elif par.post_max:
            #Postsynaptic max approach 
            tau_max, activity, image_output, image_output2 = compute_tau_max_update_max( state, p.lifMod.A_tar, par, z_MSConv, True)
        
        elif par.post_spikes:
            #Postsynaptic output spike approach 
            tau_max, activity, image_output, image_output2 = compute_tau_max_update_post_spikes(z_MSConv, state, OF_mean, OF_magnitudes, p.lifMod)
        
        #Update delays such that they are still linearly spaced between tau_min and tau_max
        delays_new = linspace(par.MSConv_tau_min*torch.ones_like(tau_max), tau_max, p.lifMod.m).to(dtype=torch.long)

    return (
        z_new,
        LIFModFeedForwardStateNT(LIFRefracFeedForwardState(
            LIFFeedForwardState(v_new, i_new), rho_new), X_new, state.buffer, tau_max, delays_new, activity), image_output, image_output2
    )


class LIFModFeedForwardStateWTA(NamedTuple):
    """State of a modified feed forward LIF neuron with absolute refractory period.

    Parameters:
        lifRefrac (LIFRefracFeedForwardState): state of the feed forward LIFRefrac neuron integration
        buffer (torch.Tensor): buffer containing delayed spiked stiffness_bufffer (torch.Tensor): buffer containing previous values of stiffness
        v_th (torch.Tensor): current voltage threshold for each neuron 
        S (torch.Tensor): current synaptic stiffness at each neuron

    """

    lifRefrac: LIFRefracFeedForwardState
    buffer: torch.Tensor
    v_th: torch.Tensor
    S:torch.Tensor
   


def lif_mod_feed_forward_step_WTA(
    par,
    i_new: torch.Tensor,
    state: LIFModFeedForwardStateWTA,
    k: tuple,
    s: tuple,
    layer: str,
    training: bool,
    v_th_adaptive: bool,
    shape: tuple,
    p: LIFModParameters,
    dt: float = 10**(-3),

) -> Tuple[torch.Tensor, torch.Tensor, LIFModFeedForwardStateWTA]:
    """Computes a single euler-integration step of a modified feed forward
     LIF neuron-model with a refractory period. It takes as input the input current as generated by an arbitrary torch
    module or function and the sum of all the presynaptic traces for each neuron. More specifically it implements one integration
    step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}} - X
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes and X the sum of the neuron's presynaptic traces.

    Parameters:
        par: SNN parameters
        i_new (torch.Tensor): input current 
        state (LIFModFeedForwardState): state at the current time step
        k (tuple): Kernel used for current layer 
        s(tuple): Stride used for current layer
        layer (str): string specifying which layer is considered for plotting
        training (bool): indicates whether current SNN layer is being trained 
        v_th_adaptve (bool): specifies whether adptive threshold is used in this layer
        shape (tuple): shape of neuron layer (out_dim, height, width)
        p (LIFModParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """

    # compute voltage updatees
    dv = dt * p.lifRefrac.lif.tau_mem_inv * \
        ((p.lifRefrac.lif.v_leak - state.lifRefrac.lif.v) +
         i_new.view(par.batch_size, *shape))
    v_decayed = state.lifRefrac.lif.v + dv

   
    # compute new spikes
    z_new = threshold(v_decayed - state.v_th,
                      p.lifRefrac.lif.method, p.lifRefrac.lif.alpha)

    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lifRefrac.lif.v_reset

    # compute refractory update
    v_new, z_new, rho_new = compute_refractory_update(
        state.lifRefrac, z_new, v_new, p.lifRefrac)

    # Apply WTA
    z_new, z_new_batch, z_new_maps, boxes, box_indices= compute_wta(
        z_new, rho_new, v_decayed, v_new, p, k, s, training)

    # Only apllying delays for layers that are not being trained (the delay is the delay for the nex layer)
    if par.training == False:
        
        # Apply delays
        z_new = delays(state, p, p.delays, z_new, dt)
        
        # Apply adaptive threshold
        stiffness_goal = p.S_tar
        
        if v_th_adaptive:
            v_th, stiffness, image_output = compute_vth_update(z_new_maps, state, p, stiffness_goal, layer)
        
        else: 
            v_th, stiffness = (state.v_th, state.S)
            image_output  = np.uint8(np.zeros((*z_new.shape[3:], 3)))
            
      
    else:
        #Do not apply delays when training is performed 
        z_new = z_new.unsqueeze(2)
        
        #Do not perform v_th and tau_max updates when training is performed
        v_th, stiffness = (state.v_th, state.S)
        image_output = np.uint8(np.zeros((*z_new.shape[3:], 3)))

    return (
        z_new,
        LIFModFeedForwardStateWTA(LIFRefracFeedForwardState(LIFFeedForwardState(
            v_new, i_new), rho_new), state.buffer, v_th, stiffness),
        image_output, boxes, box_indices
    )
