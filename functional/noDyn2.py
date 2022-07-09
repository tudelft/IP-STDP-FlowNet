

from typing import NamedTuple, Tuple

import torch
import _testimportmultiple
import time


class noDynState(NamedTuple):
    """State of neuron with no internal dynamics.

    Parameters:
        X (torch.Tensor): presynaptic trace state 
        buffer(torch.Tensor): buffer containing delayed spikes
    """

    X: torch.Tensor 
    buffer: torch.Tensor 


class noDynParameters(NamedTuple):
    """Parameters of neuron with no internal dynamics.

    Parameters:
        alpha_mod (torch.Tensor): presynaptic scaling factor  
        lambda_x (torch.Tensor) : time constant of system
         
    """

    alpha_mod: torch.Tensor = torch.as_tensor(0.25)
    lambda_x: torch.Tensor = torch.as_tensor(5)
    delays: torch.Tensor = torch.Tensor([1]).long()
    m: torch.Tensor = torch.as_tensor(1)



def compute_presynaptic_trace_update(
    z_new: torch.Tensor,
    state: noDynState,
    p: noDynParameters = noDynParameters(),
    dt: float = 0.01,
) -> Tuple[torch.Tensor]:
     
    """Compute the presynaptic update according to the specified ODE: 
     .. math::
        \begin{align*}
            \dot{X} &= 1/\lambda_{\text{x}} (-X + \alpha s)\\
        \end{align*}
    
    where :math:`s` corresponds to the variable "raw_input".

    Parameters
        state (LIFModState): Initial state of the modified refractory neuron.
        p (torch.Tensor): Modified refractoryp.
        z_new (torch.tensor): Output spikes
    """

    #Compute presynaptic trace update  
    dX = dt*1/p.lambda_x *(-state.X + p.alpha_mod*z_new)
    X = state.X + dX

    return X

def delays(
    state: noDynState, 
    p: noDynParameters,
    z_new: torch.tensor, 
    dt: float
)-> Tuple[torch.Tensor]:

    #start_time = time.time()
    #Moving all entries by one index 
    state.buffer.rotate(1) 
    #Putting latest spikes into buffer 
    state.buffer[0] = z_new
    #print ("deque took ", time.time() - start_time, "to run")

    # start_time = time.time()
    # #Moving all entries by one index 
    # state.buffer2[1:] = state.buffer2[:-1]
    # #Putting latest spikes into buffer 
    # state.buffer2[0] = z_new
    # print ("indices took ", time.time() - start_time, "to run")

   



    #Computing factor of timestep [specified by dt] and delay[ms]
    factor = int(10**(-3)/dt)
    #Outputting spikes with desired delays 
    z_new = torch.stack(tuple(state.buffer))[factor*p.delays]
    #Reshaping the tensor to the suitable format for 3D convolution (Batch size, number of channels, number of synapses, height, width)
    z_new = z_new.permute(1,2,0,3,4)

    return z_new


class noDynFeedForwardState(NamedTuple):
    """State of a modified feed forward LIF neuron with absolute refractory period.

    Parameters:
        X (torch.Tensor): presynaptic trace state 
        buffer(torch.Tensor): buffer containing delayed spikes

    """
    X: torch.Tensor
    buffer: torch.Tensor 
    #buffer2:torch.Tensor


def no_dyn_feed_forward_step(
    batch_size: torch.Tensor,
    input_tensor: torch.Tensor,
    state: noDynFeedForwardState,
    shape: tuple,
    p: noDynParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor,noDynFeedForwardState]:
    r"""Forwards the input spikes of the event camera with no internal dyanmics, keeps track of the presynaptic traces for the following layer.
    


    Parameters:
        input_tensor (torch.Tensor): input spikes from the event camera 
        s (LIFModFeedForwardState): state at the current time step
        p (LIFModParameters): parameters of the noDyn neuron
        dt (float): Integration timestep to use
    """

    # compute new spikes
    z_new = input_tensor

    #Applying delays 
    z_new = delays(state, p, z_new, dt)

    #Compute presynaptic trace update 
    X_new = compute_presynaptic_trace_update(z_new, state, p, dt)

    return (
        z_new,
        noDynFeedForwardState(X_new, state.buffer)
    )


if __name__ == "__main__":
    noDyn_parameters = noDynParameters(alpha_mod = 0.25, lambda_x = 0.5)
    