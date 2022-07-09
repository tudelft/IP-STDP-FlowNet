

'''This file contains an implementation of a neuron with no interal dynamics and keeps track of its postsynaptic spike train'''

from typing import Tuple
import numpy as np
import torch

from norse.torch.functional.lif import LIFState, LIFFeedForwardState

from functional.noDyn2 import (
    noDynParameters,
    noDynState,
    noDynFeedForwardState,
    no_dyn_feed_forward_step,
)

from collections import deque

class noDynFeedForwardCell(torch.nn.Module):
    """Module that computes a single step of a neuron with no internal dynamics and keeps track of its postsynaptic trace.
    

    Parameters:
        shape: Shape of the processed spike input
        p (LIFRefracParameters): parameters of the noDyn neuron
        dt (float): Integration timestep to use

    Examples:
        >>> batch_size = 16
        >>> noDyn = noDynFeedForwardCell((20, 30))
        >>> input = torch.randn(batch_size, 20, 30)
        >>> s0 = noDyn.initial_state(batch_size)
        >>> output, s0 = noDyn(input, s0)
    """

    def __init__(
        self, shape, p: noDynParameters = noDynParameters(), dt: float = 10**(-3),
    ):
        super(noDynFeedForwardCell, self).__init__()
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, device, dtype) -> noDynFeedForwardState:
        return noDynFeedForwardState(
          
            X=torch.zeros(batch_size, self.shape[0], self.p.m, self.shape[1], self.shape[2],  device=device, dtype=dtype),
            buffer = deque(torch.zeros(int(max(self.p.delays)*10**(-3)/self.dt) + 1, batch_size,  *self.shape, device=device, dtype=dtype)),
            #buffer2 = torch.zeros(int(max(self.p.delays)*10**(-3)/self.dt) + 1, batch_size,  *self.shape, device=device, dtype=dtype)
            
        )

    def forward(
        self, batch_size: torch.Tensor, input_tensor: torch.Tensor,  state: noDynFeedForwardState
    ) -> Tuple[torch.Tensor, noDynFeedForwardState]:
        return no_dyn_feed_forward_step(batch_size, input_tensor, state, shape = self.shape, p=self.p, dt=self.dt)



if __name__ == "__main__":
    noDyn = noDynFeedForwardCell(torch.zeros(2,2), noDyn_parameters)