

'''This file contains an implementation of the modified LIFneuron suggested in'Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception'''

from typing import Tuple
import numpy as np
import torch


from functional.STDP_federico2 import (
    STDPFedericoParameters,
    STDPFedericoState,
    stdp_federico_step2
)


class STDPFedericoFeedForwardCell(torch.nn.Module):
    """Module that computes a single euler-integration step of Federicos STDP learning ruke. More specifically
    it implements one integration step of the following equation.

    .. math::
        \\begin{align*}
            \Delta W_{i,j,d} = \eta (LTP + LTD)\\\\
            LTP = LTP_W \cdot LTP_{bar{X}},\\\\
            LTP_W = e^{-(W_{i,j,d} - w_{init})}, \\\\
            LTP_\bar{X} = e^{\bar{X}_{i,j,d}(t)} - a \\\\

            LTD = LTD_W \cdot LTD_{bar{X}},\\\\
            LTD_W = -e^{(W_{i,j,d} - w_{init})}, \\\\
            LTD_\bar{X} = e^{1 - \bar{X}_{i,j,d}(t)} - a \\\\

        \end{align*}

    Parameters:
        shape: shape of the weights required for the neural layer to be trained 
        p (STDPFedericoParameters): parameters of the STDP learning rule
        dt (float): Integration timestep to use

    """

    def __init__(
        self, weight_shape, shape, p: STDPFedericoParameters = STDPFedericoParameters(), dt: float = 10**(-3),
    ):
        self.weight_shape = weight_shape
        self.shape = shape
        self.p = p
        self.dt = dt

    def initial_state(self, batch_size, weights, device, dtype) -> STDPFedericoState:
        return STDPFedericoState(
            #weights = self.p.w_init *torch.ones(*self.weight_shape, device = device, dtype = dtype), 
            #Load weights from previous run 
            weights = weights,
            Li_buffer =  torch.ones(self.p.ma_len, batch_size, self.shape[0],  device = device, dtype = dtype),
            nu_spikes = torch.zeros(batch_size, *self.shape,  device = device, dtype = dtype)
        )

    def forward1(
        self, x: torch.Tensor, z: torch.Tensor,kernel: torch.Tensor, stride: torch.Tensor, pad:torch.Tensor, device, state: STDPFedericoState
        #self, x: torch.Tensor, z: torch.Tensor, s: int, device, state: STDPFedericoState, 
    ) -> Tuple[torch.Tensor, STDPFedericoState]:
        return stdp_federico_step2(x, z, kernel, stride, pad, device, state, shape = self.weight_shape, p=self.p, dt=self.dt)
       


    