import argparse
from typing import NamedTuple, Tuple

import torch


def configs():
    parser = argparse.ArgumentParser()
    
    #Simulation parameters 
    parser.add_argument('--dt',
                        type=float,
                        help="Simulation time step [s]",
                        metavar='',
                        default= 0.001)
    parser.add_argument('--batch_size',
                        type=int,
                        help="Batch size",
                        metavar='',
                        default= 1)
    parser.add_argument('--iterations',
                        type=int,
                        help="Number of iteration",
                        metavar='',
                        default= 1)
    parser.add_argument('--new_weights',
                        type=bool,
                        help="Specifies if new set of weights shall be created",
                        metavar='',
                        default= False)
    parser.add_argument('--SSConv_weights_name',
                        type=str,
                        help="Name of trained SSConv weights",
                        metavar='',
                        default= 'SSConvWeights_rot_disk.pt')
    parser.add_argument('--SSConv_weights_dir',
                        type=str,
                        help="Directory of SSConv weights",
                        metavar='',
                        default= 'weights/SSConv/rot_disk')
    parser.add_argument('--MSConv_weights_name',
                        type=str,
                        help="Name of trained MSConv weights",
                        metavar='',
                        default= 'MSConvWeights_rot_disk')
    parser.add_argument('--MSConv_weights_dir',
                        type=str,
                        help="Directory of MSConv weights",
                        metavar='',
                        default= 'weights/MSConv/rot_disk')
    parser.add_argument('--training',
                        type=bool,
                        help="Specifies if training shall be performed. Change dataset directory accordingly",
                        metavar='',
                        default= False)
    
    #Flags for tau_max update rule 
    parser.add_argument('--pre_ave', 
                       type = bool,
                       help = 'Use presynaptic averaging approach for tau_max update',
                       default = True)
    parser.add_argument('--pre_max', 
                       type = bool, 
                       help = 'Use presynaptic max approach for tau_max update',
                       default = False)
    parser.add_argument('--post_max', 
                       type = bool, 
                       help = 'Use postsynaptic max approach for tau_max update',
                       default = False)
    parser.add_argument('--post_spikes', 
                       type = bool, 
                       help = 'Use postsynaptic spike approach for tau_max update',
                       default = False)
    
    #Flags for saving results 
    parser.add_argument('--save_videos', 
                       type = bool, 
                       help = 'Set to true to save video of all cv2 plots. Only recommended for short runs',
                       default = True)
    parser.add_argument('--save_results', 
                       type = bool, 
                       help = 'Set to true to save time history of v_th, tau_max, S, and A .Only recommended for short runs',
                       default = True)
    
    
    
    #Parameters concerning the dataset 
    parser.add_argument('--par_name',
                        type=str,
                        help="Name of the dataset",
                        metavar='',
                        default='ODA_dataset')
    parser.add_argument('--directory',
                        type=str,
                        help="Directory of the dataset. Note different folders for training and inference",
                        metavar='',
                        default= 'data_tensors/ODA_dataset')
    parser.add_argument('--height',
                        type=int,
                        help="Height of the input image",
                        metavar='',
                        default= 180)
    parser.add_argument('--width',
                        type=int,
                        help="Width of the input image",
                        metavar='',
                        default= 240)
    
    
    
    #Input layer architecture
    parser.add_argument('--input_in_dim0',
                        type=int,
                        help="Number of channels in input layer",
                        metavar='',
                        default= 2)
    parser.add_argument('--input_out_dim',
                        type=int,
                        help="Number of output maps in input layer",
                        metavar='',
                        default= 2)
    parser.add_argument('--input_kernel',
                        type=int,
                        help="Kernel size in input layer",
                        metavar='',
                        default= 2)
    parser.add_argument('--input_stride',
                        type=int,
                        help="Stride in input layer",
                        metavar='',
                        default= 2)
    parser.add_argument('--input_padding',
                        type=int,
                        help="Padding in input layer",
                        metavar='',
                        default= 0)
    parser.add_argument('--input_m',
                        type=int,
                        help="Number of delays in input layer",
                        metavar='',
                        default= 1)
    
    
    
    #SSConv layer architecture
    parser.add_argument('--SSConv_out_dim',
                        type=int,
                        help="Number of output maps in SSConv layer",
                        metavar='',
                        default= 32)
    parser.add_argument('--SSConv_kernel',
                        type=int,
                        help="Kernel size in SSConv layer",
                        metavar='',
                        default= 5)
    parser.add_argument('--SSConv_stride',
                        type=int,
                        help="Stride in SSConv layer",
                        metavar='',
                        default= 1)
    parser.add_argument('--SSConv_padding',
                        type=int,
                        help="Padding in SSConv layer",
                        metavar='',
                        default= 0)
    parser.add_argument('--SSConv_m',
                        type=int,
                        help="Number of delays in SSConv layer",
                        metavar='',
                        default= 1)
    
    #SSConv layer original neuron parameters
    parser.add_argument('--SSConv_w_init',
                        type=float,
                        help="Initial value of weights in SSConv layer",
                        metavar='',
                        default= 0.5)
    parser.add_argument('--SSConv_lambda_v',
                        type=float,
                        help="Membrane potential time constant in SSConv layer [s]",
                        metavar='',
                        default= 0.005)
    parser.add_argument('--SSConv_lambda_X',
                        type=float,
                        help="Presynaptic trace time constant in SSConv layer [s]",
                        metavar='',
                        default= 0.005)
    parser.add_argument('--SSConv_v_th',
                        type=float,
                        help="Initial voltage threshold in SSConv layer",
                        metavar='',
                        default= 0.3)
    parser.add_argument('--SSConv_alpha',
                        type=float,
                        help="Presynaptic trace scaling factor in SSConv layer",
                        metavar='',
                        default= 0.3)
    parser.add_argument('--SSConv_delay',
                        type=torch.Tensor,
                        help="Delay in SSConv layer",
                        metavar='',
                        default= torch.Tensor([1]).long())
    parser.add_argument('--SSConv_delta_refr',
                        type=float,
                        help="Refractory period in SSConv layer [ms]",
                        metavar='',
                        default= 1)

    
    #SSConv layer adaptive neuron parameters 
    parser.add_argument('--SSConv_lambda_vth',
                        type=float,
                        help="Voltage threshold decay factor in SSConv layer",
                        metavar='',
                        default= 0.6)
    parser.add_argument('--SSConv_vth_rest',
                        type=float,
                        help="Voltage threshold resting value in SSConv layer",
                        metavar='',
                        default= 0.01)
    parser.add_argument('--SSConv_vth_conv_params',
                        type=list,
                        help="Voltage threshold convolutional parameters in SSConv layer",
                        metavar='',
                        default= [(1, 86, 116 ),(1, 1, 1),(0,0, 0)])
    parser.add_argument('--SSConv_S_tar',
                        type=torch.tensor,
                        help="Target value of synaptic stiffness in SSConv layer",
                        metavar='',
                        default= torch.as_tensor(2))
    parser.add_argument('--SSConv_n_vth',
                        type=float,
                        help="Voltage threshold learning rate in SSConv layer",
                        metavar='',
                        default= 0.05) #0.05
    
    #Merge layer architecture
    parser.add_argument('--merge_out_dim',
                        type=int,
                        help="Number of output maps in merge layer",
                        metavar='',
                        default= 1)
    parser.add_argument('--merge_kernel',
                        type=int,
                        help="Kernel size in merge layer",
                        metavar='',
                        default= 1)
    parser.add_argument('--merge_stride',
                        type=int,
                        help="Stride in merge layer",
                        metavar='',
                        default= 1)
    parser.add_argument('--merge_padding',
                        type=int,
                        help="Padding in merge layer",
                        metavar='',
                        default= 0)
    parser.add_argument('--merge_m',
                    type=int,
                    help="Number of delays in merge layer",
                    metavar='',
                    default= 1)
    
    #Merge layer neuron parameters
    parser.add_argument('--merge_w_init',
                        type=float,
                        help="Initial value of weights in merge layer",
                        metavar='',
                        default= 1)
    parser.add_argument('--merge_lambda_v',
                        type=float,
                        help="Membrane potential time constant in merge layer [s]",
                        metavar='',
                        default= 0.005)
    parser.add_argument('--merge_lambda_X',
                        type=float,
                        help="Presynaptic trace time constant in merge layer [s]",
                        metavar='',
                        default= 0.005)
    parser.add_argument('--merge_v_th',
                        type=float,
                        help="Voltage threshold in merge layer",
                        metavar='',
                        default= 0.)
    parser.add_argument('--merge_delay',
                        type=torch.Tensor,
                        help="Delay in merge layer",
                        metavar='',
                        default= torch.Tensor([1]).long())
    parser.add_argument('--merge_delta_refr',
                        type=float,
                        help="Refractory period in merge layer [ms]",
                        metavar='',
                        default= 1)

    
    
    #MSConv layer architecture
    parser.add_argument('--MSConv_out_dim',
                        type=int,
                        help="Number of output maps in MSConv layer",
                        metavar='',
                        default= 64)
    parser.add_argument('--MSConv_kernel',
                        type=int,
                        help="Kernel size in MSConv layer",
                        metavar='',
                        default= 5)
    parser.add_argument('--MSConv_stride',
                        type=int,
                        help="Stride in MSConv layer",
                        metavar='',
                        default= 1)
    parser.add_argument('--MSConv_padding',
                        type=int,
                        help="Padding in MSConv layer",
                        metavar='',
                        default= 0)
    parser.add_argument('--MSConv_m',
                        type=int,
                        help="Number of delays in MSConv layer",
                        metavar='',
                        default= 10)
    parser.add_argument('--MSConv_m_after',
                        type=int,
                        help="Number of delays after MSConv layer",
                        metavar='',
                        default= 1)
    
    #MSConv layer original neuron parameters
    parser.add_argument('--MSConv_w_init_exc',
                        type=float,
                        help="Initial value of excitatory weights in MSConv layer",
                        metavar='',
                        default= 0.5)
    parser.add_argument('--MSConv_w_init_inh',
                        type=float,
                        help="Initial value of inhibitory weights in MSConv layer",
                        metavar='',
                        default= -0.5)
    parser.add_argument('--MSConv_lambda_v',
                        type=float,
                        help="Membrane potential time constant in MSConv layer [s]",
                        metavar='',
                        default= 0.03)
    parser.add_argument('--MSConv_lambda_X',
                        type=float,
                        help="Presynaptic trace time constant in MSConv layer [s]",
                        metavar='',
                        default= 0.03)
    parser.add_argument('--MSConv_v_th',
                        type=float,
                        help="Initial voltage threshold in MSConv layer",
                        metavar='',
                        default= 0.2)
    parser.add_argument('--MSConv_alpha',
                        type=float,
                        help="Presynaptic trace scaling factor in MSConv layer",
                        metavar='',
                        default= 0.3)
    parser.add_argument('--MSConv_tau_min',
                        type=torch.Tensor,
                        help="Minimum delay in MSConv layer",
                        metavar='',
                        default= torch.as_tensor(1.))
    parser.add_argument('--MSConv_tau_max',
                        type=torch.Tensor,
                        help="Initial maximum delay in MSConv layer",
                        metavar='',
                        default= torch.as_tensor(200.))
    parser.add_argument('--MSConv_delays_after',
                        type=torch.Tensor,
                        help="Delay that is applied to output of MSonv layer",
                        metavar='',
                        default= torch.Tensor([0]).long())
    parser.add_argument('--MSConv_beta',
                        type=float,
                        help="Weight of inhibitory synaptic efficacies",
                        metavar='',
                        default= 0.5)
    parser.add_argument('--MSConv_delta_refr',
                        type=float,
                        help="Refractory period in MSConv layer [ms]",
                        metavar='',
                        default= 1)

    #MSConv layer adaptive neuron parameters 
    parser.add_argument('--MSConv_lambda_vth',
                        type=float,
                        help="Voltage threshold decay factor in MSConv layer",
                        metavar='',
                        default= 0.5)
    parser.add_argument('--MSConv_vth_rest',
                        type=float,
                        help="Voltage threshold resting value in MSConv layer",
                        metavar='',
                        default= 0.1)
    parser.add_argument('--MSConv_vth_conv_params',
                        type=list,
                        help="Voltage threshold convolutional parameters in MSConv layer (kenerl size, stride, padding)",
                        metavar='',
                        default= [(1, 82, 112),(1, 1, 1),(0,0,0)])
    parser.add_argument('--MSConv_S_tar',
                        type=torch.tensor,
                        help="Target value of synaptic stiffneMS in MSConv layer",
                        metavar='',
                        default= torch.as_tensor(2))
    parser.add_argument('--MSConv_n_vth',
                        type=float,
                        help="Voltage threshold learning rate in MSConv layer",
                        metavar='',
                        default= 0.05) #0.05
    parser.add_argument('--MSConv_tau_max_conv_params',
                        type=list,
                        help="Convolutional parameters for tau_max update",
                        metavar='',
                        default= [(1, 86, 116),(1, 1, 1),(0,0, 0)])
    parser.add_argument('--MSConv_A_tar',
                        type=torch.tensor,
                        help="Target value of relative optic flow magnitude in MSConv layer",
                        metavar='',
                        default= 7.8)
    parser.add_argument('--MSConv_n_tau_max',
                        type=float,
                        help="Maximum synaptic delay learning rate in MSConv layer",
                        metavar='',
                        default= -0.5) #-0.5
    parser.add_argument('--MSConv_c_i_th',
                        type=float,
                        help="Presynaptic trace quantity threshold",
                        metavar='',
                        default= 0)
    parser.add_argument('--MSConv_c_j_th',
                        type=float,
                        help="Presynaptic trace magnitude threshold",
                        metavar='',
                        default= 0.02)
    
    
    
    #Training parameters 
    parser.add_argument('--training_n',
                        type=float,
                        help="Learning rate during training",
                        metavar='',
                        default= 10**(-4))
    parser.add_argument('--training_a',
                        type=float,
                        help="Factor to control spread of weights",
                        metavar='',
                        default= 0)
    parser.add_argument('--training_L',
                        type=float,
                        help="Magnitude of stopping criterion",
                        metavar='',
                        default= 5*10**(-2))
    parser.add_argument('--training_ma_len',
                        type=float,
                        help="Length of moving average window to compute stopping criterion",
                        metavar='',
                        default= 250)
    
    
    #Map colours in the SSConv layer 
    parser.add_argument('--map_colours',
                        type=list,
                        help="BGR map colours used in the SSConv layer",
                        metavar='',
                        default= [
                        (255, 0, 0),      
                        (165,63,2),
                        (227,111,74),
                        (225,149,133),
                        (185,135,125),
                        (212,193,190),
                        (227,187,181),
                        (192,188,214),
                        (132,119,187),
                        (185,175,230),
                        (145,123,224),
                        (106,63,211),
                        (128, 0, 128),    
                        (0, 0, 255),     
                        (59,6,142),
                        (0, 140, 255),     
                        (8,151,239),
                        (212,156,247),
                        (198,211,234),
                        (141,185,240),
                        (225,196,246),
                        (235,225,243),
                        (199,222,198),
                        (231,234,213),
                        (214,222,156),
                        (147,213,141),
                        (255, 255, 0),    
                        (192,207,15),
                        (56,198,17),
                        (0, 255, 0),      
                        (0, 128, 0),     
                        (0, 128, 128),    
                        (128, 128, 0),   
                        (128, 0, 0),    
                        (192, 192, 192),  
                        (128, 128, 240)])
    
    
    
    #Parse arguments so they can be accessed for the part below 
    args = parser.parse_args()
    
    #Define function to add additional arguments which are directly computed from the ones defined above
    def compute_weight_shapes(
        layer: str, 
        in_dim: int, 
        out_dim: int, 
        m: int, 
        kernel: int, 
        stride: int, 
        padding: int,
        w_inits: list,
        w_names:list):
        
        """
        Add the number of input channels, the 3D kernel size, the 3D stride, the shape of the weights, the presynaptic trace weights and the weights to the parser.
        Parameters: 
            layer (str): Layer for which the operations are performed 
            in_dim (int): The number of input channels in the layer
            out_dim (int): The number of output channels in the layer
            m (int): The number delays in the layer
            kernel (int): The kernel size in the layer
            stride (int): The stride in the layer
            padding (int): The padding in the layer
            w_inits (list): The initial values of the weights in the layer. Specify two values for the MSConv layer corresponding to excitatory and inhibitory weights 
            w_names (list): The names the weights in the layer. Specify two values for the MSConv layer corresponding to excitatory and inhibitory weights 
        """
            
       
    
        parser.add_argument('--{layer}_in_dim'.format(layer = layer),
                            type=int,
                            help="Number of input channels in {layer} layer".format(layer = layer),
                            metavar='',
                            default= in_dim)
        
        parser.add_argument('--{layer}_kernel3D'.format(layer = layer),
                            type=tuple,
                            help="3D kernel of {layer} layer".format(layer = layer),
                            metavar='',
                            default= (m, kernel, kernel))
        
        parser.add_argument('--{layer}_stride3D'.format(layer = layer),
                            type=tuple,
                            help="3D stride of {layer} layer".format(layer = layer),
                            metavar='',
                            default= (m, stride, stride))
        
        parser.add_argument('--{layer}_padding3D'.format(layer = layer),
                            type=tuple,
                            help="3D padding of {layer} layer".format(layer = layer),
                            metavar='',
                            default= (0, padding, padding))
        
        parser.add_argument('--{layer}_weight_shape'.format(layer = layer),
                            type=int,
                            help="Shape of {layer} weights".format(layer = layer),
                            metavar='',
                            default= (out_dim, in_dim, m, kernel, kernel))
           
        parser.add_argument('--{layer}_weights_pst'.format(layer = layer),
                            type=int,
                            help="Weights of presynaptic trace in {layer} layer".format(layer = layer),
                            metavar='',
                            default= torch.ones((out_dim, in_dim, m, kernel, kernel)) )
        
        for i, w_init in enumerate(w_inits):
            parser.add_argument('--{layer}_weights{kind}'.format(layer = layer, kind = w_names[i]),
                            type=int,
                            help="Initial {layer} weights{kind}".format(layer = layer, kind = w_names[i]),
                            metavar='',
                            default= (w_init*torch.ones((out_dim, in_dim, m, kernel, kernel))))
    
    
    #Extract already defined arguments ([input, SSConv, merge, MSConv])
    #Number of input channels (= number of output channels in previous layer)
    in_dims = [args.input_in_dim0, args.input_out_dim, args.SSConv_out_dim, args.merge_out_dim]
    
    #Number of output channels
    out_dims = [args.input_out_dim, args.SSConv_out_dim, args.merge_out_dim, args.MSConv_out_dim]
    
    #Number of delays 
    ms = [args.input_m, args.SSConv_m, args.merge_m, args.MSConv_m]
    
    #Kernel sizes 
    kernels = [args.input_kernel, args.SSConv_kernel, args.merge_kernel, args.MSConv_kernel]
    
    #Stride sizes 
    strides = [args.input_stride, args.SSConv_stride, args.merge_stride, args.MSConv_stride]
    
    #Padding sizes 
    paddings = [args.input_padding, args.SSConv_padding, args.merge_padding, args.MSConv_padding]
    
    #Initial values of weights, using sublists since MSConv layer has excitatory and inhibitory weights
    w_inits = [[1], [args.SSConv_w_init], [args.merge_w_init], [args.MSConv_w_init_exc, args.MSConv_w_init_inh]]
    
    #Names of weights 
    w_names = [[''], [''], [''], ['_exc', '_inh']]
    
    #Add additional arguements for the SSConv, merge and MSConv layers
    for idx, layer in enumerate(['input', 'SSConv', 'merge', 'MSConv']):
        compute_weight_shapes(layer, in_dims[idx], out_dims[idx], ms[idx], kernels[idx], strides[idx], paddings[idx], w_inits[idx], w_names[idx] )

    #Add linearly spaced delays in MSConv layer
    parser.add_argument('--MSConv_delay',
                        type=torch.tensor,
                        help="Linearly spaced delays in MSConv layer",
                        metavar='',
                        default= torch.linspace(args.MSConv_tau_min, args.MSConv_tau_max, args.MSConv_m).to(dtype=torch.long))
    
    
    #Parse arguments 
    args = parser.parse_args()
    
    return args



