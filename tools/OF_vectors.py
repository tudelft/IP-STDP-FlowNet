import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from scipy.stats import linregress

from tools.plotting import plot_histogram

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })




class compute_OF():

    def __init__(
        self, par, MSConv_kernels_exc, MSConv_kernels_inh, device, gamma = 0
    ):
        #Loading MSConv weights 
        self.MSConv_kernels_exc = MSConv_kernels_exc
        self.MSConv_kernels_inh = MSConv_kernels_inh
        self.gamma = gamma
        self.par = par
        self.beta = self.par.MSConv_beta

        #Combining excitatory and inhibitory weights 
        self.MSConv_kernels = self.MSConv_kernels_exc + self.beta * self.MSConv_kernels_inh


    def compute_OF(self):

        #Setting up tensor to collect OF results 
        OF = np.zeros((self.MSConv_kernels.shape[0], 2))
        OF_norm, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v = (0, 0, 0, 0, 0, 0, 0)

        
        #Determine max for each map 
        MSConv_kernels_max = torch.amax(self.MSConv_kernels, dim = (3,4), keepdim=True)

        #Determeine min sum for each map 
        MSConv_kernels_min = torch.amin(self.MSConv_kernels, dim = (3,4) , keepdim=True)

        #Normalize MSConv_kernels (for automatic parameter tuning)

        MSConv_kernels_norm = (self.MSConv_kernels - MSConv_kernels_min)/(MSConv_kernels_max - MSConv_kernels_min)
        
        
        
        
        #Compute sum of weights in each group of synapses
        MSConv_kernels_sum = torch.sum(self.MSConv_kernels, dim = (3,4))

        #Determine max sum for each map 
        MSConv_kernels_sum_max, __ = torch.max(MSConv_kernels_sum, dim = 2, keepdim=True)

        #Determeine min sum for each map 
        MSConv_kernels_sum_min, __ = torch.min(MSConv_kernels_sum, dim = 2, keepdim=True)

      
        #Normalize sums 
        MSConv_kernel_sum_norm = (MSConv_kernels_sum - MSConv_kernels_sum_min)/(MSConv_kernels_sum_max - MSConv_kernels_sum_min)

        #Finding which sum of weights is larger than scaled maximum sum 
        condition = torch.ge(MSConv_kernel_sum_norm, self.gamma)

        #Determine indices of tau min and tau max 
        indices = torch.nonzero(condition)

        #Computing number of synaptic groups larger than scaled max for each map
        __, num_larger = torch.unique(indices[:,0], return_counts = True)

        #Splitting indices into bits corresponding to the different maps
        split_indices = torch.split(indices,tuple(num_larger))
        tau_min = [element[0,2] for element in split_indices]
        tau_max = [element[-1,2] for element in split_indices] 
        
    
        #Extracting histograms
        tuple0 = tuple(torch.arange(self.MSConv_kernels.shape[0]))
        tuple1 = tuple(torch.arange(self.MSConv_kernels.shape[1]))


        hist_tau_min_hor = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_min], dim = 1)
        hist_tau_min_ver = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_min], dim = 2)

        hist_tau_max_hor = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_max], dim = 1)
        hist_tau_max_ver = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_max], dim = 2)

        #Computing difference in histograms 
        hist_hor = (hist_tau_min_hor - hist_tau_max_hor).squeeze() 
        hist_ver = -(hist_tau_min_ver - hist_tau_max_ver).squeeze()#Applying minus to make OF upwards positive 

        #Estimating slope with least squares (norm(Ax-B) = 0)
        A_hor = torch.arange(hist_hor.shape[-1]).expand(hist_hor.shape[0], -1)
        A_ver = torch.arange(hist_ver.shape[-1]).expand(hist_ver.shape[0], -1)

        linregress2D = np.vectorize(linregress, signature='(n),(n)->(),(),(),(),()')
        
        theta_u = linregress2D(np.array(A_hor), np.array(hist_hor.cpu()))
        theta_v = linregress2D(np.array(A_ver), np.array(hist_ver.cpu()))
        
        
        tau_max = torch.tensor(tau_max)
        tau_min = torch.tensor(tau_min)
        
        #Computing OF 
        OF[:,0] = theta_u[0]/(tau_max - tau_min)
        OF[:,1] = theta_v[0]/(tau_max - tau_min)
       
        # #Normalizing OF according to vector length 
        OF_lengths = (OF[:,0]**2 + OF[:,1]**2)**0.5
        OF_lengths_max = np.max(OF_lengths)
        OF_lengths_min = np.min(OF_lengths)
        
        scale = 1/OF_lengths_max
        OF_norm = scale*OF
      
        return OF_norm, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v, OF


    

    def plot_histograms(self,map_nu, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v):
        #Plotting results 
        plt.figure()
        plt.subplot(1,2,1)
        
        plt.bar(A_hor[map_nu], np.array(hist_hor[map_nu].cpu()))
        plt.plot(theta_u[1][map_nu]+ theta_u[0][map_nu]*A_hor[map_nu], 'r')
        plt.grid()
        plt.title("Horizontal histogram")
        plt.xlabel("X-neuron")
        plt.ylabel("Sum of weights")
        plt.subplot(1,2,2)
       
        plt.bar(A_ver[map_nu], np.array(hist_ver.cpu()[map_nu]))
        plt.plot(theta_v[1][map_nu]+ theta_v[0][map_nu]*A_ver[map_nu], 'r')
        plt.grid()
        plt.title("Vertical histogram")
        plt.xlabel("Y-neuron")
        plt.ylabel("Sum of weights")
        plt.show()

    def find_OF_from_colour(self, map_colours, colourmap):
        
        #Load image
        img = cv2.imread(colourmap)
        
        for colour in map_colours:
            x,y = np.where(np.all(img==colour,axis=2))


def plot_OF(colourmap, OF):
    fig, ax = plt.subplots(figsize = (0.5*2*3.50069, 0.5*2*3.50069))
    img = cv2.imread(colourmap)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent = (-1,1,-1,1))
    ax.scatter(OF[:,0], OF[:,1], marker = 'x', c ='gray')
    ax.set_xlabel('u [-]')
    ax.set_ylabel('v [-]')
    ax.set_xticks(np.arange(-1, 1.5, 0.5))
    ax.set_yticks(np.arange(-1, 1.5, 0.5))
    # ax.set_axis_off()
    n = np.arange(1, len(OF)+1)

    #Print map numbers next to crosses
    for i, txt in enumerate(n):
        xy = (OF[i,0], OF[i,1])
        xy_text = (OF[i,0] + 0.03, OF[i,1] - 0.013)
       
        #Manually move some overlapping text 
        if i in [55, 57, 26, 40, 45, 56, 29, 62]:
            xy_text = (OF[i,0] - 0.13, OF[i,1] - 0.01)
      
        # ax.annotate(txt,xy, xy_text, color ='white', fontsize = 'tiny')
        ax.annotate(txt,xy, xy_text, color ='white', fontsize = 6)
        
    
    fig.savefig('OF_map_rot_disk.pgf' , bbox_inches = 'tight', pad_inches = 0.0)
    fig.savefig('OF_map_rot_disk.pdf' , bbox_inches = 'tight', pad_inches = 0.0)
    
    # plt.show()


def find_OF_from_colour(map_colours, colourmap):
    
    #Load image
    img = cv2.imread(colourmap)
    #Initialize OF vector 
    OF = np.zeros((map_colours.shape[0], 2))
    
    for idx, colour in enumerate(map_colours):
        #Computing difference between colourmap and provided colours
        diff = np.abs(img - np.array(colour))
        #Compute sum of differences per colour channel 
        diff_sum = np.sum(diff, axis = 2)
        #Find index with smallest difference
        smallest_idx = np.unravel_index(np.argmin(diff_sum, axis=None), diff_sum.shape)
        #Convert index into OF vector 
        u = (smallest_idx[1] - img.shape[1]/2)/(img.shape[1]/2-1)
        v = (smallest_idx[0]-img.shape[0]/2)/(-img.shape[0]/2-1)

        OF[idx] = np.array([u,v])
    
    return OF

def plot_map_colours(title, size, position, waitkey, map_colours):
    img = np.round(map_colours.view(8,8,3))
    img = np.array(img).astype('uint8')

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size)
    cv2.moveWindow(title, *position) 
    cv2.imshow(title, img )
    cv2.waitKey(waitkey)
    
def flow_viz_np(flow_x, flow_y):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi 
    ang *= 180. / np.pi / 2.
    ang -=60
    ang = ang%180
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow_rgb


