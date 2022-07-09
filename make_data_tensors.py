import bz2

import _pickle as cPickle
import numpy as np
import pandas as pd
import torch
import torchvision

from config_files import configs_ODA, configs_rot_disk


def get_csv_input(height, width, dir, dir_result, dt = 10**(-3), dt_data = 10**(-6), data_flipped = False, start = 0, finish = 0):
        '''This function converts the panda data frane obtained from a csv file into a tensor with diescrete timebins containing the events. 
        
        Parameters: 
            height (int): height of the image
            width (int): width of the image
            dir (str): directory containing the csv file 
            dir_result (str): name of directory in which result shall be saved
            dt (float): size of time bins (seconds)
            dt_data (float): time step in event data
            data_flipped (bool): specifies whether polarity is given denoted by [-1,1] instead of [0,1]
            start (int): start time from which to consider events [mus]
            finish (int): specifies how much time to disregard at the end of the sequence [mus]
        '''
        
        #Read image data 
        data = pd.read_csv(dir, header = 0, names= ['t', 'x', 'y', 'pol'])
  
        #Determine time of first event 
        offset = data.t[0 + start] 
        end = data.t.iloc[-1 - finish] 

        #Determine length of data
        length = np.ceil((end - offset)*(dt_data)/dt) + 1 


        #Initializing tensor with correct dimensions(time [ms] x 1 x number of channels x pixel height x pixel width)
        Input = torch.zeros((int(length), 1, 2, height, width))
        
        if data_flipped:
            #Converting -1, 1 polarity to 0,1 values 
            data.pol = round((data.pol+1)/2)
            #Reading out data
        
        for i in range(start, len(data.t) - finish):
            #Finding index of event
            idx = np.floor((data.t[i] - offset)*(dt_data)/dt)
            Input[int(idx), 0, int(data.pol[i]), height - 1 - int(data.y[i]), width - 1 - int(data.x[i])] = 1
                
        with bz2.BZ2File(dir_result, 'w') as f: 
            cPickle.dump(Input, f)
           

        return Input





def make_rotating_disk_data(height, width, dir, dir_result, dt = 10**(-3), dt_data = 10**(-6), data_flipped = False):
        '''This function creates tensors containing the events for the rotating disk sequence and additionally creates a cropped and rotated version (csv files). 
        
        Parameters:
            height (int): height of the image
            width (int): width of the image
            dir (str): directory containing the aedat files
            dir_results (str): directory in which tensors shall be saved
            dt (float): simulation time step
            dt_data (float): times step of the data
        '''

        print(' \t - Creating unaltered sequence')
        data = get_csv_input(height, width, dir, dir_result + 'inference/rot_disk.pbz2', dt = dt, dt_data = dt_data, data_flipped=data_flipped)

        #Crop sequence to square window
        print('\t - Saving cropped sequence')
        data_cropped = torchvision.transforms.CenterCrop(height)(data)
        with bz2.BZ2File(dir_result + 'training/rot_disk_cropped.pbz2', 'w') as f: 
            cPickle.dump(data_cropped, f)

        #Rotate sequence by +180 degrees
        print('\t - Saving sequence rotated +180 degrees')
        data_rotated_plus_180 = torch.rot90(data_cropped, (2), (3,4))
        with bz2.BZ2File(dir_result + 'training/rot_disk_rotated_plus180.pbz2', 'w') as f: 
            cPickle.dump(data_rotated_plus_180, f)
        
        #Rotate sequence by +90 degrees
        print('\t - Saving sequence rotated +90 degrees')
        data_rotated_plus = torch.rot90(data_cropped, (1), (3,4))
        with bz2.BZ2File(dir_result + 'training/rot_disk_rotated_plus90.pbz2', 'w') as f: 
            cPickle.dump(data_rotated_plus, f)

        #Rotate sequence by -90 degrees
        print('\t - Saving sequence rotated -90 degrees')
        data_rotated_minus = torch.rot90(data_cropped, (-1), (3,4))
        with bz2.BZ2File(dir_result + 'training/rot_disk_rotated_minus90.pbz2', 'w') as f: 
            cPickle.dump(data_rotated_minus, f)
            
            
            
def make_ODA_data(height, width, foldernames, dir, dir_result, dt = 10**(-3), dt_data = 10**(-9), data_flipped = True):
        '''This function creates tensors containing the events for all sequences in the ODA dataset (csv files). 
        
        Parameters:
            height (int): height of the image
            width (int): width of the image
            foldernames (list): list of the names of the folders containing the csv files to be converted
            dir (str): directory containing the aedat files
            dir_results (str): directory in which tensors shall be saved
            dt (float): simulation time step
            dt_data (float): times step of the data
        '''
        for folder in foldernames:
            print("\t - Creating sequence {}".format(folder))
            dir_sequence = dir + '/{}/dvs.csv'.format(folder)
            dir_name = dir_result + '/ODA_dataset{}.pbz2'.format(folder)
            data = get_csv_input(height, width, dir_sequence, dir_name, dt = dt, dt_data = dt_data, data_flipped=data_flipped, start = 0, finish = 20000)

if __name__ == '__main__':
    
    #Specify which parameters to use 
    par = configs_rot_disk.configs()

    #Make rotating disk tensor
    print("\n Making rotating disk tensors \n")
    make_rotating_disk_data(par.height, par.width,'data/rot_disk/IMU_rotDisk/events.csv', 'data_tensors/rot_disk/')
    
    #Specify which parameters to use 
    par = configs_ODA.configs()
    
    #Specify which sequences to convert to tesnsors
    folder_names = [3, 10, 345]

    #Make ODA tensors
    print()
    print("\n Making ODA data \n")
    make_ODA_data(par.height, par.width, folder_names, 'data/ODA_Dataset/dataset', 'data_tensors/ODA_dataset')










