from typing import Dict

import torch
import numpy as np
from scipy import ndimage
from monai.transforms import RandGaussianNoise, RandGaussianSmooth, RandScaleIntensity, RandShiftIntensity


class RandomRotate():

    def _transform(self, image, angle):
        return ndimage.rotate(image, angle, order=0, reshape=False)
        
    def __call__(self, sample):
        res = {}
        angle = np.random.randint(-20, 20)
        for key, value in sample.items():
            res[key] = self._transform(value, angle)
        return res


class ToTensor():

    def _transform(self, image, is_label= False):
        
        if is_label:
            return torch.from_numpy(image).long()
        else:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
            return torch.from_numpy(image)
    
    def __call__(self, sample):

        res = {}
        for key, value in sample.items():
            if key == "label":
                res[key] = self._transform(value, is_label=True)
            else:
                res[key] = self._transform(value, is_label=False)
        
        return res

class RandomNoiseClass():

    def _transform(self, image, is_label: bool = False):  

        if is_label:
            return image
        else:
            add_noise = RandGaussianNoise(prob=1.0)
            return add_noise(image)
   
    def __call__(self, sample):

        res = {}
        # print(sample)
        for key, value in sample.items():
            if key == "label":
                res[key] = self._transform(value, is_label=True)
            else:
                res[key] = self._transform(value, is_label=False)  
      
        return res

class RandomSmoothingClass():

    def _transform(self, image, is_label: bool = False):  

        if is_label:
            return image
        else:
            smooth = RandGaussianSmooth(prob=0.5)
            return smooth(image)
   
    def __call__(self, sample):

        res = {}
        # print(sample)
        for key, value in sample.items():
            if key == "label":
                res[key] = self._transform(value, is_label=True)
            else:
                res[key] = self._transform(value, is_label=False)
       
      
        return res

class RandomIntensityShiftClass():

    def _transform(self, image, is_label: bool = False):  

        if is_label:
            return image
        else:
            shift_intensity = RandShiftIntensity(offsets=(5,5), prob=0.3)
            return shift_intensity(image)
   
    def __call__(self, sample):

        res = {}
        # print(sample)
        for key, value in sample.items():
            if key == "label":
                res[key] = self._transform(value, is_label=True)
            else:
                res[key] = self._transform(value, is_label=False)
      
        return res
    
class RandomIntensityScaleClass():

    def _transform(self, image, is_label: bool = False):  

        if is_label:
            return image
        else:
            scale_intensity = RandScaleIntensity(factors=(5,5), prob=0.3)
            return scale_intensity(image)
   
    def __call__(self, sample):

        res = {}
        # print(sample)
        for key, value in sample.items():
            if key == "label":
                res[key] = self._transform(value, is_label=True)
            else:
                res[key] = self._transform(value, is_label=False)
       
        return res
        

class WeightCrop():

    def __init__(self, 
                 output_size: tuple[int, int, int],
                 dataset = "isles-24"
                ):

        self._output_size = [int(val) for val in output_size]
        self.dataset = dataset

    def _transform(self, image:Dict):
        
        dwi, adc, label = image["dwi"], image["adc"], image["label"]

        # pad the images if needed
        if label.shape[0] <= self._output_size[0] or label.shape[1] <= self._output_size[1] or label.shape[2] <= self._output_size[2]:
            pw = max((self._output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self._output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self._output_size[2] - label.shape[2]) // 2 + 3, 0)
            dwi = np.pad(dwi, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            adc = np.pad(adc, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        
        #perform the cropping
        (w, h, d) = dwi.shape
        if label.sum() > 0:
            # returns indices of all non-zero elements in label
            mask = np.nonzero(label)
            
            num_label_voxel = mask[0].shape[0] 

            # gets patch center i.e. a random voxel from the list becomes the patch center
            center_index = np.random.randint(0, num_label_voxel)
            center_x, center_y, center_z = mask[0][center_index], mask[1][center_index], mask[2][center_index]
            
            # the center is shifted by a random amount of voxels 
            # the assertion I added confirmed that there are some iterations which result in a patch without any voxels.
            w1 = np.random.randint(-10, 10)+self._output_size[0]//2
            h1 = np.random.randint(-10, 10)+self._output_size[1]//2
            d1 = np.random.randint(-10, 10)+self._output_size[2]//2
            lefttop_x, lefttop_y, lefttop_z = center_x-w1, center_y-h1, center_z-d1
            minx = max(lefttop_x, 0)
            miny = max(lefttop_y, 0)
            minz = max(lefttop_z, 0)
            maxx = minx +  self._output_size[0]
            maxy = miny +  self._output_size[1]
            maxz = minz +  self._output_size[2]
            if maxx>= w or maxy >= h or maxz >=d:
                maxx = min(maxx, w-1)
                maxy = min(maxy, h-1)
                maxz = min(maxz, d-1)
                minx = maxx - self._output_size[0]
                miny = maxy - self._output_size[1]
                minz = maxz - self._output_size[2]

            label = label[minx:maxx, miny:maxy, minz:maxz]
            dwi = dwi[minx:maxx, miny:maxy, minz:maxz]
            adc = adc[minx:maxx, miny:maxy, minz:maxz]
                
            assert(list(label.shape) == list(self._output_size))
            # assert(label.sum() > 0)
        else:
            w1 = np.random.randint(0, w - self._output_size[0])
            h1 = np.random.randint(0, h - self._output_size[1])
            d1 = np.random.randint(0, d - self._output_size[2])

            label = label[w1:w1 + self._output_size[0], h1:h1 + self._output_size[1], d1:d1 + self._output_size[2]]
            dwi = dwi[w1:w1 + self._output_size[0], h1:h1 + self._output_size[1], d1:d1 + self._output_size[2]]
            adc = adc[w1:w1 + self._output_size[0], h1:h1 + self._output_size[1], d1:d1 + self._output_size[2]]

            assert(list(label.shape) == list(self._output_size))
            # assert(label.sum() > 0)

        return {'dwi':dwi, 'adc':adc, 'label': label}
    
    
    def __call__(self, 
                 sample: Dict
                ):

        res = {}
        to_transform = {}
        # for key, value in sample.items():
        #     to_transform[key] = value
        to_transform = self._transform(sample)

        # for key, value in to_transform.items():
        #     res[key] = value

        return to_transform



class RandomFlipRotate():
    
    def _transform(self,image):
        res = {}
        k = np.random.randint(0, 4)
        for key, value in image.items():
            if key == "file_path":
                continue
            res[key] = np.rot90(value, k)
            arr = np.flip(res[key], (0, 1, 2)).copy()
            res[key] = arr
        return res
        
    def __call__(self, sample):
        res = {}
        res = self._transform(sample)
        return res
