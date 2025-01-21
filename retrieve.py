# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:07:21 2024

@author: Luke Brown
"""

#import required modules
import joblib
import rasterio
import numpy as np

#specify input subset, reflectance scale factor, and whether to compute LAI or FAPAR
input_subset='example_data/S2A_MSIL2A_20180727T185921_N0206_R013_T10TER_20180728T001255.SAFE.tif'
scale_factor=10000
variable='LAI'

#load GPR model
if variable=='LAI':
    gpr=joblib.load('models/lai.pkl')
    upper_lim=10
elif variable=='FAPAR':
    gpr=joblib.load('models/fapar.pkl')
    upper_lim=1

#load image
image=rasterio.open(input_subset)

#read in metadata
metadata=image.meta.copy()
bands=metadata['count']

#read in bands, applying scaling factor
b1=image.read(1)/scale_factor
b2=image.read(2)/scale_factor
b3=image.read(3)/scale_factor
b4=image.read(4)/scale_factor
b5=image.read(5)/scale_factor
b6=image.read(6)/scale_factor
b7=image.read(7)/scale_factor
b8=image.read(8)/scale_factor
b8a=image.read(9)/scale_factor
b9=image.read(10)/scale_factor
b11=image.read(11)/scale_factor
b12=image.read(12)/scale_factor
scl=image.read(13)
vza=image.read(14)
vaa=image.read(15)
sza=image.read(16)
saa=image.read(17)

#calculate cosine of VZA, SZA & RAA
cos_vza=np.cos(np.radians(vza))
cos_sza=np.cos(np.radians(sza))
cos_raa=np.cos(np.radians(vaa-saa))

#get shape of image
shape=np.shape(b1)

#construct GPR input array, flattening bands
inputs=np.array([b1.flatten(),
                 b2.flatten(),
                 b3.flatten(),
                 b4.flatten(),
                 b5.flatten(),
                 b6.flatten(),
                 b7.flatten(),
                 b8.flatten(),
                 b8a.flatten(),
                 b9.flatten(),
                 b11.flatten(),
                 b12.flatten(),
                 cos_sza.flatten(),
                 cos_vza.flatten(),
                 cos_raa.flatten()]).T

#compute GPR predictions
gpr_mean,gpr_std=gpr.predict(inputs,return_std=True)

#reshape GPR predictions
gpr_mean=gpr_mean.reshape(shape)
gpr_std=gpr_std.reshape(shape)

#restrict to minima/maxima
gpr_mean[gpr_mean<0]=0
gpr_mean[gpr_mean>upper_lim]=upper_lim

#mask non-vegetated/non-soil pixels using scene classification
gpr_mean[~np.isin(scl,[4,5])]=np.nan
gpr_std[~np.isin(scl,[4,5])]=np.nan

#prepare output metadata
metadata.update({'count':2})

#write output image
with rasterio.open(input_subset[:-4]+'_'+variable+'.tif','w', **metadata) as output:
    output.write_band(1,gpr_mean)
    output.write_band(2,gpr_std)
