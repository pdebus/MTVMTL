import numpy as np
import nrrd

name = "gk2-rcc-mask"
#name = "dt-helix"
data, header =nrrd.read(name + ".nhdr")

roi = True

dims = data.shape
value_dim = dims[0]
num_pic = np.prod(dims[1:])

print "Dims, Values, Pixelnumber"
print dims
print value_dim
print num_pic

xyz_dims = np.array(dims[1:])

roi_size = 16
roi_middle = xyz_dims / 2
roi_start = roi_middle - roi_size
roi_end = roi_middle + roi_size

print "ROI:"
print roi_start
print roi_end

if roi:
    roi = data[:,roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
    num_pic = np.prod(roi.shape[1:])
    print roi.shape
    print num_pic
    roiflat = np.reshape(np.transpose(roi, (3,2,1,0)),(num_pic, value_dim))
    spdlistarray = np.zeros((num_pic, 9))    
    maskarray = roiflat[:,0]
    for i in xrange(num_pic):
        D = roiflat[i,:]
        spdlistarray[i,:] = np.array([D[1], D[2], D[3], D[2], D[4], D[5], D[3], D[5], D[6]])

else:
    xyzflatarray = np.reshape(np.transpose(data, (3,2,1,0)),(num_pic, value_dim))
    spdlistarray = np.zeros((num_pic, 9))
    maskarray = xyzflatarray[:,0]
    for i in xrange(num_pic):
        D = xyzflatarray[i,:]
        spdlistarray[i,:] = np.array([D[1], D[2], D[3], D[2], D[4], D[5], D[3], D[5], D[6]])

np.savetxt(name + ".csv", spdlistarray, fmt='%g', delimiter=',')
np.savetxt(name + "_inp.csv", maskarray, fmt='%g', delimiter=',')