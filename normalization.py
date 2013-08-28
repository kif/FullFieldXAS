#!/usr/bin/python
import sift
import numpy
import fabio
import os
import scipy.misc
import pyopencl, pyopencl.array
from sift.utils import calc_size
from sift.opencl import ocl
'''
TODO:
-more proper way to get frames id: id = sum(precedents[:groupe])+index2 (index3 = 0 pour data)
-remove hard-coded paths
-user-specified inputs


'''


class Normalizer(object):
    """
    Normalizes images (subtract dark current and divide by flat-field image, taking into account the exposure time)
    """

    def __init__(self, devicetype="GPU",):
        """
        Constructor of the class
        """

        #TODO: the following are user-defined
        self.verbose = True
        self.frames_folder = "/mnt/data/ID21-FullFieldXanes/CdCO3_1_3f"
        self.corrected_folder = "testframes"
        self.aligned_folder = "aligned"
        self.darkprefix = 'CdCO3_1_3f_dark_'
        self.dataprefix = 'CdCO3_1_3f_data_'
        self.flatprefix = 'CdCO3_1_3f_ref_'
        self.suffix = ".edf"

        self.devicetype = devicetype
        kernel_path = "openCL/correction.cl"
        kernel_src = open(kernel_path).read()
        self.ctx = ocl.create_context(devicetype=devicetype)
#        self.device
#        self.ctx = pyopencl.Context(devices=[pyopencl.get_platforms()[self.device[0]].get_devices()[self.device[1]]])
#        print self.ctx.devices[0]
        self.queue = pyopencl.CommandQueue(self.ctx)
        self.program = pyopencl.Program(self.ctx, kernel_src).build() #.build('-D WORKGROUP_SIZE=%s' % wg_size)
        self.wg = (1, 1)
        self.dark_data = None
        self.dark_ref = None

    def normalize(self, raw, flat1, flat2):
        '''
        Normalizes the image with OpenCL
        NOTA: images are passed as numpy.array, so the read (edf/hdf5) is done before calling this function
        '''

        output_height, output_width = raw.shape
        shape = calc_size((output_width, output_height), self.wg)
        gpu_raw = pyopencl.array.to_device(self.queue, raw)
        gpu_dark3 = pyopencl.array.to_device(self.queue, self.dark_data)
        gpu_dark6 = pyopencl.array.to_device(self.queue, self.dark_ref)
        gpu_flat1 = pyopencl.array.to_device(self.queue, flat1)
        gpu_flat2 = pyopencl.array.to_device(self.queue, flat2)
        gpu_output = pyopencl.array.empty(self.queue, (output_height, output_width), dtype=numpy.float32, order="C")
        output_height, output_width = numpy.int32((output_height, output_width))

        k1 = self.program.correction(self.queue, shape, self.wg,
                gpu_raw.data, gpu_dark3.data, gpu_dark6.data, gpu_flat1.data, gpu_flat2.data, gpu_output.data,
                output_width, output_height)
        res = gpu_output.get()

        return res



    def normalizeFolderEDF(self):
        '''
        Performs correction on all EDF images in the folder
        '''

        listFlats = []
        listDarks = []
        listData = []
        dirname = self.frames_folder
        for oneFile in os.listdir(dirname):
            if oneFile.startswith(self.flatprefix) and oneFile.endswith(self.suffix):
                oneCompleteFile = os.path.abspath(os.path.join(dirname, oneFile))
                listFlats.append(oneCompleteFile)

            if oneFile.startswith(self.darkprefix) and oneFile.endswith(self.suffix):
                oneCompleteFile = os.path.abspath(os.path.join(dirname, oneFile))
                listDarks.append(oneCompleteFile)

            if oneFile.startswith(self.dataprefix) and oneFile.endswith(self.suffix):
                oneCompleteFile = os.path.abspath(os.path.join(dirname, oneFile))
                listData.append(oneCompleteFile)

        #makes an associative array from listDarks
        arrayDarks = {}
        for dark in listDarks:
            ex = fabio.open(dark).header["exposure_time"]
            arrayDarks[ex] = dark

        #performs correction on each data frame
        for data_fullname in listData:
            data = os.path.basename(data_fullname)
            if self.verbose: print("processing frame %s..." % data)
            #get the ID of the data
            data_id = data[len(self.dataprefix):-len(self.suffix)]
            #get the corresponding flats
            ex1, ex2 = -1.0, -1.0
            for ref_fullname in listFlats:
                ref = os.path.basename(ref_fullname)
                ref_id = ref[len(self.flatprefix):-len(self.suffix)]
                if self.dark_data is None:
                    exp_data = fabio.openheader(ref_fullname).header["exposure_time"]
                    dark_data = arrayDarks[exp_data]
                    self.dark_data = fabio.open(dark_data).data

                if ref_id == data_id:
                    flat1 = ref_fullname[:]
                    #get the exposure time of flat 1
                    exp_ref = fabio.openheader(ref_fullname).header["exposure_time"]
                    #get the corresponding Dark ("dark data")
                    if self.dark_ref is None:
                        dark_ref = arrayDarks[exp_ref]
                        self.dark_ref = fabio.open(dark_ref).data

                if int(ref_id[-4:]) == (int(data_id[-4:]) + 1): #FIXME: hard-coded !
                    flat2 = ref_fullname[:]
                    #get the exposure time of flat 2
                    ex2 = fabio.open(ref_fullname).header["exposure_time"]
                    #get the corresponding Dark ("dark ref")
            if ex1 == -1.0 or ex2 == -1.0:
                print("Error: failed to get flats for %s" % (data))
                flat2 = flat1
                ex2 = ex1

            cor = self.normalize(fabio.open(data_fullname).data, fabio.open(flat1).data, fabio.open(flat2).data)
            fabio.edfimage.edfimage(data=cor).write(os.path.join(self.corrected_folder , "frame_%s.edf" % data_id))
#            scipy.misc.imsave(os.path.join(self.corrected_folder , "frame_%s.png" % data_id), cor)
            if self.verbose: print("Done")



    def matchingCorrection(self, matching):
        '''
        Given the matching between two images, determine the transformation to align the image on the other image
        '''
        #computing keypoints matching
        N = matching.shape[0]
        #solving normals equations for least square fit
        X = numpy.zeros((2 * N, 6))
        X[::2, 2:] = 1, 0, 0, 0
        X[::2, 0] = matching.x[:, 0]
        X[::2, 1] = matching.y[:, 0]
        X[1::2, 0:3] = 0, 0, 0
        X[1::2, 3] = matching.x[:, 0]
        X[1::2, 4] = matching.y[:, 0]
        X[1::2, 5] = 1
        y = numpy.zeros((2 * N, 1))
        y[::2, 0] = matching.x[:, 1]
        y[1::2, 0] = matching.y[:, 1]
        A = numpy.dot(X.transpose(), X)
        sol = numpy.dot(numpy.linalg.inv(A), numpy.dot(X.transpose(), y))
#        sol = numpy.dot(numpy.linalg.pinv(X),y)
        return sol



    def imageReshape(self, img):
        '''
        Reshape the image to get a bigger image with the input image in the center
        '''
        image_height, image_width = img.shape
        output_height, output_width = image_height * int(sqrt(2)), image_width * int(sqrt(2))
        image2 = numpy.zeros((output_height, output_width), dtype=numpy.float32)
        d1 = (output_width - image_width) / 2
        d0 = (output_height - image_height) / 2
        image2[d0:-d0, d1:-d1] = numpy.copy(img)
        image = image2
        image_height, image_width = output_height, output_width
        return image, image_height, image_width



    def siftAlign(self):
        '''
        Call SIFT to align images
        Assume that all the images have the same dimensions !
        '''

        mp = sift.MatchPlan(devicetype=self.devicetype)

        #TODO: place the following in a separate routine (in SIFT module ?)
        kernel_path = "openCL/transform.cl"
        kernel_src = open(kernel_path).read()
        program = pyopencl.Program(self.ctx, kernel_src).build() #.build('-D WORKGROUP_SIZE=%s' % wg_size)
        wg = 8, 8 #FIXME: hard-coded



        i = 0
        for img in os.listdir(self.save_folder):
            if i == 0: #compute SIFT keypoints on the first image
                i = 1
                plan = sift.SiftPlan(template=img, devicetype=self.devicetype)
                kp_first = plan.keypoints(img)
            else:
                kp = plan.keypoints(img)
                m = mp.match(kp_first, kp)
                sol = self.matchingCorrection(m)

                correction_matrix = numpy.zeros((2, 2), dtype=numpy.float32)
                correction_matrix[0] = sol[0:2, 0]
                correction_matrix[1] = sol[3:5, 0]
                matrix_for_gpu = correction_matrix.reshape(4, 1) #for float4 struct
                offset_value[0] = sol[2, 0]
                offset_value[1] = sol[5, 0]

                img, image_height, image_width = self.imageReshape(img)
                gpu_image = pyopencl.array.to_device(self.queue, img)
                gpu_output = pyopencl.array.empty(self.queue, (image_height, image_width), dtype=numpy.float32, order="C")
                gpu_matrix = pyopencl.array.to_device(self.queue, matrix_for_gpu)
                gpu_offset = pyopencl.array.to_device(self.queue, offset_value)
                image_height, image_width = numpy.int32((image_height, image_width))
                output_height, output_width = image_height, image_width

                if i == 1: shape = calc_size((output_width, output_height), wg)
                k1 = program.transform(self.queue, shape, wg,
                        gpu_image.data, gpu_output.data, gpu_matrix.data, gpu_offset.data,
                        image_width, image_height, output_width, output_height, fill_value, mode)
                res = gpu_output.get()

#                scipy.misc.imsave(self.aligned_folder + "/frame" + str(i) +".png", res)
                i += 1




if __name__ == '__main__':

    Normalizer().normalizeFolderEDF()








