"""探测人脸，并生成相应的人脸特征，保存到文件"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet.facenet
from facenet.align import detect_face
import h5py
import random

import matplotlib.pyplot as plt

'''
    1、从图片文件列表中提取人脸图片（一张图片一个人脸or none）
    2、生成图片的特征向量（加载预训练好的facenet网络）
    3、保存人脸和特征到对应文件
    4、计算任意两张人脸的L2距离
'''

class detect_face:
    def __init__(self,model,image_size = 160, margin = 44,gpu_memory_fraction = 1.0):
        self.image_size = image_size            #facenet模型要求的人脸大小，是模型训练时确定的
        self.margin = margin
        self.pnet,self.rnet,self.onet = self.create_mtcnn(gpu_memory_fraction)

        #初始化facenet模型
        with tf.Graph().as_default():
            with tf.Session().as_default():
                # 把模型加载到当前默认graph的默认session中
                facenet.facenet.load_model(model)
                self.graph = tf.get_default_graph()
                self.sess = tf.get_default_session()

    def create_mtcnn(self,gpu_memory_fraction):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  # GPU内存使用比例上限
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = facenet.align.detect_face.create_mtcnn(sess, None)
                return pnet,rnet,onet

    #运行facenet模型推断，生成特征向量,images是白化处理后的一批数据（batch,h,w,c）
    def extract_fea(self,images):
        # 模型输入图片tensor和输出特征tensor
        images_placeholder = self.graph.get_tensor_by_name("input:0")
        embeddings = self.graph.get_tensor_by_name("embeddings:0")
        # 训练or not
        phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")

        # 运行前向传播过程获取特征向量
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb

    #探测一张图片，提取人脸，返回人脸bounding_box、加上边框后的bb和白化处理后的ndarray
    def detect_faces_bb(self,image_file):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        img = misc.imread(os.path.expanduser(image_file), mode='RGB')  # PIL返回ndarray
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = facenet.align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                                  factor)
        img_whitened = []
        img_aligned = []

        # 人脸剪切时增加margin边框
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])  # np.squeeze删去长度为1的axis
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]  # RGB图片剪切(h,w,c)
            aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')  # 剪切的人脸resize统一尺寸

            # 对人脸进行白化处理，并生成特征向量
            prewhitened = facenet.facenet.prewhiten(aligned) # 白化图片,求所有axis上均值、方差，y = (x-u)/std,目标：成零均值和单位方差
            img_aligned.append(aligned)
            img_whitened.append(prewhitened)

        images_aligned = np.stack(img_aligned)
        images_whitened = np.stack(img_whitened)

        return bounding_boxes,images_aligned,images_whitened

    #提取图片列表，提取人脸并生成特征向量,保存到指定目录和hdf5文件
    def load_detect_extract_fea(self,image_paths_list,output_path,output_fea_h5file):
        output_path = os.path.expanduser(output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tmp_image_paths = copy.copy(image_paths_list)
        img_list = []           #白化处理后的人脸
        face_list = []          #保存的人脸文件
        for image in tmp_image_paths:
            img = misc.imread(os.path.expanduser(image), mode='RGB')  # PIL返回ndarray

            bounding_boxes,images_aligned,images_whitened = self.detect_faces_bb(image)
            if len(bounding_boxes) < 1:  # 返回的事一个list，五个元素，x,y,x,y,p后者为人脸概率
                image_paths_list.remove(image)
                print("%s找不到人脸，移除 "%(image))
                continue
            print("%-32s找到:%2d张人脸"%(os.path.basename(image),len(bounding_boxes)) )

            #创建以该文件为名称的子目录
            image_filename = os.path.basename(image)
            face_subdir = os.path.join(output_path,image_filename)
            if not os.path.exists(face_subdir):
                os.mkdir(face_subdir)

            img_list.append(images_whitened)

            for i in range(len(bounding_boxes)):
                aligned = images_aligned[i]

                #保存到对齐的人脸到指定文件夹
                aligned_face_name = "{}_{}.png".format(image_filename,i)
                aligned_file = os.path.join(face_subdir,aligned_face_name)
                misc.imsave(aligned_file,aligned)

                face_list.append(aligned_face_name)       #仅通过提取出来的face文件名子来区分，不包含子目录

        images = np.vstack(img_list)
        fea = self.extract_fea(images)
        face_id = np.asarray(face_list)

        dt = h5py.special_dtype(vlen=str)
        with h5py.File(output_fea_h5file, 'w') as f:
            f.create_dataset('face512', data=fea)
            ds = f.create_dataset('faceid', face_id.shape, dtype=dt)
            ds[:] = face_id

        return fea

    #从一个目录中提取所有图片的人脸，并生成特征向量，保存到指定目录和hdf5文件
    def load_detect_extract_fea_fromdir(self, image_dirs, output_path, output_fea_h5file):
        assert os.path.exists(image_dirs)
        image_paths_list = []
        for file in os.listdir(image_dirs):
            image_file = os.path.join(image_dirs,file)
            file_ext = os.path.splitext(image_file)[1]
            if os.path.isfile(image_file) and file_ext in ('.jpg','jpeg','.png','.bmp','gif'):
                image_paths_list.append(image_file)

        self.load_detect_extract_fea(image_paths_list,output_path,output_fea_h5file)

    #输入一个图片，从指定的人脸特征目录中获取最相似的人脸
    def search_face(self,image_file,faces_dir,nb_return = 6):
        assert os.path.exists(faces_dir)
        face_fea_file = os.path.join(faces_dir,'face_fea.h5')
        assert os.path.exists(face_fea_file)

        #提取人脸
        bounding_boxes, images_aligned, images_whitened = self.detect_faces_bb(image_file)

        if len(bounding_boxes) < 1:
            print("%s找不到人脸，返回 "%(image_file))
            return
        #生成特征
        face_fea = np.squeeze(self.extract_fea(images_whitened)[0])

        with h5py.File(face_fea_file,'r') as f:
            face512 = f['face512'][:]
            faceid = f['faceid'][:]
            distance = np.sqrt(np.sum( np.square(np.subtract(face512,face_fea)),axis=1))

            match_index = np.argsort(distance)[0:nb_return]
            match_dist = distance[match_index]
            match_faceid = faceid[match_index]

            match_face = zip(match_faceid,match_dist)

            # 仅仅显示少量图片
            row, col = 3, 3
            plt.figure(row * col, figsize=(12, 10))
            n = 1
            for mface,mdis in match_face:
                mface_dir = mface[0:mface.rindex('_')]
                face_path = os.path.join(os.path.join(faces_dir,mface_dir),mface)
                photo = misc.imread(face_path)
                caption = "%s(%.2f)"%(mface,mdis)
                if n <= row * col:
                    plt.subplot(row, col, n)
                    plt.imshow(photo)
                    plt.title(caption)
                else:
                    break
                n += 1

            plt.show()

            return match_faceid,match_dist

    #比较两张图片中的人，每张图片中人脸不能超过1个
    def verify_face(self,image1_file,image2_file,threadhold = 1.0):
        assert os.path.exists(image1_file)
        assert os.path.exists(image2_file)

        # 提取人脸
        bounding_boxes1, images_aligned1, images_whitened1 = self.detect_faces_bb(image1_file)
        bounding_boxes2, images_aligned2, images_whitened2 = self.detect_faces_bb(image2_file)
        if len(bounding_boxes1) < 1:
            print('%s没有检测到人脸'%(image1_file))
        if len(bounding_boxes2) < 1:
            print('%s没有检测到人脸' % (image2_file))

        # 获得特征向量
        # face_fea1 = np.squeeze(self.extract_fea(images_whitened1)[0])
        # face_fea2 = np.squeeze(self.extract_fea(images_whitened2)[0])
        # distance = np.sqrt(np.sum(np.square(np.subtract(face_fea1,face_fea2))))

        face_fea1 = self.extract_fea(images_whitened1)
        face_fea2 = self.extract_fea(images_whitened2)

        distance = np.zeros((face_fea1.shape[0],face_fea2.shape[0]))

        # 打印距离矩阵，图片之间两两距离
        print('距离矩阵')
        print('    ', end='')  # end=''不换行
        for i in range(face_fea2.shape[0]):
            print('    %1d     ' % i, end='')
        print('')
        for i in range(face_fea1.shape[0]):
            print('%1d  ' % i, end='')
            for j in range(face_fea2.shape[0]):
                dist = np.sqrt(np.sum(np.square(np.subtract(face_fea1[i, :],face_fea2[j, :]))))
                distance[i,j] = dist
                print('  %1.4f  ' % dist, end='')
            print('')

        match_index = np.argwhere( distance <= threadhold )

        row = match_index.shape[0]
        col = 2
        plt.figure(row*col, figsize=(8,6))
        n = 1
        for i in range(row):
            for j in np.arange(0,col,2):
                if n <= row * col:
                    plt.subplot(row,col,n)
                    plt.imshow(images_aligned1[match_index[i][0]])

                    plt.subplot(row,col, n+1)
                    plt.imshow(images_aligned2[match_index[i][1]])
                else:
                    break
                n = n + 2

        plt.show()

        if row > 0:
            print('%s和%s中含有%d对同一个人，在threadhold=%f时'%(image1_file,image2_file,row,threadhold))
            return True
        else:
            print('%s和%s中未含有同一个人，在threadhold=%f时' % (image1_file, image2_file, threadhold))
            return False

    def detect(self,input_dir,output_dir,random_order = False,detect_multiple_faces = True):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = facenet.facenet.get_dataset(input_dir)

        print('Creating networks and loading parameters')

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            if random_order:
                random.shuffle(dataset)
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if random_order:
                        random.shuffle(cls.image_paths)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print(image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.facenet.to_rgb(img)
                            img = img[:, :, 0:3]

                            bounding_boxes, _,_ = self.detect_faces_bb(image_path)
                            nrof_faces = bounding_boxes.shape[0]
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                det_arr = []
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    if detect_multiple_faces:
                                        for i in range(nrof_faces):
                                            det_arr.append(np.squeeze(det[i]))
                                    else:
                                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                        img_center = img_size / 2
                                        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                             (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                        index = np.argmax(
                                            bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                        det_arr.append(det[index, :])
                                else:
                                    det_arr.append(np.squeeze(det))

                                for i, det in enumerate(det_arr):
                                    det = np.squeeze(det)
                                    bb = np.zeros(4, dtype=np.int32)
                                    bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                                    bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                                    bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                                    bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                    scaled = misc.imresize(cropped, (self.image_size, self.image_size),
                                                           interp='bilinear')
                                    nrof_successfully_aligned += 1
                                    filename_base, file_extension = os.path.splitext(output_filename)
                                    if detect_multiple_faces:
                                        output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                    else:
                                        output_filename_n = "{}{}".format(filename_base, file_extension)
                                    misc.imsave(output_filename_n, scaled)
                                    text_file.write(
                                        '%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='提取指定图片中人脸，并生成特征向量到hdf5文件')

    parser.add_argument('mode', type=str, choices=['DETECT','GENERATE', 'SEARCH','VERIFY'],
                        help='GENERATE:创建人脸特征，生成相应目录及特征文件;SEARCH:从生成库中搜索特定人；VERIFY:验证两张图片中人脸', default='VERIFY')
    parser.add_argument('model', type=str,
        help='包含.meta和.ckpt文件的目录 or 一个模型定义文件(.pb)')

    parser.add_argument('--detect_dir', type=str, help='[DETECT],待提取人脸的图片目录，必须子目录')
    parser.add_argument('--detect_out_dir', type=str, help='[DETECT],提取取人脸结果目录')

    parser.add_argument('--image_files', type=str, nargs='+', help='[GENERATE],待提取人脸的图片，可以添加多个')   #nargs='+'表明多个
    parser.add_argument('--image_dir', type=str,help='[GENERATE],待提取人脸的图片目录，不要包含子目录')
    parser.add_argument('--faces_dir', type=str,help='[GENERATE,SEARCH],人脸图片及特征文件所在目录')
    parser.add_argument('--search_image', type=str,help='[SEARCH],待搜索图片')
    parser.add_argument('--verify_images', type=str, help='[VERIFY],待对比图片，需要两张',nargs=2)

    parser.add_argument('--image_size', type=int,
        help='facenet模型输入图片尺寸，应该model目录中的训练模型相匹配', default=160)
    parser.add_argument('--margin', type=int,
        help='人脸提取时周边边框尺寸', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='本进程占用GPU内存的比例上限', default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    model = args.model
    image_size = args.image_size
    margin = args.margin
    gpu_memory_fraction = args.gpu_memory_fraction

    df = detect_face(model,image_size,margin,gpu_memory_fraction)

    if args.mode == 'DETECT':
        '''
        DETECT ./facenet/pretrained_models/20180402-114759 --detect_dir ./testdata/raw --detect_out_dir ./testdata/output/test3  
        '''
        input_dir = args.detect_dir
        output_dir = args.detect_out_dir
        df.detect(input_dir,output_dir)

    if args.mode == 'GENERATE':
        '''
        GENERATE ./facenet/pretrained_models/20180402-114759 --image_files ./testdata/raw/unknown/zark6.jpg ./testdata/raw/unknown/zark5.jpg --image_dir ./testdata/raw/unknown --faces_dir ./testdata/output/test2  
        '''

        output_dir_face = './testdata/output/test'
        output_dir_face = args.faces_dir
        output_fea_h5file = os.path.join(output_dir_face, 'face_fea.h5')

        image_paths = args.image_files
        image_dir = args.image_dir

        if image_paths is not None:
            df.load_detect_extract_fea(image_paths, output_dir_face, output_fea_h5file)

        if image_dir is not None:
            df.load_detect_extract_fea_fromdir(image_dir, output_dir_face, output_fea_h5file)
            # 读取生成的人脸文件、人脸特征
            with h5py.File(output_fea_h5file, 'r') as f:
                face512 = f['face512'][:]
                faceid = f['faceid'][:]
                print(face512.shape)

    if args.mode == 'SEARCH':
        '''
        SEARCH ./facenet/pretrained_models/20180402-114759 --faces_dir ./testdata/output/test --search_image ./testdata/raw/unknown/zark7.jpg
        '''
        # to_search_img = '/home/mika/PycharmProjects/faceone/testdata/raw/unknown/zark7.jpg'
        # output_dir_face = './testdata/output/test'

        to_search_img = args.search_image
        output_dir_face = args.faces_dir
        df.search_face(to_search_img, faces_dir=output_dir_face, nb_return=9)

    if args.mode == 'VERIFY':
        '''
        VERIFY ./facenet/pretrained_models/20180402-114759 --verify_images ./testdata/raw/unknown/z2.jpg ./testdata/raw/unknown/z3.jpg
        '''
        #比较两张图片中的人脸
        # face1 = '/home/mika/PycharmProjects/faceone/testdata/raw/unknown/z2.jpg'
        # face2 = '/home/mika/PycharmProjects/faceone/testdata/raw/unknown/z3.jpg'

        verify_images = args.verify_images
        face1 = verify_images[0]
        face2 = verify_images[1]

        df.verify_face(face1,face2,threadhold=1.11)

