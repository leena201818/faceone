# faceone

本项目参照并利用facenet模型构建，实现人脸探测、人脸特征库构建、人脸检索、人脸比对等功能。

python detect_and_genfea.py mode model [options]

【人脸探测】

        '''
        DETECT ./facenet/pretrained_models/20180402-114759 --detect_dir ./testdata/raw --detect_out_dir ./testdata/output/test3
        '''
        
        其中 model实际上是不需要的
        
        --detect_dir 待探测人脸的图片所在目录，注意，每个人一个目录
        
        --detect_out_dir 人脸结果目录，一个人对应一个目录，一张图片可以探测多个人脸
        

【人脸特征库构建】 注意输出目录face_dir中face_fea.h5用来放置特征，每个源图片对应一个子目录，放置提取出来的人脸

        '''
        GENERATE ./facenet/pretrained_models/20180402-114759 --image_files ./testdata/raw/unknown/zark6.jpg ./testdata/raw/unknown/zark5.jpg --image_dir ./testdata/raw/unknown --faces_dir ./testdata/output/test2
        '''
        
        其中 model是facenet一个预训练库，要求image_size = 160
        
        --image_files 可以列出多个图片文件
        
        --image_dir   待提取图片目录，不能包含子目录
        
        --faces_dir   人脸图片及特征文件所在目录

【人脸检索】库中是否含有给定图片中的人

        '''
        SEARCH ./facenet/pretrained_models/20180402-114759 --faces_dir ./testdata/output/test --search_image ./testdata/raw/unknown/zark7.jpg
        '''
        
        其中
        
        --faces_dir   人脸图片及特征文件所在目录，是本程序生成的人脸特征库目录，里面必须包含face_fea.h5文件和相应的子目录人脸
        
        --search_image 待搜索的图片,本程序仅仅探测一个人脸

【人脸比对】可以找出两张图片中所有相似的人脸对，给定距离阈值theadhold = 1.11。

        '''
        VERIFY ./facenet/pretrained_models/20180402-114759 --verify_images ./testdata/raw/unknown/z2.jpg ./testdata/raw/unknown/z3.jpg
        '''
        
        --verify_images 待匹配人脸的两张图片，均可含有多张人脸
        
