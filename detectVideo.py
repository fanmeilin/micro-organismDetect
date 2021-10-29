from flv2jpg import flv2jpg
import cv2
import os,glob
from detectInfo import Detect_img
def makeVideo(Video_img_list,fps,size,save_dir):
    """
    func:由切分好的视频帧图像得到检测后的视频
    input:
    return:
    """
    # video = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    video = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img in Video_img_list:
        img = cv2.resize(img, size)  # 将图片转换为1280*720
        video.write(img)

def detectVideo(test_video_path,detect_img,fps=24,size=(640,480),sift_thres = 0.5):
    img_root = flv2jpg(test_video_path,save_flag=True) #读取路径
    save_dir_anno = os.path.join(os.path.dirname(test_video_path),"annotation/")
    #获取批量数据的检测信息
    Video_img_list = detect_img.get_info(source=img_root, save_img=True, save_dir=save_dir_anno,sift_thres=sift_thres)
    save_dir = os.path.join(os.path.dirname(test_video_path),"VideoDetect.mp4")
    makeVideo(Video_img_list, fps, size, save_dir)


if __name__ == "__main__":
    # detect_img = Detect_img(weights="./weights/best.pt")
    test_video_path = "./data/lunchong3.flv"
    # detectVideo(test_video_path, detect_img, fps=120, size=(640, 480), sift_thres=0)


    save_dir_anno = os.path.join(os.path.dirname(test_video_path), "annotation/")
    Video_img_listPath = glob.glob(save_dir_anno+"*.jpg")
    Video_img_list = [cv2.imread(path) for path in Video_img_listPath ]
    save_dir = os.path.join(os.path.dirname(test_video_path), "VideoDetect30.mp4")
    makeVideo(Video_img_list, fps=30, size=(640,480), save_dir=save_dir)
