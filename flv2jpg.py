import cv2
import os


def flv2jpg(videopath, mytimeF=1, save_flag = False):
    """
    将flv格式视频每隔mytimeF个帧保存一张图片
    :param videopath: 存储flv视频的文件夹
    :param mytimeF: 间隔帧数
    :return:
    """
    save_dir = os.path.join(os.path.dirname(videopath), "images")
    os.makedirs(save_dir, exist_ok=True)
    num = 1  # 保存图片计数
    print(videopath)
    vc = cv2.VideoCapture(videopath)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    timeF = mytimeF  # 每隔mytimeF帧
    count = 1  # 帧计数
    while rval:
        rval, frame = vc.read()
        if count % timeF == 0 and frame is not None:
            print("current frame count:", count)
            print("num:", num)

            if save_flag:
                save_path = os.path.join(save_dir, str(num) + "_" + videopath.split("/")[-1] + ".jpg")
                cv2.imwrite(save_path, frame)
                print(save_path)
                num += 1
        count += 1
    vc.release()
    return save_dir


if __name__ == "__main__":
    test_video = "./data/lunchong3.flv"
    flv2jpg(test_video,save_flag=True)
