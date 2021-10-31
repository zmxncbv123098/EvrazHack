from YoloTensorRTWrapper import YoLov5TRT
import ctypes
import cv2.cv2 as cv2


ctypes.CDLL("weights/libmyplugins.so")

trt_wrapper = YoLov5TRT(engine_file_path="weights/best_40.engine")


def single_image_predict(im_pth):
    img = cv2.imread(im_pth)

    output, _ = trt_wrapper.infer([img])
    print(output)

    return output

