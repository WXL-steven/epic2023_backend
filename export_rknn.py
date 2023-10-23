import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = './Checkpoint/best_onnx.onnx'
RKNN_MODEL = './Checkpoint/davit_t.rknn'


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'resnet50v2\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        raise FileNotFoundError("ONNX model not found.")

    # pre-process config
    print('--> config model')
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.82, 58.82, 58.82]],
        target_platform='rk3588',
        model_pruning=True,  # 启用模型剪枝
    )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        inputs=['input0'],
        input_size_list=[[1, 3, 224, 224]],
    )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(
        do_quantization=False,  # 不启用量化
        # rknn_batch_size=1,
    )
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    print(f"Model exported to {RKNN_MODEL}")

    rknn.release()
