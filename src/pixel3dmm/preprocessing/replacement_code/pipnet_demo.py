import traceback

import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '../..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *
from mobilenetv3 import mobilenetv3_large


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

if not len(sys.argv) == 3:
    print('Format:')
    print('python lib/demo.py config_file image_file')
    exit(0)


experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

def get_cstm_crop(image, detections):
    #Image.fromarray(image).show()
    image_width = image.shape[1]
    image_height = image.shape[0]
    det_box_scale = 1.42 #2.0#1.42
    det_xmin = detections[2]
    det_ymin = detections[3]
    det_width = detections[4]
    det_height = detections[5]
    if det_width > det_height:
        det_ymin -= (det_width - det_height)//2
        det_height = det_width
    if det_width < det_height:
        det_xmin -= (det_height - det_width)//2
        det_width = det_height

    det_xmax = det_xmin + det_width - 1
    det_ymax = det_ymin + det_height - 1


    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
    det_ymin -= int(det_height * (det_box_scale - 1) / 2)
    det_xmax += int(det_width * (det_box_scale - 1) / 2)
    det_ymax += int(det_height * (det_box_scale - 1) / 2)
    if det_xmin < 0 or det_ymin < 0:
        min_overflow = min(det_xmin, det_ymin)
        det_xmin += -min_overflow
        det_ymin += -min_overflow
    if det_xmax > image_width -1 or det_ymax > image_height - 1:
        max_overflow = max(det_xmax - image_width -1, det_ymax - image_height-1)
        det_xmax -= max_overflow
        det_ymax -= max_overflow

    det_width = det_xmax - det_xmin + 1
    det_height = det_ymax - det_ymin + 1
    det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
    return det_crop
    #Image.fromarray(det_crop).show()
    #exit()

def demo_image(image_dir, pid, cam_dir, net, preprocess, cfg, input_size, net_stride, num_nb, use_gpu, device, flip=False, start_frame=0,
               vertical_crop : bool = False,
               static_crop : bool = False,
               ):
    detector = FaceBoxesDetector('FaceBoxes', '../PIPNet/FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.6
    det_box_scale = 1.2
    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(
        os.path.join('../..', 'PIPNet', 'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

    net.eval()

    #CROP_DIR = '/mnt/rohan/cluster/angmar/sgiebenhain/now_crops_pipnet/'
    #os.makedirs(CROP_DIR, exist_ok=True)


    if start_frame > 0:
        files = [f for f in os.listdir(f'{image_dir}/') if f.endswith('.jpg') or f.endswith('.png') and (((int(f.split('_')[-1].split('.')[0])-start_frame) % 3 )== 0)]
    else:
        files = [f for f in os.listdir(f'{image_dir}/') if f.endswith('.jpg') or f.endswith('.png')]
    files.sort()

    if not vertical_crop:
        all_detections = []
        all_images = []
        #all_normals = []
        succ_files = []
        for file_name in files:
                image = cv2.imread(f'{image_dir}/{file_name}')
                #normals = cv2.imread(f'{image_dir}/../normals/{file_name[:-4]}.png')

                if len(image.shape) < 3 or image.shape[-1] != 3:
                    continue

                image_height, image_width, _ = image.shape
                detections, _ = detector.detect(image, my_thresh, 1)
                dets_filtered = [det for det in detections if det[0] == 'face']
                dets_filtered.sort(key=lambda x: -1 * x[1])
                detections = dets_filtered
                if detections[0][1] < 0.75:
                    raise ValueError("Found face with too low detections confidence as max confidence")
                all_detections.append(detections[0])
                all_images.append(image)
                #all_normals.append(normals)
                succ_files.append(file_name)

        if static_crop:
            det1 = np.mean(np.array([x[2] for x in all_detections]), axis=0)
            det2 = np.mean(np.array([x[3] for x in all_detections]), axis=0)
            det3 = np.mean(np.array([x[4] for x in all_detections]), axis=0)
            det4 = np.mean(np.array([x[5] for x in all_detections]), axis=0)
            det_smoothed = np.stack([det1, det2, det3, det4], axis=0).astype(np.int32)
            all_detections_smoothed = []  # = [[x[0], x[1], x_smoothed[0], x_smoothed[1], x_smoothed[2], x_smoothed[3]] for x, x_smoothed in zip()]
            for i, det in enumerate(all_detections):
                all_detections_smoothed.append(
                    [det[0], det[1], det_smoothed[0], det_smoothed[1], det_smoothed[2], det_smoothed[3]])
            all_detections = all_detections_smoothed
        else:
            if len(all_detections) > 11:
                WINDOW_LENGTH = 11
                det1 = smooth(np.array([x[2] for x in all_detections]), window_len=WINDOW_LENGTH)
                det2 = smooth(np.array([x[3] for x in all_detections]), window_len=WINDOW_LENGTH)
                det3 = smooth(np.array([x[4] for x in all_detections]), window_len=WINDOW_LENGTH)
                det4 = smooth(np.array([x[5] for x in all_detections]), window_len=WINDOW_LENGTH)
                det_smoothed = np.stack([det1, det2,det3,det4], axis=1).astype(np.int32)
                all_detections_smoothed = [] #= [[x[0], x[1], x_smoothed[0], x_smoothed[1], x_smoothed[2], x_smoothed[3]] for x, x_smoothed in zip()]
                for i, det in enumerate(all_detections):
                    all_detections_smoothed.append([det[0], det[1], det_smoothed[i, 0], det_smoothed[i, 1], det_smoothed[i, 2], det_smoothed[i, 3]])
                all_detections = all_detections_smoothed
        # TODO: smooth detections!!!
        for file_name, detection, image in zip(succ_files, all_detections, all_images):

                        img_crop = get_cstm_crop(image, detection)
                        #n_crop = get_cstm_crop(normals, detection)
                        image = img_crop
                        # save cropped image
                        os.makedirs(f'{image_dir}/../cropped/', exist_ok=True)
                        #os.makedirs(f'{image_dir}/../cropped_normals/', exist_ok=True)
                        cv2.imwrite(f'{image_dir}/../cropped/{file_name}', cv2.resize(image, (512, 512)))
                        #cv2.imwrite(f'{image_dir}/../cropped_normals/{file_name[:-4]}.png', cv2.resize(n_crop, (512, 512)))
    else:
        for file_name in files:
            image = cv2.imread(f'{image_dir}/{file_name}')
            if image.shape[0] != image.shape[1]:
                image = image[220:-220, 640:-640, :]
            os.makedirs(f'{image_dir}/../cropped/', exist_ok=True)
            cv2.imwrite(f'{image_dir}/../cropped/{file_name}', cv2.resize(image, (512, 512)))


    lms = []
    image_dir = f'{image_dir}/../cropped/'
    for file_name in files:
                image = cv2.imread(f'{image_dir}/{file_name}')

                if len(image.shape) < 3 or image.shape[-1] != 3:
                    continue
                if flip:
                    image = cv2.transpose(image)

                image_height, image_width, _ = image.shape
                detections, _ = detector.detect(image, my_thresh, 1)
                pred_export = None
                dets_filtered = [det for det in detections if det[0] == 'face']
                dets_filtered.sort(key=lambda x: -1 * x[1])
                detections = dets_filtered

                #print("detections:", detections)
                for i in range(min(1, len(detections))):
                    if detections[i][1] < 0.99:
                        continue
                    det_xmin = detections[i][2]
                    det_ymin = detections[i][3]
                    det_width = detections[i][4]
                    det_height = detections[i][5]
                    det_xmax = det_xmin + det_width - 1
                    det_ymax = det_ymin + det_height - 1


                    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
                    # remove a part of top area for alignment, see paper for details
                    det_ymin += int(det_height * (det_box_scale - 1) / 2)
                    det_xmax += int(det_width * (det_box_scale - 1) / 2)
                    det_ymax += int(det_height * (det_box_scale - 1) / 2)
                    det_xmin = max(det_xmin, 0)
                    det_ymin = max(det_ymin, 0)
                    det_xmax = min(det_xmax, image_width - 1)
                    det_ymax = min(det_ymax, image_height - 1)
                    det_width = det_xmax - det_xmin + 1
                    det_height = det_ymax - det_ymin + 1
                    cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                    det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
                    #np.save(f'{CROP_DIR}/{pid[:-4]}.npy', np.array([det_ymin, det_ymax, det_xmin, det_xmax]))
                    det_crop = cv2.resize(det_crop, (input_size, input_size))
                    inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
                    #inputs.show()
                    inputs = preprocess(inputs).unsqueeze(0)
                    inputs = inputs.to(device)
                    lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net,
                                                                                                             inputs,
                                                                                                             preprocess,
                                                                                                             input_size,
                                                                                                             net_stride,
                                                                                                             num_nb)
                    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                    tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                    tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
                    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
                    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                    lms_pred = lms_pred.cpu().numpy()
                    lms_pred_merge = lms_pred_merge.cpu().numpy()
                    pred_export = np.zeros([cfg.num_lms, 2])
                    for i in range(cfg.num_lms):
                        x_pred = lms_pred_merge[i * 2] * det_width
                        y_pred = lms_pred_merge[i * 2 + 1] * det_height
                        pred_export[i, 0] = (x_pred + det_xmin) / image_width
                        pred_export[i, 1] = (y_pred + det_ymin) / image_height
                        cv2.circle(image, (int(x_pred) + det_xmin, int(y_pred) + det_ymin), 1, (0, 0, 255), 2)
                        if i == 76:
                            cv2.circle(image, (int(x_pred) + det_xmin, int(y_pred) + det_ymin), 1, (255, 0, 0), 2)

                if pred_export is not None:
                    print('exporting stuff to ' + image_dir)
                    landmakr_dir =  f'{image_dir}/../PIPnet_landmarks/'
                    os.makedirs(landmakr_dir, exist_ok=True)
                    np.save(landmakr_dir + f'/{file_name[:-4]}.npy', pred_export)
                    lms.append(pred_export)
                    exp_dir = image_dir + '/../PIPnet_annotated_images/'
                    os.makedirs(exp_dir, exist_ok=True)
                    cv2.imwrite(exp_dir + f'/{file_name}', image)

                # cv2.imshow('1', image)
                # cv2.waitKey(0)

    lms = np.stack(lms, axis=0)
    os.makedirs(f'{image_dir}/../pipnet', exist_ok=True)
    np.save(f'{image_dir}/../pipnet/test.npy', lms)


def run(exp_path, image_dir, start_frame = 0,
        vertical_crop : bool = False,
        static_crop : bool = False
        ):
    experiment_name = exp_path.split('/')[-1][:-3]
    data_name = exp_path.split('/')[-2]
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

    my_config = importlib.import_module(config_path, package='PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    save_dir = os.path.join('../PIPNet/snapshots', cfg.data_name, cfg.experiment_name)

    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                           net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=cfg.pretrained)
        net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                           net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=cfg.pretrained)
        net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                            net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v2':
        mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
        net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v3':
        mbnet = mobilenetv3_large()
        if cfg.pretrained:
            mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
        net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    else:
        print('No such backbone!')
        exit(0)

    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)

    weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs - 1))
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose(
        [transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])


    #for pid in pids:
    pid = "FaMoS_180424_03335_TA_selfie_IMG_0092.jpg"
    pid = "FaMoS_180426_03336_TA_selfie_IMG_0152.jpg"



    demo_image(image_dir, pid, None, net, preprocess, cfg, cfg.input_size, cfg.net_stride, cfg.num_nb,
                           cfg.use_gpu,
                           device, start_frame=start_frame, vertical_crop=vertical_crop, static_crop=static_crop)



if __name__ == '__main__':
    base_path = '/mnt/rohan/cluster/valinor/jschmidt/becominglit/1015/HEADROT/img_cc_4/cam_220700191/'
    base_path = '/home/giebenhain/try_tracking_obama2/rgb'
    #base_base_path = '/home/giebenhain/test_videos_p3dmm_full/'
    base_base_path = '/mnt/rohan/cluster/andram/sgiebenhain/test_video_p3dmm_full/'
    v_names = [f for f in os.listdir(base_base_path) if f.startswith('th1k')]
    print(v_names)
    #v_names = list(range(800, 813))
    #v_names = ['yu', 'marc', 'karla', 'karla_light', 'karla_glasses_hat', 'karla_glasses'] #['merlin', 'haoxuan']
    for video_name in v_names:
        base_path = f'{base_base_path}/{video_name}/rgb/'
        #if os.path.exists(f'{base_path}/../cropped/'):
        #    print('SKIP', base_path)
        #    continue
        start_frame = -1
        vertical_crop=True
        try:
            run('experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py', base_path, start_frame=start_frame, vertical_crop=False, static_crop=True)
        except Exception as ex:
            traceback.print_exc()
