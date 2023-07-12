import cv2
import torch
import tqdm
import os
import numpy as np
np.set_printoptions(suppress=True, precision=3)
import h5py
import copy

from utils.group import HeatmapParser
import utils.img
import data.Tooth.ref as ds

parser = HeatmapParser()


def post_process(det, mat_, trainval, r_x, r_y, resolution=None):

    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
    res = det.shape[1:3]
    n = resolution[0] / res[0]
    # for i in range(det.shape[0]):
    #     print(i, np.unravel_index(det[i].argmax(), det[i].shape))
    print("det: ", det.shape)
    cropped_preds = parser.parse(np.float32([det]))[0]
    # print("crop_preds: ", cropped_preds)

    if len(cropped_preds) > 0:
        cropped_preds[:, :, :2] = utils.img.kpt_affine(cropped_preds[:, :, :2] * n, mat)  # size 1x3x3

    preds = np.copy(cropped_preds)
    print("preds first: ", preds)
    # for inverting predictions from input res on cropped to original image
    if trainval != 'cropped':
        for j in range(preds.shape[1]):
            preds[0, j, 0] = preds[0, j, 0] / r_x
            preds[0, j, 1] = preds[0, j, 1] / r_y

    print("-------pred: ", np.around(preds, 3))
    print("*********************************")
    return preds


def inference(img, func, config, r_x, r_y):
    """
    forward pass at test time
    calls post_process to post process results
    """
    height, width = img.shape[0:2]
    center = (width / 2, height / 2)
    scale = max(height, width) / 200
    res = (config['train']['input_res'], config['train']['input_res'])

    mat_ = utils.img.get_transform(center, scale, res)[:2]
    cv2.imshow("inp", img)
    cv2.waitKey(0)
    inp = img / 255.0

    def array2dict(tmp):
        return {
            'det': tmp[0][:, :, :3],
        }

    out = array2dict(func([inp]))

    return post_process(np.minimum(out['det'][0, -1], 1), mat_, 'valid', r_x, r_y, res)


def mpii_eval(pred, gt, normalizing, num_train, bound=0.5):
    """
    Use PCK with threshold of .5 of normalized distance (presumably head size)
    """

    correct = {'all': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                       'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0, 'shoulder': 0},
               'visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                           'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0, 'shoulder': 0},
               'not visible': {'total': 0, 'ankle': 0, 'knee': 0, 'hip': 0, 'pelvis': 0,
                               'thorax': 0, 'neck': 0, 'head': 0, 'wrist': 0, 'elbow': 0, 'shoulder': 0}}
    count = copy.deepcopy(correct)
    correct_train = copy.deepcopy(correct)
    count_train = copy.deepcopy(correct)
    idx = 0
    for p, g, normalize in zip(pred, gt, normalizing):
        for j in range(g.shape[1]):
            vis = 'visible'
            if g[0, j, 0] == 0:  # # not in picture!
                continue
            if g[0, j, 2] == 0:
                vis = 'not visible'
            joint = 'ankle'
            if j == 1 or j == 4:
                joint = 'knee'
            elif j == 2 or j == 3:
                joint = 'hip'
            elif j == 6:
                joint = 'pelvis'
            elif j == 7:
                joint = 'thorax'
            elif j == 8:
                joint = 'neck'
            elif j == 9:
                joint = 'head'
            elif j == 10 or j == 15:
                joint = 'wrist'
            elif j == 11 or j == 14:
                joint = 'elbow'
            elif j == 12 or j == 13:
                joint = 'shoulder'

            if idx >= num_train:
                count['all']['total'] += 1
                count['all'][joint] += 1
                count[vis]['total'] += 1
                count[vis][joint] += 1
            else:
                count_train['all']['total'] += 1
                count_train['all'][joint] += 1
                count_train[vis]['total'] += 1
                count_train[vis][joint] += 1
            error = np.linalg.norm(p[0]['keypoints'][j, :2] - g[0, j, :2]) / normalize
            if idx >= num_train:
                if bound > error:
                    correct['all']['total'] += 1
                    correct['all'][joint] += 1
                    correct[vis]['total'] += 1
                    correct[vis][joint] += 1
            else:
                if bound > error:
                    correct_train['all']['total'] += 1
                    correct_train['all'][joint] += 1
                    correct_train[vis]['total'] += 1
                    correct_train[vis][joint] += 1
        idx += 1

    # # breakdown by validation set / training set
    for k in correct:
        print(k, ':')
        for key in correct[k]:
            print('Val PCK @,', bound, ',', key, ':', round(correct[k][key] / max(count[k][key], 1), 3), ', count:', count[k][key])
            print('Tra PCK @,', bound, ',', key, ':', round(correct_train[k][key] / max(count_train[k][key], 1), 3), ', count:', count_train[k][key])
        print('\n')


def get_img(config):
    """
    Load testing images
    """
    input_res = config['train']['input_res']
    output_res = config['train']['output_res']
    tooth = ds.Tooth()
    num_train, num_val = tooth.getLength()

    tr = tqdm.tqdm(range(0, num_train), total=num_train)

    for i in tr:
        img_name, kp, vis = tooth.getAnnots(i)
        print(img_name, kp)
        # # img
        orig_img = cv2.imread(os.path.join(ds.img_dir, img_name))  # [:, :, ::-1]
        im = cv2.resize(orig_img, (input_res, input_res))

        # # kp
        r_x = input_res / orig_img.shape[1]
        r_y = input_res / orig_img.shape[0]

        for i in range(len(kp)):
            kp[i][0] = int(kp[i][0] * r_x)
            kp[i][1] = int(kp[i][1] * r_y)
        kp2 = np.insert(kp, 2, vis, axis=1)
        kps = np.zeros((1, 3, 3))
        kps[0] = kp2
        print(kps)

        yield kps, im, r_x, r_y


def main():
    from train import init
    func, config = init()

    def runner(imgs):
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

    def do(img, r_x, r_y):
        ans = inference(img, runner, config, r_x, r_y)
        if len(ans) > 0:
            ans = ans[:, :, :3]

        # # ans has shape N,16,3 (num preds, joints, x/y/visible)
        pred = []
        for i in range(ans.shape[0]):
            pred.append({'keypoints': ans[i, :, :]})
        return pred

    gts = []
    preds = []
    normalizing = []

    num = 0
    for anns, img, r_x, r_y in get_img(config):
        gts.append(anns)
        pred = do(img, r_x, r_y)
        # preds.append(pred)
        # normalizing.append(n)
        num += 1

    # mpii_eval(preds, gts, normalizing, num)


if __name__ == '__main__':
    main()
