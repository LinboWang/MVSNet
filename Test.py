import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from utils.dataloader import test_dataset
from lib.network import MVSNet as Model
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--model_name', type=str, default="MVSNet")
    parser.add_argument('--pth_path', type=str, default='./checkpoint/MVSNet.pth')
    opt = parser.parse_args()
    model = Model()
    save_model = torch.load(opt.pth_path)
    model.load_state_dict(save_model)
    model.cuda()
    model.eval()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']:
        data_path = 'D://datasets/polyp/TestDataset/{}'.format(_data_name)
        save_path = './result_map/{}/{}/'.format(opt.model_name, _data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        DSC = 0.
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            pred = model(image)
            res = pred
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)
        print(_data_name, 'Finish!')
