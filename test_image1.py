import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from SNSRGAN_model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', default='data/test/LR/32.png', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='SNSRGAN.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()

def fill_label(labels, n_class, img_size):
    fill = torch.zeros([n_class, n_class, img_size, img_size])
    for i in range(n_class):
        fill[i, i, :, :] = 1

    y_fill = fill[labels]

    return y_fill


model.load_state_dict(torch.load('trained_models/' + MODEL_NAME, map_location=lambda storage, loc: storage))
image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)

label = 1
y_fill = fill_label(label, 14, 256)
y_fill = Variable(y_fill).unsqueeze(0)
start = time.perf_counter()
out = model(image, y_fill)
elapsed = (time.perf_counter() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('SR.png')
