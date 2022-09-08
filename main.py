# -*- coding: utf-8 -*-
# @Time    : 2022/9/3 0:00
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 16:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : main.py
# @Software: PyCharm
from comet_ml import Experiment


import random
import numpy as np

from data_loader import VideoFrameAndMaskDataset


import torchmetrics
import torch.optim as optim
import torch.backends.cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from train import *
from utils.loss import *
from model.VideoInpaintingModel import VideoInpaintingModel_G, VideoInpaintingModel_T, VideoInpaintingModel_S

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_comet = False
autocast_button = False

random.seed(48)
np.random.seed(48)
torch.manual_seed(48)
torch.cuda.manual_seed_all(48)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# seed = 24
# seed_everything(24)

hyper_params = {
    "mode": 'Video_inpainting',
    "ex_number": '3D_Gated Convolution Model',
    "input_size": (3, 256, 256),
    "batch_size": 1,
    "learning_rate": 1e-4,
    "epochs": 200,
    "threshold": 28,
    "checkpoint": False,
    "Img_Recon": True,
    "src_path": 'E:/BJM/Video_Inpainting',
    "check_path": 'F:/BJM/Remote_Image_Inpainting/2022-08-24-14-59-27.160160/save_model/Epoch_10_eval_16.614881643454233.pt'
}

experiment = object
lr = hyper_params['learning_rate']
mode = hyper_params['mode']
Epochs = hyper_params['epochs']
src_path = hyper_params['src_path']
batch_size = hyper_params['batch_size']
input_size = hyper_params['input_size'][1:]
threshold = hyper_params['threshold']
Checkpoint = hyper_params['checkpoint']
Img_Recon = hyper_params['Img_Recon']
check_path = hyper_params['check_path']

opti = {"norm": "SN",
        "nf": 64,
        "bias": True,
        "conv_type": "gated",
        "conv_by": "2dtsm"
        }
d_s_args = {
    "nf": 64,
    "use_sigmoid": True,
    "norm": "SN",
    "conv_type": "vanilla",
    "conv_by": "3d"}

d_t_args = {
    "nf": 64,
    "use_sigmoid": True,
    "norm": "SN",
    "conv_type": "vanilla",
    "conv_by": "2dtsm"}


# ===============================================================================
# =                                    Comet                                    =
# ===============================================================================

if train_comet:
    experiment = Experiment(
        api_key="sDV9A5CkoqWZuJDeI9JbJMRvp",
        project_name="Motion_Image_Enhancement",
        workspace="LovingThresh",
    )

# ===============================================================================
# =                                     Data                                    =
# ===============================================================================

path = r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_train\train/'
input_dir = r'*\choice_images\*'
mask_dir = r'*\choice_images_mask_ALL\*'

val_path = r'O:\Dataset\multitemporal-urban-development\archive_2\SN7_buildings_test_public\test_public/'
val_input_dir = r'*\choice_images\*'
val_mask_dir = r'*\choice_images_mask_ALL\*'

train_dataset = VideoFrameAndMaskDataset(path, input_dir, mask_dir, (256, 256))
val_dataset = VideoFrameAndMaskDataset(val_path, val_input_dir, val_mask_dir, (256, 256))
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# ===============================================================================
# =                                     Model                                   =
# ===============================================================================

generator = VideoInpaintingModel_G(opti)
discriminator_T = VideoInpaintingModel_T(d_t_args=d_t_args)
discriminator_S = VideoInpaintingModel_S(d_s_args=d_s_args)

# ===============================================================================
# =                                    Setting                                  =
# ===============================================================================


loss_function_D = {'loss_function_dis': nn.BCELoss()}

loss_function_G_ = {'loss_function_dis': nn.BCELoss()}

loss_function_G = {
    'ReconLoss': ReconLoss(),
    'perceptual_loss': VGGLoss(),
    'StyleLoss': StyleLoss(),
    'EdgeLoss': EdgeLoss()
}

eval_function_l2 = torchmetrics.functional.mean_squared_error
eval_function_psnr = torchmetrics.functional.image.psnr.peak_signal_noise_ratio
eval_function_ssim = torchmetrics.functional.image.ssim.structural_similarity_index_measure

eval_function_G = {'eval_function_psnr': eval_function_psnr,
                   'eval_function_ssim': eval_function_ssim,
                   'eval_function_L2': eval_function_l2
                   }


optimizer_ft_D_T = optim.Adam(discriminator_T.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_ft_D_S = optim.Adam(discriminator_S.parameters(), lr=lr, betas=(0.5, 0.999))

optimizer_ft_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


exp_lr_scheduler_D_T = lr_scheduler.StepLR(optimizer_ft_D_T, step_size=10, gamma=0.8)
exp_lr_scheduler_D_S = lr_scheduler.StepLR(optimizer_ft_D_S, step_size=10, gamma=0.8)
exp_lr_scheduler_G = lr_scheduler.StepLR(optimizer_ft_G, step_size=10, gamma=0.8)

# ===============================================================================
# =                                  Copy & Upload                              =
# ===============================================================================

output_dir = copy_and_upload(experiment, hyper_params, train_comet, src_path)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
train_writer = SummaryWriter('{}/trainer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))
val_writer = SummaryWriter('{}/valer_{}'.format(os.path.join(output_dir, 'summary'), timestamp))

# ===============================================================================
# =                                Checkpoint                                   =
# ===============================================================================

if Checkpoint:
    checkpoint = torch.load(check_path)
    generator.load_state_dict(checkpoint)
    print("Load CheckPoint!")


# ===============================================================================
# =                                    Training                                 =
# ===============================================================================


train_GAN(generator, discriminator_T, discriminator_S, optimizer_ft_G, optimizer_ft_D_T, optimizer_ft_D_S,
          loss_function_G_, loss_function_G, loss_function_D, exp_lr_scheduler_G, exp_lr_scheduler_D_T, exp_lr_scheduler_D_S,
          eval_function_G, train_loader, val_loader, Epochs, device, threshold,
          output_dir, train_writer, val_writer, experiment, train_comet)
