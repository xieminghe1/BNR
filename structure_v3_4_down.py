import math, pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        self.net1 = nn.Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=None, groups=1)

    def forward(self, x):

        out = self.net1(x)

        return out
class FreTransfer_1(nn.Module):
    def __init__(self):
        super(FreTransfer_1, self).__init__()
        self.net1 = nn.Conv2d(1, 4, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=None, groups=1)

    def forward(self, x):

        out = self.net1(x)

        return out


class Transfer(nn.Module):
    def __init__(self):
        super(Transfer, self).__init__()
        self.net1 = nn.Conv2d(4, 16, kernel_size=(2, 2), stride=(2, 2), padding=0, groups=1, bias=None)


    def forward(self, x):
        out = self.net1(x)
        return out

class TransferInv(nn.Module):
    def __init__(self):
        super(TransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(16, 4, kernel_size=(2, 2), stride=(2, 2), padding=0, groups=1, bias=None)

    def forward(self, x):
        out = self.net1(x)
        return out

class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(16, 4, kernel_size=(2, 2), stride=(2, 2), padding=0, groups=1, bias=None)

    def forward(self, x):
        out = self.net1(x)
        return out

class Fusion(nn.Module):
    def __init__(self, in_chn, feature_chn=16, block_num=3):
        super(Fusion, self).__init__()
        m_list = [nn.Conv2d(in_chn, feature_chn, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        for b in range(block_num - 2):
            m_list += [nn.Conv2d(feature_chn, feature_chn, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        m_list += [nn.Conv2d(feature_chn, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()]
        self.model = nn.Sequential(*m_list)

    def forward(self, x):
        # with torch.no_grad():
        return self.model(x)
class ColorTransfer(nn.Module):
    def __init__(self):
        super(ColorTransfer, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)


    def forward(self, x):
        out = self.net1(x)
        return out

class Denoise(nn.Module):
    def __init__(self, in_chn, feature_chn=64, block_num=3):
        super(Denoise, self).__init__()
        m_list = [nn.Conv2d(in_chn, feature_chn, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        for b in range(block_num - 2):
            m_list += [nn.Conv2d(feature_chn, feature_chn, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        m_list += [nn.Conv2d(feature_chn, 16, kernel_size=3, stride=1, padding=1)]
        self.model = nn.Sequential(*m_list)

    def forward(self, x):
        return self.model(x)


class Refine(nn.Module):
    def __init__(self, in_chn, feature_chn=16, block_num=3):
        super(Refine, self).__init__()
        m_list = [nn.Conv2d(in_chn, feature_chn, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        for b in range(block_num - 2):
            m_list += [nn.Conv2d(feature_chn, feature_chn, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        m_list += [nn.Conv2d(feature_chn, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()]
        self.model = nn.Sequential(*m_list)

    def forward(self, x):
        return self.model(x)


class VideoDenoise(nn.Module):
    def __init__(self, feature_chn=16, top=False):
        super(VideoDenoise, self).__init__()
        self.fusion = Fusion(10)
        # self.denoise = Denoise(in_chn = 25 if from_down else 21, feature_chn=feature_chn, block_num=6 if top else 3)
        self.denoise = Denoise(in_chn = 21, feature_chn=feature_chn, block_num=3)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.top = top

    def forward(self, prev, curr, prev_sigma, curr_sigma, blend):
        ll0 = prev[:, 0:4, :, :]
        ll1 = curr[:, 0:4, :, :]

        # fusion
        fusion_in = torch.cat([ll1, ll0, prev_sigma, curr_sigma], dim=1)
        fusion_in = self.avg_pool(fusion_in)
        print(fusion_in.shape)
        gamma = self.fusion(fusion_in) * blend
        gamma = F.interpolate(gamma, size=(360,640))
        # fusion
        # fusion_out = torch.mul(prev, gamma) + torch.mul(curr, (1 - gamma))
        fusion_out = curr - torch.mul(curr-prev,gamma)
        curr_sigma=curr_sigma.reshape(1,64,45,80)
        prev_sigma=prev_sigma.reshape(1,64,45,80)
        gamma=gamma.reshape(1,64,45,80)
        sigma=gamma*gamma*prev_sigma+(1-gamma)*(1-gamma)*curr_sigma
        sigma = sigma.reshape(1,1,360,640)
        curr_sigma=curr_sigma.reshape(1,1,360,640)
        prev_sigma=prev_sigma.reshape(1,1,360,640)
        gamma = gamma.reshape(1,1,360,640)

        denoise_in = torch.cat([fusion_out, ll1, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        return gamma, fusion_out, denoise_out, sigma

class MainDenoise(nn.Module):
    def __init__(self):
        super(MainDenoise, self).__init__()

        # self.ft_1 = FreTransfer_1()
        self.vd = VideoDenoise(feature_chn=16)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.refine = Refine(33)
        self.trans = Transfer()
        self.trans_inv = TransferInv()

    def transform(self, x):

        out = self.trans(x)
        return out

    def transforminv(self, x):

        out = self.trans_inv(x)
        return out

    def forward(self, noisy_img, fusion_img=torch.randn((1,4,720,1280)), prev_sigma0=torch.randn((1,1,360,640)), coeff_a=1, coeff_b=1, blend=0, motion_fr=0., static_dr=0., black_level=torch.randn((1,4,8,8)), curBlend_ratio_3d=0., curBlend_ratio_2d=0.):


        black_level = F.interpolate(black_level, size=(45, 80), mode='bilinear', align_corners=False)
        black_level = F.interpolate(black_level, size=(720, 1280))

        noisy_img = noisy_img.reshape(1,256,180,80)
        black_level = black_level.reshape(1,256,180,80)

        cvt_k = 0.024641 / coeff_a + coeff_b * 0
        cvt_k =torch.max(cvt_k, torch.ones_like(cvt_k)*0.8) 
        cvt_k1 = 1/cvt_k
        
 
        noisy_img = ((noisy_img - black_level).clip(0., 4095) + 112) / (4319 - black_level)*cvt_k
        noisy_img = noisy_img.reshape(1,4,720,1280)
        curr_ft = self.transform(noisy_img)


        curr_sigma = curr_ft[:, 0:1, :, :] * 0.024641 + 0.001317
        curr_ft=curr_ft.reshape(1,256,180,80)
        fusion_img=fusion_img.reshape(1,256,180,80)

        curr_sigma=curr_sigma.reshape(1,64,45,80)
        prev_sigma0=prev_sigma0.reshape(1,64,45,80)
        
        prev_ft = curr_ft*(1-blend) + fusion_img*blend
        prev_sigma = curr_sigma*(1-blend) + prev_sigma0*blend    

        prev_sigma=prev_sigma.reshape(1,1,360,640)
        curr_sigma=curr_sigma.reshape(1,1,360,640)
        prev_sigma0=prev_sigma0.reshape(1,1,360,640)
        
        curr_ft=curr_ft.reshape(1,16,360,640)
        fusion_img=fusion_img.reshape(1,16,360,640)
        prev_ft=prev_ft.reshape(1,16,360,640)   


        gamma, fusion_out, denoise_out, sigma = self.vd(prev_ft, curr_ft, prev_sigma, curr_sigma, blend)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out, sigma], axis=1)  
        refine_in = self.max_pool(refine_in)
        omega = self.refine(refine_in)
        print(omega.shape)
        omega=omega.reshape(1,64,45,20)
        omega = F.interpolate(omega, size=(45,80), mode="bilinear")
        omega = omega.reshape(1,1,360,640)
        gamma=gamma.reshape(1,64,45,80)
        gamma_clip = gamma.clip(motion_fr, static_dr)
        gamma_clip = gamma_clip.reshape(1,1,360,640)
        gamma=gamma.reshape(1,1,360,640)
        # gamma_clip = gamma.clip(motion_fr, static_dr)


        # 简化后的计算
        refine_out_pre = denoise_out-torch.mul(denoise_out-fusion_out,gamma_clip)
        refine_out = refine_out_pre-torch.mul(refine_out_pre-fusion_out,omega)


        gamma_ref_3d = gamma.minimum(curBlend_ratio_3d)
        gamma_ref_2d = curBlend_ratio_2d - gamma.minimum(curBlend_ratio_2d)

        diff = refine_out - curr_ft
        refine_out = refine_out - torch.mul(diff, gamma_ref_3d + gamma_ref_2d)
        refine_out = self.transforminv(refine_out)
        refine_out = refine_out * cvt_k1  * 4095 + 15
    
        return refine_out, fusion_out, sigma


if __name__ == "__main__":
    model = MainDenoise()

    stat = torch.load('./model_ks_v3_4_groff_dlon_2048_112all_finetune2_0930.pth', map_location='cpu')
    # stat = torch.load('./ks_model/model_ks_anchor102400_v3_4_n4c_0911.pth', map_location='cpu')
    # stat = torch.load('./ks_model/model_ks_anchor102400_v3_4_gamrefoff_dlon_0909.pth', map_location='cpu')
    # stat = torch.load('./ks_model/model_57000_ks_anchor102400_n4c_groff_dlon_2048_0910.pth', map_location='cpu')
    # stat = torch.load('./model_ks_anchor102400_v3_4_n4c_0904.pth', map_location='cpu')

    stat = {k.replace('module.', ''):v for k,v in stat['model'].items()}
    del stat['ft_1.net1.weight']
    del stat['ft.net1.weight']
    del stat['fti.net1.weight']
    del stat['ftis.0.weight']
    del stat['ftis.0.bias']
    model.load_state_dict(stat)
    noisy_img = torch.randn((1,4,720,1280))
    # fusion_img = torch.randn((1,4,720,1280))
    fusion_img = torch.randn((1,16,360,640))
    prev_sigma0 = torch.randn((1,1,360,640))
    b = (15914.9-5372.21) / 51200 * (56227-51200) + 5372.21
    a = (90.54-40.84) / 51200 * (56227-51200) + 40.84
    coeff_a = torch.tensor((a)).reshape(1,1,1,1)
    coeff_b = torch.tensor((b)).reshape(1,1,1,1)
    blend = torch.tensor((1.)).reshape(1,1,1,1)
    motion_fr=torch.tensor((0.1)).reshape(1,1,1,1)
    static_dr=torch.tensor((0.8)).reshape(1,1,1,1)
    # black_level=torch.tensor((112.)).reshape(1,1,1,1)
    black_level=torch.randn((1,4,8,8))

    curBlend_ratio_3d = torch.tensor((0.9)).reshape(1,1,1,1)
    curBlend_ratio_2d = torch.tensor((0.1)).reshape(1,1,1,1)

    dummy_input = (noisy_img, fusion_img, prev_sigma0, coeff_a, coeff_b, blend, motion_fr, static_dr, black_level, curBlend_ratio_3d, curBlend_ratio_2d)
    input_names = ("noisy_img", "fusion_img", "prev_sigma0", "coeff_a", "coeff_b", "blend", "motion_fr", "static_dr", "black_level", "curBlend_ratio_3d", "curBlend_ratio_2d")
    dynamic_axes = {'noisy_img': {0: 'batch', 2: 'height', 3: 'width'}, 'fusion_img': {0: 'batch', 2: 'height', 3: 'width'}, 'prev_sigma0': {0: 'batch', 2: 'height', 3: 'width'}}

    # torch.onnx.export(model, dummy_input, 'new102400ks_anchor102400_v3_4_groff_dloff_2048_112all_finetune2_normorg_0930.onnx' ,input_names=input_names) 
    torch.onnx.export(model, dummy_input, 'reshape_test_down.onnx' ,input_names=input_names) 
