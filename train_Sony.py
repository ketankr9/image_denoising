import os,time

import numpy as np
import rawpy
import glob, scipy.io

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import SeeInDark
from net_canny import Net
from pytorch_msssim import MSSSIM

saveimg = True
ps = 512 #patch size for training
device = torch.device('cuda') #'cuda:'+os.environ['CUDA']#torch.device('cuda') #if torch.cuda.is_available() else 'cpu')
DIR = '/home/cse/ug/15074014/'
if os.environ["USER"] == "ketankr9":
    DIR = '/home/ketankr9/workspace/mtp/codes/'
    device = torch.device('cpu')
    ps = 256
input_dir = DIR + 'Sony/short/'
gt_dir = DIR + 'Sony/long/'
result_dir = model_dir = DIR + 'Sony/result_Sony_edge_psnr_ssim/'

os.system('mkdir -p '+result_dir)
chpkdir = model_dir+'checkpoint_sony_resume.pth'

writer = SummaryWriter(result_dir+'log')

print(device)

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))




save_freq = 100

DEBUG = 0
if DEBUG == 1:
    save_freq = 100
    train_ids = train_ids[0:5]
    test_ids = test_ids[0:5]

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

# LOSS REDUCED MEAN
mseclass = nn.MSELoss(reduction='mean')
def reduce_mean(out_im, gt_im):
    return mseclass(out_im, gt_im)

# LOSS VGG
from utils import VGGLoss, GaussianSmoothing
vggloss = VGGLoss(device=device)
gaussianSmoothing = GaussianSmoothing(3, 5, 1, device=device)

# LOSS COLORLOSS
def colorloss(out, gt):
    out = gaussianSmoothing(out)
    gt  = gaussianSmoothing(gt)
    return torch.abs(out-gt).mean()

# MSSSIM
msssim = MSSSIM().to(device)

# LOSS CANNY
canny = Net(threshold=3.0, device=device).to(device)
canny.eval()
def canny_loss(out_im, gt_im, ch=""):
    blurred_img1, grad_mag1, grad_orientation1, thin_edges1, thresholded1, early_threshold1 = canny(gt_im)
    blurred_img2, grad_mag2, grad_orientation2, thin_edges2, thresholded2, early_threshold2 = canny(out_im)

    if ch == '1':
        return mseclass(thresholded1, thresholded2)
    elif ch == '1bool':
        return mseclass(thresholded1!=zero, thresholded2!=zero)
    elif ch == '2':
        return mseclass(early_threshold1, early_threshold2)
    elif ch == '2bool':
        return mseclass(early_threshold1!=zero, early_threshold2!=zero)
    elif ch == '3':
        return mseclass(thresholded1, thresholded2) + mseclass(early_threshold1, early_threshold2)
    elif ch == '3bool':
        return mseclass(thresholded1!=zero, thresholded2!=zero) + mseclass(early_threshold1!=zero, early_threshold2!=zero)

    return mseclass(thresholded1/(thresholded1.max()+1), thresholded2/(thresholded2.max()+1))

#Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

g_loss = np.zeros((5000,1))

learning_rate = 1e-4
model = SeeInDark().to(device)
opt = optim.Adam(model.parameters(), lr = learning_rate)

#load last saved model weights
if os.path.isfile(chpkdir):
    checkpoint = torch.load(chpkdir)
    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['optimizer'])
    lastepoch = checkpoint['epoch'] + 1
else:
    lastepoch = 0
    model._initialize_weights()

print("*****lastepoch***** ", lastepoch)
for epoch in range(lastepoch,4001):
    cnt=0
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    E_loss = {'vgg':0, 'c_loss':0, 'mse':0, 'total':0}
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.random_integers(0,len(in_files)-1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)

        st=time.time()
        cnt+=1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)


        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]


        if np.random.randint(2,size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))


        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        in_img = torch.from_numpy(input_patch).permute(0,3,1,2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)


        model.zero_grad()
        out_img = model(in_img)

        # c_loss = colorloss(out_img, gt_img)
        # vgg_loss = vggloss.loss(out_img, gt_img)
        # mse_loss = reduce_mean(out_img, gt_img)
        # loss = c_loss + vgg_loss + mse_loss

        alpha, beta = 0.8, 0.1
        can_loss = (1-alpha-beta)*canny_loss(out_img, gt_img)
        mse_loss = alpha*reduce_mean(out_img, gt_img)
        msssim_loss = beta*(1-msssim(out_img, gt_img))
        loss = mse_loss + can_loss + msssim_loss

        loss.backward()

        opt.step()
        g_loss[ind]=loss.item()
        out=("%d %d C:%.3f S:%.3f P:%.3f Loss=%.3f Time=%.3f"%(epoch,cnt,can_loss, msssim_loss, mse_loss, np.mean(g_loss[np.where(g_loss)]),time.time()-st))
        print(out)
        try:
            os.system('echo ' + out + ' >> '+result_dir+'jobout.txt')
        except:
            pass
        E_loss = {'CANNY':0, 'MSE':0, 'MSSSIM':0, 'total':0}
        E_loss['CANNY'] += can_loss
        E_loss['MSE'] += mse_loss
        E_loss['MSSSIM'] += msssim_loss
        E_loss['total'] += np.mean(g_loss[np.where(g_loss)])

        if epoch%save_freq==0:
            if not os.path.isdir(result_dir + '%04d'%epoch):
                os.makedirs(result_dir + '%04d'%epoch)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output,0),1)

            if saveimg:
                temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)
                # scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))
                # torch.save(model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)
    
    torch.save({'epoch': epoch, \
        'model': model.state_dict(), \
        'optimizer': opt.state_dict(),\
        }, model_dir+'checkpoint_sony_resume.pth')

    writer.add_scalar('Loss/Edge', E_loss['CANNY'], epoch)
    writer.add_scalar('Loss/MSSSIM', E_loss['MSSSIM'], epoch)
    writer.add_scalar('Loss/RMean', E_loss['MSE'], epoch)
    writer.add_scalar('LossTotal', E_loss['total'], epoch)

