import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from models.base_model import UNET

def find_padding(img, depth=2**4):
    B, C, H, W = img.shape

    h_pad = (depth - H % depth) % depth
    w_pad = (depth - W % depth) % depth
    return h_pad, w_pad

def get_pretrained_path(model_name):
    # 'SRUNET_x2', 'SRUNET_x3', 'SRUNET_x4', 'SRUNET_x234', 'SRUNET_interpolation', 'SRUNET_x234_interpolation'

    current_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    if model_name == 'SRUNET_x2':
        return current_path + '/pretrained/SRUNET_scale_x2.pt'
    elif model_name == 'SRUNET_x3':
        return current_path + '/pretrained/SRUNET_scale_x3.pt'
    elif model_name == 'SRUNET_x4':
        return current_path + '/pretrained/SRUNET_scale_x4.pt'
    elif model_name == 'SRUNET_x234':
        return current_path + '/pretrained/SRUNET_scale_x234.pt'
    # elif model_name == 'SRUNET_interpolation':
    #     return current_path + '/pretrained/SRUNET_x3.pt'
    # elif model_name == 'SRUNET_x234_interpolation':
    #     return current_path + '/pretrained/SRUNET_x3.pt'
    else:
        raise Exception('Model not found')


def upscale_image(img, model_name, scale_factor):
    # get img width height
    width, height = img.size
    img_mode = img.mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((height * scale_factor, width * scale_factor),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    
    #Load Model
    checkpoint = torch.load(get_pretrained_path(
        model_name), map_location=torch.device('cpu'))
    model = UNET()
    model.load_state_dict(checkpoint['best_model_state_dict'])
    model.eval()

    data = transform(img).clamp(0, 1).unsqueeze(0)
    # print(data.shape, img.mode)
    # return img
    h_pad, w_pad = find_padding(data)
    data = F.pad(data, (0, w_pad, 0, h_pad), mode='reflect')
    

    with torch.no_grad():
        img_scale_pred = model(data).clamp(0, 1)
        if h_pad > 0 and w_pad > 0:
            img_scale_pred = img_scale_pred[..., :-h_pad, :-w_pad]
        elif h_pad > 0:
            img_scale_pred = img_scale_pred[..., :-h_pad, :]
        elif w_pad > 0:
            img_scale_pred = img_scale_pred[..., :, :-w_pad]
        else:
            img_scale_pred = img_scale_pred

        img_scale_pred = img_scale_pred.squeeze(0)
    return transforms.ToPILImage()(img_scale_pred).convert(img_mode)

def enhanced_image(img, model_name):
    img_mode = img.mode
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    
    #Load Model
    checkpoint = torch.load(get_pretrained_path(
        model_name), map_location=torch.device('cpu'))
    model = UNET()
    model.load_state_dict(checkpoint['best_model_state_dict'])
    model.eval()

    data = transform(img).clamp(0, 1).unsqueeze(0)
    h_pad, w_pad = find_padding(data)
    data = F.pad(data, (0, w_pad, 0, h_pad), mode='reflect')
    

    with torch.no_grad():
        img_scale_pred = model(data).clamp(0, 1)
        if h_pad > 0 and w_pad > 0:
            img_scale_pred = img_scale_pred[..., :-h_pad, :-w_pad]
        elif h_pad > 0:
            img_scale_pred = img_scale_pred[..., :-h_pad, :]
        elif w_pad > 0:
            img_scale_pred = img_scale_pred[..., :, :-w_pad]
        else:
            img_scale_pred = img_scale_pred

        img_scale_pred = img_scale_pred.squeeze(0)
    return transforms.ToPILImage()(img_scale_pred).convert(img_mode)