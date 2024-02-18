import os
import time
import torch
import torch.nn.functional as F
import models.models as m
import numpy as np

from utils.utils import *
from omegaconf import OmegaConf

# Initialize the patch
def patch_initialization(image_size=image_size, noise_percentage=0.03):
    # для дообучения
    # patch = cv2.imread("4n_8.png")
    # patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    # patch = (patch/255).transpose(2, 0, 1).astype('float64')

    # для начального обучения
    mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
    patch = np.random.rand(image_size[0], mask_length, mask_length)
    np.clip(patch, 0.0, 1.0)
    return patch

def mask_generation(patch=None, image_size=image_size):
    applied_patch = np.zeros(image_size)
    (_, width, height) = patch.shape

    # patch rotation
    rotation_angle = np.random.choice(4)
    for i in range(patch.shape[0]):
        patch[i] = np.rot90(patch[i], rotation_angle) 

    # patch location
    x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])

    for i in range(3):
        applied_patch[i, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch[i,:,:]

    mask = applied_patch.copy()
    mask[mask != 0] = 1.0

    # mask = np.clip(mask, 0.0, 1.0)
    # applied_patch = np.clip(applied_patch, 0.0, 1.0)
    return patch, applied_patch, mask, x_location, y_location



def load_surrogate_model():
    """ Load white-box and black-box models

    :return:
        face recognition and attribute recognition models
    """

    # Load pretrain white-box FR surrogate model
    fr_model = m.IR_152((112, 112))
    fr_model.load_state_dict(torch.load('./models/ir152.pth'))
    fr_model.to(device)
    fr_model.eval()

    # Load pretrain white-box AR surrogate model
    ar_model = m.IR_152_attr_all()
    ar_model.load_state_dict(torch.load('./models/ir152_ar.pth'))
    ar_model.to(device)
    ar_model.eval()

    return fr_model, ar_model

'''
    Obtain intermediate features by hooker
'''
layer_name = "ir_152.body.49"
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

gouts = []
def backward_hook(module, gin, gout):
    gouts.append(gout[0].data)
    return gin

def infer_fr_model(attack_img, victim_img, fr_model):
    """ Face recognition inference

    :param attack_img:
            attacker face image
    :param victim_img:
            victim face image
    :param fr_model:
            face recognition model
    :return:
        feature representations for the attacker and victim face images
    """
    attack_img_feat = fr_model(attack_img)
    victim_img_feat = fr_model(victim_img)
    return attack_img_feat, victim_img_feat

def infer_ar_model(attack_img, victim_img, ar_model):
    """ Face attribute recognition inference

    Args:
        :param attack_img:
            attacker face image
        :param victim_img:
            victim face image
        :param ar_model:
            attribute recognition model

    :return:
        intermediate feature representations for the attacker and victim face images
    """

    ar_model(attack_img)
    attack_img_mid_feat = activation[layer_name].clone()
    attack_img_mid_feat = torch.flatten(attack_img_mid_feat)
    attack_img_mid_feat = attack_img_mid_feat.expand(1, attack_img_mid_feat.shape[0])

    ar_model(victim_img)
    victim_img_mid_feat = activation[layer_name].clone().detach_()
    victim_img_mid_feat = torch.flatten(victim_img_mid_feat)
    victim_img_mid_feat = victim_img_mid_feat.expand(1, victim_img_mid_feat.shape[0])

    return attack_img_mid_feat, victim_img_mid_feat


def sibling_attack(patch, attack_img, victim_img, fr_model, ar_model, config):
    """ Perform Sibling-Attack

    Args:
        :param attack_img:
            attacker face image
        :param victim_img:
            victim face image
        :param fr_model:
            face recognition model
        :param ar_model:
            attribute recognition model
        :param config:
            attacking configurations

    :return:
        adversarial face image
    """
    epochs = config.attack['outer_loops'] # общие все итерации
    alpha = config.attack['alpha'] # learning rate
    eps = config.attack['eps'] # колебания добавки
    INNER_MAX_EPOCH = config.attack['inner_loops'] # внутренняя стабилизация (N из статьи)
    magic = config.attack['gamma']

    for layer in list(ar_model.named_modules()):
        if layer[0] == layer_name:
            fw_hook = layer[1].register_forward_hook(get_activation(layer_name))
            bw_hook = layer[1].register_backward_hook(backward_hook)

    ori_attack_img = attack_img.clone() # сохраняем чистое изображение рандомного человека из обучающего датасета

    for i in range(1, epochs+1):
        patch, applied_patch, mask, x_location, y_location = mask_generation(patch, image_size=(3, 112, 112))

        # patch = np.zeros((3, patch1.shape[2], patch1.shape[2]))
        patch = applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
        patch = torch.tensor(patch, requires_grad=True, device=device, dtype=torch.float32)
        applied_patch1 = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)

        # patch = patch.unsqueeze(0)
        # patch, df_size = resize_patch(patch, df_size, flag_resize)
        # patch = patch.squeeze(0)
        # applied_patch1, mask, x_location, y_location = mask_applied_patch_after_resize(patch)

        # проверить совместимость типов для attack_img
        attack_img_per = torch.mul(mask.type(torch.FloatTensor), applied_patch1.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), attack_img.type(torch.FloatTensor))
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()
        im = attack_img_per.squeeze().cpu().numpy().transpose(1, 2, 0)
        ax.imshow(im)
        plt.show()

        pre = time.time()
        if i % 2 == 0:
            # первая ветка
            INNER_LR = 1.0 / 255.0 * magic
            attack_img_tmp = attack_img_per.clone() # копируют изображение на входе
            attack_img_tmp_list = []

            for j in range(INNER_MAX_EPOCH):
                attack_img_tmp.requires_grad = True
                attack_img_feat, victim_img_feat = infer_fr_model(attack_img_tmp, victim_img, fr_model) # находят признаки для скопированного входного изображения
                fr_adv_loss = 1 - cos_simi(attack_img_feat, victim_img_feat) # считают лосс для задачи FR
                fr_model.zero_grad()
                fr_adv_loss.backward()
                sign_grad = attack_img_tmp.grad.sign()

                # core of PGD algorithm
                # обновление патча
                applied_patch1 = - 1.0 * INNER_LR * sign_grad + applied_patch1.type(torch.FloatTensor)
                applied_patch1 = torch.clamp(applied_patch1, min=0, max=1)
                applied_patch1 = applied_patch1.cpu().numpy()
                patch = applied_patch1[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
                patch = torch.tensor(patch, requires_grad=True, device=device, dtype=torch.float32)
                applied_patch1 = torch.from_numpy(applied_patch1)

                attack_img_per = torch.mul(mask.type(torch.FloatTensor), applied_patch1.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
                attack_img_per = torch.clamp(attack_img_per, min=0, max=1)
                attack_img_per = attack_img_per.to(device)

                # adv_img = attack_img_tmp - 1.0 * INNER_LR * sign_grad # обновляют скопированное входное изображение 
                # eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps) # находят добавку, вычитая из скопированного входного изображения обновленное входное изображение 
                
                adv_img = torch.clamp(attack_img_per, min=-1, max=1).detach_() # теперь adv_img это чистое входное изображение + найденная здесь добавка
                attack_img_tmp_list.append(adv_img) # новое adv_img добавляем в лист
                attack_img_tmp = adv_img.clone() # тепрь мы признаки будем искать от чистого изображения + найденная на этой внутренней эпохе итерации

            while gouts:
                tensor = gouts.pop()
                tensor.detach_()

            AR_grad_temp_list = []
            for attack_img_tmp in attack_img_tmp_list: # из собранного нами листа с чистыми изображениями + на каждой внутренней эпохе найденной добавкой
                attack_img_tmp.requires_grad = True
                attack_img_mid_feat, victim_img_mid_feat = infer_ar_model(attack_img_tmp, victim_img, ar_model)  #находим признаки уже для AR
                ar_adv_loss = 1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)  # считаем лосс уже для AR
                ar_model.zero_grad()

                ar_adv_loss.backward()
                grad = attack_img_tmp.grad
                AR_grad_temp_list.append(grad.clone())  # сохраняем градиенты для AR
                attack_img_tmp.detach_()

            aggr_grad_pic = torch.zeros_like(attack_img_per)
            for AR_grad_temp in AR_grad_temp_list:
                aggr_grad_pic += AR_grad_temp

            # use aggregrated gradients
            attack_img = attack_img_tmp_list[-1].clone() # берем последнее изображение из листа (чистое изображение с последне полученной добавкой )
            attack_img.requires_grad = True
            attack_img_mid_feat, victim_img_mid_feat = infer_ar_model(attack_img, victim_img, ar_model)
            ar_adv_loss = 1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)

            ar_model.zero_grad()
            ar_adv_loss.backward()

            w = 0.0001
            sign_grad = (attack_img.grad + w * aggr_grad_pic).sign()

            # core of PGD algorithm
            # use 1-step FR adv example as mid
             
            applied_patch1 = - magic * alpha * sign_grad + applied_patch1.type(torch.FloatTensor)
            applied_patch1 = torch.clamp(applied_patch1, min=0, max=1)
            applied_patch1 = applied_patch1.cpu().numpy()
            patch = applied_patch1[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
            patch = torch.tensor(patch, requires_grad=True, device=device, dtype=torch.float32)
            applied_patch1 = torch.from_numpy(applied_patch1)

            attack_img_per = torch.mul(mask.type(torch.FloatTensor), applied_patch1.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            attack_img_per = torch.clamp(attack_img_per, min=0, max=1)
            attack_img_per = attack_img_per.to(device)

            # adv_img = attack_img - magic * alpha * sign_grad
            # eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
            attack_img = torch.clamp(attack_img_per, min=-1, max=1).detach_()

            print("[Epoch-%d](FR-branch) AR loss: %f, time cost: %fs" % (i, ar_adv_loss.item(), time.time() - pre))
        else:
            INNER_LR = 1.0 / 255.0 * (1 - magic)
            attack_img_tmp = attack_img_per.clone()
            attack_img_tmp_list = []

            for j in range(INNER_MAX_EPOCH):
                attack_img_tmp.requires_grad = True
                attack_img_mid_feat, victim_img_mid_feat = infer_ar_model(attack_img_tmp, victim_img, ar_model)

                ar_adv_loss = 1 - cos_simi(attack_img_mid_feat, victim_img_mid_feat)
                ar_model.zero_grad()

                ar_adv_loss.backward()
                sign_grad = attack_img_tmp.grad.sign()

                # core of PGD algorithm
                # обновление патча
                applied_patch1 = - 1.0 * INNER_LR * sign_grad + applied_patch1.type(torch.FloatTensor)
                applied_patch1 = torch.clamp(applied_patch1, min=0, max=1)
                applied_patch1 = applied_patch1.cpu().numpy()
                patch = applied_patch1[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
                patch = torch.tensor(patch, requires_grad=True, device=device, dtype=torch.float32)
                applied_patch1 = torch.from_numpy(applied_patch1)

                attack_img_per = torch.mul(mask.type(torch.FloatTensor), applied_patch1.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
                attack_img_per = torch.clamp(attack_img_per, min=0, max=1)
                attack_img_per = attack_img_per.to(device)

                # adv_img = attack_img_tmp - 1.0 * INNER_LR * sign_grad
                # eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
                adv_img = torch.clamp(ori_attack_img_per, min=-1, max=1).detach_()
                attack_img_tmp_list.append(adv_img)
                attack_img_tmp = adv_img.clone()

            while gouts:
                tensor = gouts.pop()
                tensor.detach_()

            FR_grad_temp_list = []
            for attack_img_tmp in attack_img_tmp_list:
                attack_img_tmp.requires_grad = True
                attack_img_feat, victim_img_feat = infer_fr_model(attack_img_tmp, victim_img, fr_model)
                fr_adv_loss = 1 - cos_simi(attack_img_feat, victim_img_feat)
                fr_model.zero_grad()
                fr_adv_loss.backward()

                grad = attack_img_tmp.grad
                FR_grad_temp_list.append(grad.clone())
                attack_img_tmp.detach_()

            aggr_grad_pic = torch.zeros_like(attack_img_per)
            for FR_grad_temp in FR_grad_temp_list:
                aggr_grad_pic += FR_grad_temp

            # use aggregrated gradients
            attack_img = attack_img_tmp_list[-1].clone()
            attack_img.requires_grad = True
            attack_img_feat, victim_img_feat = infer_fr_model(attack_img, victim_img, fr_model)
            fr_adv_loss = 1 - cos_simi(attack_img_feat, victim_img_feat)

            fr_model.zero_grad()
            fr_adv_loss.backward()

            w = 0.0001
            sign_grad = (attack_img.grad + w * aggr_grad_pic).sign()

            # core of PGD algorithm
            # use 1-step FR adv example as mid

            applied_patch1 = - (1 - magic) * alpha * sign_grad + applied_patch1.type(torch.FloatTensor)
            applied_patch1 = torch.clamp(applied_patch1, min=0, max=1)
            applied_patch1 = applied_patch1.cpu().numpy()
            patch = applied_patch1[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
            patch = torch.tensor(patch, requires_grad=True, device=device, dtype=torch.float32)
            applied_patch1 = torch.from_numpy(applied_patch1)

            attack_img_per = torch.mul(mask.type(torch.FloatTensor), applied_patch1.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            attack_img_per = torch.clamp(attack_img_per, min=0, max=1)
            attack_img_per = attack_img_per.to(device)


            # adv_img = attack_img - (1 - magic) * alpha * sign_grad
            # eta = torch.clamp(adv_img - ori_attack_img, min=-eps, max=eps)
            attack_img = torch.clamp(attack_img_per, min=-1, max=1).detach_()

            print("[Epoch-%d](AR-branch) FR loss: %f, time cost: %fs" % (i, fr_adv_loss.item(), time.time() - pre))

    return attack_img


if __name__ == '__main__':
    config = OmegaConf.load('./configs/config.yaml')
    gpu = config.attack['gpu']
    dataset_name = config.dataset['dataset_name']
    device = torch.device('cuda:' + str(gpu))

    patch = patch_initialization(image_size=(3, 112, 112))

    fr_model, ar_model = load_surrogate_model()
    attack_img_paths, victim_img_paths = obtain_attacker_victim(config)
    pairs_num = len(attack_img_paths) * len(victim_img_paths)
    for attack_img_path in attack_img_paths:
        for victim_img_path in victim_img_paths:
            print(attack_img_path, "========", victim_img_path)

            attack_img = load_img(attack_img_path, config).to(device)
            victim_img = load_img(victim_img_path, config).to(device)

            # Perform Sibling-Attack
            adv_attack_img = sibling_attack(patch, attack_img, victim_img, fr_model, ar_model, config)

            save_dir = './' + dataset_name +  '_results_adv_images/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = save_dir + victim_img_path.split('/')[-1].split('.')[0] + '+' +\
                        attack_img_path.split('/')[-1].split('.')[0] + '.png'
            save_adv_img(patch.cpu(), save_path, config)
            print("Save patch to - ", save_path)

