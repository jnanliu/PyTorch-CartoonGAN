# -*- coding : utf-8 -*-
import os
import time
import torch
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
from models import generator_network, discriminator_network, vgg19
from loss import adversarial_loss, content_loss

args = config.args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define dataloader
real_dataloader = utils.load_data(args.real_name, args.batch_size, True, True)
cart_dataloader = utils.load_data(args.cartoon_name, args.batch_size, True, True)
test_dataloader = utils.load_data("test", 1, True, True)

# define model
gen_net = generator_network.GeneratorNetwork(args.rb_num).to(device)
dis_net = discriminator_network.DiscriminatorNetwork().to(device)
vgg_net = vgg19.ExtractFeaturesNetwork().to(device)

# define loss
adv_loss = adversarial_loss.AdversarialLoss(1).to(device)
con_loss = content_loss.ContentLoss(args.clw).to(device)

# define optimizer
gen_optimizer = torch.optim.Adam(gen_net.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
pre_gen_optimizer = torch.optim.Adam(gen_net.parameters(), lr=2e-4, betas=(args.b1, args.b2))
dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr=args.d_lr, betas=(args.b1, args.b2))

# define lr_scheduler
gen_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=gen_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
dis_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=dis_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

# path of model state
generator_state_path = os.path.join(args.model_state_dir, "{}_gm_state.pth".format(args.cartoon_name))
discriminator_state_path = os.path.join(args.model_state_dir, "{}_dm_state.pth".format(args.cartoon_name))

# path of outputs
pretrain_outputs_path = os.path.join(args.results_save_dir, args.cartoon_name + "/pretrain")
if not os.path.exists(pretrain_outputs_path) :
    os.makedirs(pretrain_outputs_path)
outputs_path = os.path.join(args.results_save_dir, args.cartoon_name + "/GAN")
if not os.path.exists(outputs_path) :
    os.makedirs(outputs_path)

networks = {"gen" : gen_net, "dis" : dis_net}
state_paths = {"gen" : generator_state_path, "dis" : discriminator_state_path}

def save_pth(key_net) :
    utils.print_info("Saving {}_net state.".format(key_net))
    state = networks[key_net].state_dict()
    torch.save(state, state_paths[key_net])
    utils.print_info("Save {}_net state done.".format(key_net))

def generator_image(save_path, gen_nums) :
    utils.print_info("Generating images...")
    with torch.no_grad() :
        gen_net.eval()
        for idx, (x, _) in enumerate(test_dataloader) :
            x = x.to(device)
            gen_img = gen_net(x)
            cat_img = torch.cat((x[0], gen_img[0]), 2)
            plt.imsave(save_path + "/{}.png".format(idx),
                       ((cat_img.cpu().numpy().transpose(1, 2, 0) + 1.) / 2. * 255.).astype(np.int8))
            idx += 1
            if idx >= gen_nums :
                break
    utils.print_info("Generate {} images, saving to {}".format(gen_nums, save_path))

def pretrain() :
    utils.print_info("Start pretraining...")
    for epoch in range(args.pre_train_epoch) :
        start_time, loss, n = time.time(), 0., 0
        for x, _ in real_dataloader :
            x = x.to(device)

            pre_gen_optimizer.zero_grad()
            x_features = vgg_net(x)
            gen_x = gen_net(x)
            gen_x_features = vgg_net(gen_x)

            c_loss = con_loss(gen_x_features, x_features.detach())
            c_loss.backward()
            pre_gen_optimizer.step()

            loss += c_loss.item()
            n += 1

        print("Epoch: {:3d}, con_loss: {:3.3f}, time: {:3.3f} secs".format(epoch + 1, loss / n, time.time() - start_time))
    generator_image(pretrain_outputs_path, 5)
    save_pth("gen")
    utils.print_info("Pretrain done.")

def train() :
    utils.print_info("Start training...")
    if os.path.exists(generator_state_path) :
        if torch.cuda.is_available() :
            gen_net.load_state_dict(torch.load(generator_state_path))
        else :
            gen_net.load_state_dict(torch.load(generator_state_path, map_location='cpu'))
    if os.path.exists(discriminator_state_path) :
        if torch.cuda.is_available() :
            dis_net.load_state_dict(torch.load(discriminator_state_path))
        else :
            dis_net.load_state_dict(torch.load(discriminator_state_path, map_location='cpu'))
    real_labels = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
    fake_labels = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
    for epoch in range(args.train_epoch) :
        start_time, adv_losses, con_losses, n = time.time(), 0., 0., 0
        for (real, _), (cart, _) in zip(real_dataloader, cart_dataloader) :
            cart_smooth = cart[:, :, :, args.input_size:]
            cart = cart[:, :, :, :args.input_size]
            real, cart, cart_smooth = real.to(device), cart.to(device), cart_smooth.to(device)

            # train dis_net
            dis_optimizer.zero_grad()

            gen_real = gen_net(real)
            dis_real = dis_net(gen_real.detach())
            real_adv_loss = adv_loss(dis_real, fake_labels)

            dis_cart = dis_net(cart)
            cart_adv_loss = adv_loss(dis_cart, real_labels)

            dis_cart_smooth = dis_net(cart_smooth)
            cart_smooth_adv_loss = adv_loss(dis_cart_smooth, fake_labels)

            D_loss = real_adv_loss + cart_adv_loss + cart_smooth_adv_loss
            D_loss.backward()
            dis_optimizer.step()

            # train gen_net
            gen_optimizer.zero_grad()

            gen_real = gen_net(real)
            dis_real = dis_net(gen_real)
            real_adv_loss = adv_loss(dis_real, real_labels)

            gen_real_features = vgg_net(gen_real)
            real_features = vgg_net(real)
            real_con_loss = con_loss(gen_real_features, real_features.detach())

            G_loss = real_adv_loss + real_con_loss
            G_loss.backward()
            gen_optimizer.step()

            gen_lr_scheduler.step(epoch)
            dis_lr_scheduler.step(epoch)

            adv_losses += D_loss.item()
            con_losses += real_con_loss.item()
            n += 1

        print("Epoch: {:3d}, adv_loss: {:3.3f}, con_loss: {:3.3f}, time: {:3.3f} secs".format(epoch + 1, adv_losses / n, con_losses / n, time.time() - start_time))

        if (epoch + 1) % 10 == 0 :
            generator_image(outputs_path, 5)
            save_pth("gen")
            save_pth("dis")
    utils.print_info("Train done.")


if __name__ == "__main__" :
    utils.print_info("Options")
    pprint(vars(args))

    if args.need_pretrain :
        pretrain()

    train()