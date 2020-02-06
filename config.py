# -*- coding : utf-8 -*-
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--real_name', type=str, default="train", help="the name of real images")
parser.add_argument('--cartoon_name', type=str, default="Demon_Slayer", help="the name of cartoon images")
parser.add_argument('--need_pretrain', type=bool, default=True, help="to pretrain or not")
parser.add_argument('--rb_num', type=int, default=8, help="the number of resnet block in generator network")

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--pre_train_epoch', type=int, default=2)
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('-d_lr', type=float, default=0.001, help='learning rate of generator network')
parser.add_argument('-g_lr', type=float, default=0.001, help='learning rate of discriminal network')
parser.add_argument('-clw', type=float, default=.4, help='the weight of content loss')
parser.add_argument('-b1', type=float, default=0.5, help='beta for Adam optimizer')
parser.add_argument('-b2', type=float, default=0.999, help='beta for Adam optimizer')

parser.add_argument('--model_state_dir', type=str, default="./checkpoints", help="dirname of model state")
parser.add_argument('--results_save_dir', type=str, default="./outputs", help="dirname of model outputs")

args = parser.parse_args()