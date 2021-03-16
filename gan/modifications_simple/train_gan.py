import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle

from torch.utils.tensorboard import SummaryWriter

seed = 314
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True



def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=12)  # size of history steps in frames
    parser.add_argument('--preds',type=int,default=8)  # size of predicted trajectory in frames
    parser.add_argument('--point_dim',type=int,default=3)  # number of dimensions (x,y,z) is 3
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=1500)
    parser.add_argument('--batch_size',type=int,default=70)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--gen_pth', type=str)
    parser.add_argument('--crit_pth', type=str)
    parser.add_argument('--visual_step', type=int, default=10)
    parser.add_argument('--grad_penality', type=float, default=10)
    parser.add_argument('--crit_repeats', type=int, default=5)
    parser.add_argument('--lambda_recon', type=float, default=0.1)
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--stop_recon', type=int, default=2)




    args=parser.parse_args()
    model_name=args.name
    def mkdir(path):
        try:
            os.mkdir(path)
        except:
            pass
    paths = ['models', 'models/gen', 'models/crit', 'models/gan', f'models/gen/{args.name}', f'models/crit/{args.name}',
             f'models/gan/{args.name}', 'output', 'output/gan', f'output/gan/{args.name}']
    for path in paths: mkdir(path)

    log=SummaryWriter('logs/gan_%s'%model_name)

    log.add_scalar('eval/mad', 0, 0)
    log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## creation of the dataloaders for train and validation
    if args.val_size==0:
        train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
        val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                                    args.preds, delim=args.delim, train=False,
                                                                    verbose=args.verbose)
    else:
        train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs,
                                                              args.preds, delim=args.delim, train=True,
                                                              verbose=args.verbose)

    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)




    # import individual_TF
    # model=individual_TF.IndividualTF(3, 4, 4, N=args.layers,
    #                d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    # optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
    #                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch=0


    #mean=train_dataset[:]['src'][:,1:,2:4].mean((0,1))
    mean=torch.cat((train_dataset[:]['src'][:,1:,-3:],train_dataset[:]['trg'][:,:,-3:]),1).mean((0,1))
    #std=train_dataset[:]['src'][:,1:,2:4].std((0,1))
    std=torch.cat((train_dataset[:]['src'][:,1:,-3:],train_dataset[:]['trg'][:,:,-3:]),1).std((0,1))
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['src'][ind, 1:, -3:], train_dataset[:]['trg'][ind, :, -3:]), 1).mean((0, 1)))
        stds.append(
            torch.cat((train_dataset[:]['src'][ind, 1:, -3:], train_dataset[:]['trg'][ind, :, -3:]), 1).std((0, 1)))
    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)

    scipy.io.savemat(f'models/gan/{args.name}/norm.mat',{'mean':mean.cpu().numpy(),'std':std.cpu().numpy()})

    from gan import Generator, Critic, get_gradient, gradient_penalty, get_crit_loss, get_gen_loss
    from tqdm import tqdm

    c_lambda = args.grad_penality
    crit_repeats = args.crit_repeats

    gen = Generator(args.obs-1, args.preds, args.point_dim, args.point_dim, args.point_dim, z_dim=args.z_dim, N=args.layers,
                   d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout, device=device).to(device)
    gen_opt = torch.optim.Adam(gen.parameters())
    # gen_opt = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
    #                     torch.optim.Adam(gen.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    crit = Critic(args.point_dim, args.obs-1 + args.preds, N=args.layers, d_model=args.emb_size, d_ff=2048,
                  h=args.heads, dropout=args.dropout, device=device).to(device)
    crit_opt = torch.optim.Adam(crit.parameters())
    # crit_opt = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
    #                     torch.optim.Adam(crit.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    if args.resume_train:
        gen.load_state_dict(torch.load(f'models/gen/{args.name}/{args.gen_pth}'))
        crit.load_state_dict(torch.load(f'models/crit/{args.name}/{args.crit_pth}'))


    cur_step = -1
    for epoch in range(args.max_epoch):
        gen.train()
        crit.train()
        for id_b, batch in enumerate(tqdm(tr_dl, desc=f"Epoch {epoch}")):
            cur_step += 1
            src = (batch['src'][:,1:,-3:].to(device)-mean.to(device))/std.to(device)
            tgt = (batch['trg'][:,:,-3:].to(device)-mean.to(device))/std.to(device)
            batch_size = src.shape[0]

            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                ### Update critic ###
                crit_opt.zero_grad()
                fake_noise = gen.sample_noise(batch_size)
                fake = gen(src, fake_noise)
                fake_seq = torch.cat((src, fake.detach()), dim=1)
                real_seq = torch.cat((src, tgt), dim=1)
                crit_fake_pred = crit(fake_seq)
                crit_real_pred = crit(real_seq)

                crit_loss = get_crit_loss(crit, src, tgt, fake.detach(), crit_fake_pred, crit_real_pred, c_lambda,
                                          args.lambda_recon if epoch < args.stop_recon else 0.)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                crit_loss.backward(retain_graph=True)
                crit_opt.step()
            log.add_scalar('Loss/train/crit', mean_iteration_critic_loss, cur_step)

            ### Update generator ###
            gen_opt.zero_grad()
            fake_noise_2 = gen.sample_noise(batch_size)
            fake_2 = gen(src, fake_noise_2)
            fake_2_seq = torch.cat((src, fake_2), dim=1)
            crit_fake_pred = crit(fake_2_seq)

            gen_loss = get_gen_loss(crit_fake_pred, fake_2, tgt, args.lambda_recon if epoch < args.stop_recon else 0.)
            gen_loss.backward()
            gen_opt.step()
            log.add_scalar('Loss/train/gen', gen_loss.item(), cur_step)


            if cur_step % args.visual_step== 0:
                scipy.io.savemat(f"output/gan/{args.name}/step_{cur_step:05}.mat",
                                 {'input': batch['src'][:, 1:, :3].detach().cpu().numpy(),
                                  'gt': batch['trg'][:, :, :3].detach().cpu().numpy(),
                                  'pr': (fake_2 * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1)
                                        + batch['src'][:, -1:, :3].cpu().numpy()})

        if epoch % args.save_step == 0:
            torch.save(gen.state_dict(), f'models/gen/{args.name}/{cur_step:05}.pth')
            torch.save(crit.state_dict(), f'models/crit/{args.name}/{cur_step:05}.pth')

if __name__=='__main__':
    main()
