#!/usr/bin/env python3

import sys, getopt
sys.path.append("/groups/hongxuding/software/Python_Lib")
import torch
from vae import VAE
from utils import UnsupervisedDataset, KFoldTorch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import umap

job = ''
mode= ''
data_path = ''
model_path= ''
gene_path = ''
tf_path = ''
out_dir = ''
z_path = ''
depth = 3
beta = 0.00005
alpha = 0.5
cv = 10
lr = 1e-3
random_seed=1
# added path and mode flags (were not here previously)
opt, args = getopt.getopt(sys.argv[1:], 'j:m:i:g:t:p:o:n:b:a:f:r:s', ['job=',
                                                                  'mode=',
                                                                  'data_path=',
                                                                  'model_path=',
                                                                  'gene_path=',
                                                                  'tf_path=',
                                                                  'out_dir=',
                                                                  'depth=',
                                                                  'beta=',
                                                                  'alpha=',
                                                                  'cv=',
                                                                  'lr=',
                                                                  'random_seed=',
                                                                  'z_path='])

for o, a in opt:
    if o in ('-j', '--job'):
        job = a
    elif o in ('-i', '--data_path'):
        data_path = a
    elif o in ('-m', '--mode'): # added by Ramon
        mode = a
    elif o in ('-p','--model_path'): # added by Ramon
        model_path = a
    elif o in ('-g', '--gene_path'):
        gene_path = a
    elif o in ('-t', '--tf_path'):
        tf_path = a
    elif o in ('-o', '--out_dir'):
        out_dir = a
    elif o in ('-n', '--depth'):
        depth = int(a)
    elif o in ('-b', '--beta'):
        beta = float(a)
    elif o in ('-a', '--alpha'):
        alpha = float(a)
    elif o in ('-f', '--cv'):
        cv = int(a)
    elif o in ('-r', '--lr'):
        lr = float(a)
    elif o in ('-s', '--random_seed'):
        random_seed = int(a)
    elif o in ('-z', '--z_path'):
        z_path = a     

print("-" * 50)
print("Job Information for VAE Training")
print("-" * 50)
print("job name            : %s" % job)
print("tab-delimited data  : %s" % data_path)
print("gene list           : %s" % gene_path)
print("TF list             : %s" % tf_path)
print("output directory    : %s" % out_dir)
print("number of layers    : %s" % depth)
print("beta                : %s" % beta)
print("alpha               : %s" % alpha)
print("cv folds            : %s" % cv)
print("learning rate       : %s" % lr)
print("seed                : %s" % random_seed)
print("\n")

# Set model
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev, flush=True)
    
# Read tf and data
with open(gene_path) as f:
    genes = f.read().splitlines()
with open(tf_path) as f:
    tfs = f.read().splitlines()
mask = np.isin(np.array(genes), np.array(tfs))
train_data = np.genfromtxt(data_path, delimiter = " ")
train_ds = torch.Tensor(train_data)
train_ds = UnsupervisedDataset(train_ds, targets=torch.zeros(train_data.shape[0]))

# All below added by Ramon (using run_vae.py as template)
if job == 'train':
    print(f'Training model...', flush=True)

    # Initialize CV
    kfold = KFoldTorch(cv=cv, n_epochs=2000, lr=lr, train_p=25, test_p=25, num_workers=0, save_all=True, save_best=False, path_dir=out_dir, model_prefix=job)
    dict_params = {'mask_tf':mask, 'depth':depth, 'init_w':True, 'beta':beta, 'alpha':alpha, 'dropout':0.3, 'path_model':None, 'device':dev}
    kfold.train_kfold(VAE, dict_params, train_ds, batch_size=64)
    np.save(out_dir+'/'+job+'_'+str(cv)+'CV_vae.npy', kfold.cv_res_dict)

    # Train history
    res = np.load(out_dir+'/'+job+'_'+str(cv)+'CV_vae.npy', allow_pickle=True).item()
    colors = cm.jet(np.linspace(0, 1, cv))
    for f in range(0, cv):
        loss = res[f].get('history').get('train_loss')
        epoch = [i for i in range(0, len(loss))]
        plt.plot(epoch, loss, 'o', color=colors[f])
        loss = res[f].get('history').get('valid_loss')
        epoch = [i for i in range(0, len(loss))]
        plt.plot(epoch, loss, 'x', color=colors[f])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(handles=[mlines.Line2D([], [], marker='o', linestyle='None', color='black', label='train'),
                    mlines.Line2D([], [], marker='x', linestyle='None', color='black', label='valid')])
    plt.savefig(out_dir+'/history.pdf') 

elif job == 'encode':
    print(f'Encoding model...', flush=True)

    if not model_path:
        print(f'No model path found. Please specify -p')
        sys.exit(1)
    
    vae = VAE(mask_tf=mask, depth=depth)
    vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    vae.eval()
    
    if mode == "predict":
        run_data = np.genfromtxt(data_path, delimiter = " ")

        with torch.no_grad():
            run_data_rec, z, mu, logvar = vae.forward(torch.Tensor(run_data))
        # convert results back to numpy
        run_data_rec = run_data_rec.detach().numpy()
        z = z.detach().numpy()
        
        # run umap
        reducer = umap.UMAP(random_state=random_seed, min_dist=0.5, n_neighbors=15)
        umap_x = reducer.fit_transform(run_data)
        umap_x_rec = reducer.fit_transform(run_data_rec)

        np.savetxt(out_dir+'/'+job+'_'+mode+'_x_rec_data.csv.gz', run_data_rec, delimiter=" ")
        np.savetxt(out_dir+'/'+job+'_'+mode+'_z_reparametrization.csv.gz', z, delimiter=" ")
        np.savetxt(out_dir+'/'+job+'_'+mode+'_x_umap.csv.gz', umap_x, delimiter=" ")
        np.savetxt(out_dir+'/'+job+'_'+mode+'_x_rec_umap.csv.gz', umap_x_rec, delimiter=" ")
    
    elif mode == "generate":

        if not z_path:
            print(f"ERROR: z_path needed for data generation. Please input z_path")
            sys.exit(1)

        run_data = np.genfromtxt(z_path, delimiter = " ")
        with torch.no_grad():
            run_data_gen = vae.decode(torch.Tensor(run_data))
            
        run_data_gen = run_data_gen.detach().numpy()

        reducer = umap.UMAP(random_state=random_seed, min_dist=0.5, n_neighbors=15)
        umap_x_gen = reducer.fit_transform(run_data_gen)
    
        np.savetxt(out_dir+'/'+job+'_'+mode+'_x_rec_data.csv.gz', run_data_gen, delimiter=" ")
        np.savetxt(out_dir+'/'+job+'_'+mode+'_x_rec_umap.csv.gz', umap_x_gen, delimiter=" ")

    else:
        print(f"{mode} not recognised as mode. Mode should be 'generate' or 'predict'")
        sys.exit(1)

else:
    print(f"{job} not recognised as job. Job should be 'train' or 'encode'")
    sys.exit(1)  