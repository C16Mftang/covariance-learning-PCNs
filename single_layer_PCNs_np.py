import numpy as np
import matplotlib.pyplot as plt

# training parameters
sample_size = 500
relaxation_lr = 0.1
training_iters = 100
relaxation_iters = 100
fix_intact = True
dim = 2
artificial_data = True

if artificial_data:
    # simulation data
    mean = np.array([0,0])
    cov = np.array([[1,0.5],
                    [0.5,1]])
    np.random.seed(10)
    X = np.random.multivariate_normal(mean, cov, size=sample_size)
    X_c = X * np.concatenate([np.ones((sample_size,1))]+[np.zeros((sample_size,1))]*(dim-1), axis=1)
    update_mask = np.concatenate([np.zeros((sample_size,1))]+[np.ones((sample_size,1))]*(dim-1), axis=1)

else:
    cover_size = 10
    X = np.load('./data/MNIST.npy').reshape(-1, 28, 28)
    size = X.shape
    mask = np.ones_like(X).astype('float64')
    update_mask = np.zeros_like(X).astype('float64')
    update_mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] += 1
    update_mask = update_mask.reshape(-1, 784)
    mask[:, (size[1]-cover_size)//2:(size[1]+cover_size)//2, (size[2]-cover_size)//2:(size[2]+cover_size)//2] -= 1
    X_c = X * mask
    X = X.reshape(-1, 784)
    X_c = X_c.reshape(-1, 784)
    

def memory_ML():
    """relaxation using the maximum likelihood estimate of the covariance matrix"""
    ML_cov = np.cov(X.T)
    print(ML_cov.shape)
    print("Maximum likelihood estimate:\n", ML_cov)
    X_recon_ML = X_c
    for i in range(relaxation_iters):
        print('iteration:', i)
        errs_ML = np.matmul(X_recon_ML, np.linalg.inv(ML_cov).T) 
        delta = -errs_ML
        if fix_intact:
            X_recon_ML += relaxation_lr * (delta * update_mask)
        else:
            X_recon_ML += relaxation_lr * delta
    return X_recon_ML


def memory_Friston(batch_size, training_lr):
    """relaxation using Friston's update of covariance matrix"""
    ## learning the covariance
    F_cov = np.eye(dim)  # initial cov
    # rand_init = np.random.randn(2,2)
    # F_cov = np.matmul(rand_init, rand_init.T)
    mse_F = []
    covs = [F_cov]
    for i in range(training_iters):
        cov_per_epoch = np.zeros_like(F_cov)
        for batch_idx in range(0, sample_size, batch_size):
            batchX = X[batch_idx:batch_idx+batch_size]
            errsW_F = np.matmul(batchX, np.linalg.inv(F_cov).T) # N, 2
            grad_cov = 0.5 * (np.matmul(errsW_F.T, errsW_F)/batch_size - np.linalg.inv(F_cov))
            F_cov += training_lr * grad_cov
            cov_per_epoch += (F_cov / (sample_size//batch_size))
        covs.append(cov_per_epoch)

        ## relaxation using the current covariance
        X_recon_F = X_c
        for i in range(relaxation_iters):
            errsX_F = np.matmul(X_recon_F, np.linalg.inv(F_cov).T) 
            delta = -errsX_F
            if fix_intact:
                X_recon_F += relaxation_lr * (delta * update_mask)
        mse_F.append(np.mean((X[:,1] - X_recon_F[:,1])**2)) 
    # print("Learned covariance using Friston's update:\n", F_cov)

    return mse_F, F_cov, X_recon_F


def memory_Rafal(batch_size, training_lr):
    """relaxation using Rafal's update of covariance matrix"""
    ## learning the covariance
    R_cov = np.eye(dim)
    # R_cov = np.matmul(rand_init, rand_init.T)
    mse_R = []
    covs = [R_cov]
    for i in range(training_iters):
        cov_per_epoch = np.zeros_like(R_cov)
        for batch_idx in range(0, sample_size, batch_size):
            batchX = X[batch_idx:batch_idx+batch_size]
            errsW_R = np.matmul(batchX, np.linalg.inv(R_cov).T) # N, 2
            e = batchX
            delta = np.matmul(errsW_R.T, e)/batch_size - np.eye(dim)
            R_cov += training_lr * delta
            cov_per_epoch += (R_cov / (sample_size//batch_size))
        covs.append(cov_per_epoch)

        ## relaxation using the current covariance
        X_recon_R = X_c
        for i in range(relaxation_iters):
            errsX_R = np.matmul(X_recon_R, np.linalg.inv(R_cov).T) 
            delta = -errsX_R
            if fix_intact:
                X_recon_R += relaxation_lr * (delta * update_mask)
        mse_R.append(np.mean((X[:,1] - X_recon_R[:,1])**2))
    # print("Learned covariance using Rafal's update:\n", R_cov)

    return mse_R, covs, X_recon_R


def memory_rec(batch_size, training_lr, dendrite=True):
    # relaxation using the learned weights by recurrent PCN
    ## learning the weights
    # weights = np.random.normal(0, 0.05, (2,2))
    weights = np.zeros((2,2))
    mse_rec = []
    for i in range(training_iters):
        # print('training iter:', i)
        for batch_idx in range(0, sample_size, batch_size):
            batchX = X[batch_idx:batch_idx+batch_size]
            preds = np.matmul(batchX, weights.T)
            errsW_rec = batchX - preds
            grad_w = np.matmul(errsW_rec.T, batchX)
            np.fill_diagonal(grad_w, 0)
            weights += training_lr * grad_w

        ## relaxation using the current weights
        X_recon_rec = X_c
        for i in range(relaxation_iters):
            errsX_rec = X_recon_rec - np.matmul(X_recon_rec, weights.T)
            if dendrite:
                delta = -errsX_rec
            else:
                delta = -errsX_rec + np.matmul(errsX_rec, weights)
            X_recon_rec += relaxation_lr * (delta * update_mask)
        mse_rec.append(np.mean((X[:,1] - X_recon_rec[:,1])**2))
    # print("Learned weight matrix by rec PCN:\n", weights)

    return mse_rec, weights, X_recon_rec

def cosine_similarity_weights(batch_size, lr):
    mse_F, covs_F = memory_Friston(batch_size, lr[0])
    mse_R, covs_R = memory_Rafal(batch_size, lr[1])
    cos_sims = []
    for i in range(training_iters):
        a = covs_F[i].flatten()
        b = covs_R[i].flatten()
        cos_sim = np.inner(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cos_sims.append(cos_sim)
    return cos_sims, mse_F, mse_R

def best_lrs(batch_size):
    training_lrs = np.arange(1e-3, 1.05e-2, 5e-4)

    best_lrs_by_auc = []
    best_lrs_by_lowest = []
    for method in ['Friston', 'Rafal', 'Rec']:
        aucs = []
        lowests = []
        for training_lr in training_lrs:
            if method == 'Friston':
                mse, _ = memory_Friston(batch_size, training_lr)
            elif method == 'Rafal':
                mse, _ = memory_Rafal(batch_size, training_lr)
            else:
                mse, _ = memory_rec(batch_size, training_lr/10)
            aucs.append(sum(mse))
            lowests.append(min(mse))

        best_lr_by_auc = training_lrs[aucs.index(min(aucs))]
        best_lr_by_lowest = training_lrs[lowests.index(min(lowests))]

        print(method)
        print('Training lr with the smallest area under the learning curve:', best_lr_by_auc)
        print('Training lr with the smallest lowest point:', best_lr_by_lowest)

        best_lrs_by_auc.append(best_lr_by_auc)
        best_lrs_by_lowest.append(best_lr_by_lowest)

    fig, ax = plt.subplots(1, 2, sharey=False, figsize=(12,5))
    for method in ['Friston', 'Rafal', 'Rec']:
        if method == 'Friston':
            best_mse_by_auc, covs_F1 = memory_Friston(batch_size, best_lrs_by_auc[0])
            # best_mse_by_auc, covs_F1 = memory_Friston(batch_size, 0.0045)
            best_mse_by_lowest, covs_F2 = memory_Friston(batch_size, best_lrs_by_lowest[0])
        elif method == 'Rafal':
            best_mse_by_auc, covs_R1 = memory_Rafal(batch_size, best_lrs_by_auc[1])
            # best_mse_by_auc, covs_R1 = memory_Rafal(batch_size, 0.002)
            best_mse_by_lowest, covs_R2 = memory_Rafal(batch_size, best_lrs_by_lowest[1])
        else:
            best_mse_by_auc, w1 = memory_rec(batch_size, best_lrs_by_auc[2]/10)
            # best_mse_by_auc, w1 = memory_rec(batch_size, 0.0065/10)
            best_mse_by_lowest, w2 = memory_rec(batch_size, best_lrs_by_lowest[2]/10)


        ax[0].plot(best_mse_by_auc, label=method)
        ax[1].plot(best_mse_by_lowest, label=method)

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('MSE')
    ax[0].set_title('Best lr by AUC')
    ax[1].set_title('Best lr by lowest point')
    plt.show()

    print('Friston\'s estimate:\n', covs_F1[-1], '\n', covs_F2[-1])
    print('Rafal\'s estimate:\n', covs_R1[-1], '\n', covs_R2[-1])
    print('Rec nets estimtae:\n', w1, '\n', w2)

def plot_smth():
    mse_F, covs_F, X_recon_F = memory_Friston(1, 0.001)
    mse_R, covs_R, X_recon_R = memory_Rafal(1, 0.001)
    mse_rec, covs_rec, X_recon_rec = memory_rec(1, 0.001)
    ML_cov = np.cov(X.T)
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].scatter(X[:,0], X[:,1], alpha=0.3, color='gray', label='Data')
    ax[0].plot(X[:,0], X_recon_F[:,1], label='Friston')
    ax[0].plot(X[:,0], X_recon_R[:,1], label='Rafal')
    ax[0].plot(X[:,0], X_recon_rec[:,1], label='Recurrent')
    ax[0].plot(X[:,0], (ML_cov[0,1]/ML_cov[1,1])*X[:,0], label=r'$\Sigma_{12}/\Sigma_{22}$')
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].legend()
    ax[0].set_title(r'Linear prediction of $x_2$ from $x_1$')

    ax[1].plot(mse_F, label='Friston')
    ax[1].plot(mse_R, label='Rafal')
    ax[1].plot(mse_rec, label='Recurrent')
    ax[1].set_xlabel('Learning epochs')
    ax[1].set_ylabel('MSE')
    ax[1].set_title(r'MSE($\Vert X_{original} - X_{retrieved} \Vert^2$)')
    ax[1].legend()
    # plt.savefig('./figs/seed20', dpi=200)
    plt.show()

def plot_equivalence():
    mse_F, covs_F, X_recon_F = memory_Friston(1, 0.001)
    mse_rec, covs_rec, X_recon_rec = memory_rec(1, 0.001, dendrite=False)
    mse_d, covs_d, X_recon_d = memory_rec(1, 0.001, dendrite=True)
    ML_cov = np.cov(X.T)
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.scatter(X[:,0], X[:,1], alpha=0.4, color='gray', label='Data')
    ax.plot(X[:,0], X_recon_F[:,1], label='Explicit model', lw=3.5)
    ax.plot(X[:,0], X_recon_rec[:,1], label='Implicit model', lw=3)
    ax.plot(X[:,0], X_recon_d[:,1], label='Dendritic model', lw=2.5)
    ax.plot(X[:,0], (ML_cov[0,1]/ML_cov[1,1])*X[:,0], label=r'$\Sigma_{21}/\Sigma_{11}$')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    # ax.set_title(r'Linear prediction of $x_2$ from $x_1$')
    plt.savefig('./figs/gaussian2', dpi=400)
    plt.show()

def plot_mse():
    mse_F, covs_F, X_recon_F = memory_Friston(1, 0.001)
    mse_R, covs_R, X_recon_R = memory_Rafal(1, 0.001)
    mse_rec, covs_rec, X_recon_rec = memory_rec(1, 0.001)

    plt.plot(mse_F, label='Friston')
    plt.plot(mse_R, label='Rafal')
    plt.plot(mse_rec, label='Recurrent')
    plt.xlabel('Learning epochs')
    plt.ylabel('MSE')
    plt.title(r'MSE($\Vert X_{original} - X_{retrieved} \Vert^2$)')
    plt.legend()
    plt.savefig('./figs/unstable_mse', dpi=200)
    plt.show()

def plot_mats():
    mse_F, ws_F, X_recon_F = memory_Friston(1, 0.001)
    mse_rec, ws_rec, X_recon_rec = memory_rec(1, 0.001, dendrite=False)
    mse_d, ws_d, X_recon_d = memory_rec(1, 0.001, dendrite=True)
    ML_cov = np.cov(X.T)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ML_cov, vmin=0, vmax=1)
    ax[1].imshow(ws_F, vmin=0, vmax=1)
    im = ax[2].imshow(ws_rec, vmin=0, vmax=1)
    for i in range(3):
        ax[i].axis('off') 
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3)
    plt.savefig('./figs/covmats_gaussian2', dpi=400)
    plt.show()


def main():
    plot_equivalence()
    

if __name__ == '__main__':
    main()