import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
from tme2 import reshape_all


def perceptron_loss(w,X,y):
    w,X,y = reshape_all(w,X,y)
    return np.maximum(0,-y*X@w)

def perceptron_grad(w,X,y):
    w,X,y = reshape_all(w,X,y)
    grad = -y*X
    correct_mask = (y*X@w).reshape(-1) > 0
    grad[correct_mask, : ] = 0
    # TODO:  smarter way
    return grad.mean(0).reshape(-1,1)

def hinge_loss(w,X,y, alpha, lam):
    w,X,y = reshape_all(w,X,y)
    return np.maximum(0,alpha-y*X@w) + lam*(w*2).sum()

def hinge_grad(w,X,y, alpha, lam):
    w,X,y = reshape_all(w,X,y)
    grad = -y*X
    correct_mask = (y*X@w).reshape(-1) > alpha
    grad[correct_mask, : ] = 0
    # TODO:  smarter way
    return grad.mean(0).reshape(-1,1) + 2*lam*w

class Linear(object):
    def __init__(self,
                 loss=perceptron_loss,
                 loss_g=perceptron_grad,
                 phi= lambda x: x,
                 max_iter=100,eps=0.01):
        
        self.loss,self.loss_g = loss,loss_g
        self.phi = phi
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.losses, self.ws = None, None
        self.train_scores, self.test_scores = None, None
        
    def fit(self,X_train,y_train, X_test=None, y_test=None,m=None, w_init=None):
        # NOTE: don't initiate exactly at zeros 
        # - it will flatten out the gradient

        X_train = self.phi(X_train)

        n, d = X_train.shape
        indices = np.random.permutation(n)

        if m is None:
            m = n

        if w_init is None:
            self.w = np.random.randn(d,1)
        else:
            self.w = np.ones((d,1))*w_init
        self.ws = [self.w]
        self.losses = [self.loss(self.w,X_train, y_train).mean()]
        self.train_scores = [self.score(X_train, y_train)]
        if y_test is not None:
            X_test = self.phi(X_test)
            self.test_scores = [self.score(X_test, y_test)]
        for _ in range(self.max_iter):
            for start in range(0,n,m):
                batch_idx = indices[start:start+m]
                batch_x, batch_y = X_train[batch_idx], y_train[batch_idx]
                self.w -= self.eps*self.loss_g(self.w, batch_x, batch_y)
            self.ws.append(self.w)
            self.losses.append(self.loss(self.w,X_train, y_train).mean())
            self.train_scores.append(self.score(X_train, y_train))
            if y_test is not None:
                self.test_scores.append(self.score(X_test, y_test))     

    def predict(self,X):
        if self.w.shape[0] != X.shape[1]:
            X = self.phi(X)
        return np.sign(X @ self.w)

    def score(self,X_test,y_test):
        y_test = y_test.reshape(-1)
        y_hat = self.predict(X_test).reshape(-1)
        return (y_hat == y_test).sum() / y_test.shape[0]
    

    
    def plot_losses(self):
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Evolution of loss during training")
        plt.legend()
        plt.show()

    def plot_scores(self):
        plt.plot(range(len(self.test_scores)), self.test_scores, label="test")
        plt.plot(range(len(self.train_scores)), self.train_scores, label="train")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Evolution of train and test scores loss during training")
        # plt.ylim([0,1])
        plt.legend()
        plt.show()
    
    def show_w(self, title):
        plt.imshow(self.w.reshape((16,16)),interpolation="nearest")
        plt.title(title)
        plt.axis("off")
        plt.colorbar()
        plt.show()

def proj_bias(X):
    ones_col = np.ones(X.shape[0]).reshape(-1,1)
    return np.concatenate([ones_col,X], axis=1)

def proj_poly(X):
    n, d = X.shape
    X_proj = np.empty((n, (d+1)*(d+2)//2))
    X_proj[:,:d+1] = proj_bias(X)
    for i in range(d):
        for j in range(i,d):
            X_proj[:,d+1+i+j] = (X[:,i] * X[:,j])
    return X_proj

def linspace_basis(X,num=10):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    mesh = np.meshgrid(*[np.linspace(mins[i],maxs[i],num) for i in range(X.shape[1])])
    return np.stack(mesh, axis=-1).reshape(-1, X.shape[1])

def proj_gaus(X, basis, sigma):
    '''base by rows'''
    # n, d = X.shape
    # X_proj = np.empty((n, len(base)))
    # for i, b in enumerate(base):
    #     X_proj[:,i] = np.exp( - 0.5 * np.sum((X-b)**2, axis=1) / sigma**2)
    # return X_proj

    # smarter and faster:
    X2 = np.sum(X**2, axis=1, keepdims=True)     
    B2 = np.sum(basis**2, axis=1)                 
    cross = X @ basis.T                            
    dist2 = X2 + B2 - 2 * cross                   
    return np.exp(-dist2 / (2 * sigma**2))


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data, label):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.title(f"Digit {label}")
    plt.axis("off")
    plt.show()




if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
