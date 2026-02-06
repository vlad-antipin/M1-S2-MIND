import numpy as np
import matplotlib.pyplot as plt

from mltools import plot_data, plot_frontiere, make_grid, gen_arti

def reshape_all(w,x,y):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    x = x.reshape(y.shape[0], w.shape[0])
    return w, x, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(w,x,y):
    w, x, y = reshape_all(w, x, y)
    diff = y - x @ w
    return (diff**2) # no .mean(), returns vector of squared errors

def mse_grad(w,x,y):
    w, x, y = reshape_all(w, x, y)
    # return  2 * x * (x@w - y) # TODO: why it does the same thing?
    return 1/ y.shape[0] * 2 * x.T @ (x@w - y)

def reglog(w,x,y):
    w, x, y = reshape_all(w, x, y)
    # NOTE: if y was in {0,1}: (cross-entropy)
    # return - (y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    # but here it's in {-1,1}, so:
    return np.log(1 + np.exp(-y* x@w))

def reglog_grad(w,x,y):
    w, x, y = reshape_all(w, x, y)
    # if derive directly:
    # return (- 1/(1 + np.exp(-y* x@w)) *y * x * np.exp(-y* x@w) ).mean(0).reshape(-1,1)  
    return (- sigmoid(-y*x@w) * y * x ).mean(0)

def grad_check(f, f_grad, N=100, d=2):
    pass
    # w0 = np.random.randn(d,1)
    # w = w0 + 1e-3 * np.random.randn(d,1)
    # x = np.random.randn(N,d)
    # y = np.random.choice([-1,1],N)
    # f_taylor = f(w0,x,y) + (w-w0).T @ f_grad(w0,x,y).T
    # print( (f(w,x,y) - f_taylor) / np.linalg.norm(w - w0)**2 )

def descente_gradient(datax, datay, f_loss, f_grad,eps,  iter):
    w = np.random.randn(datax.shape[1],1)  # np.random.randn(datax.shape[1],1) # np.zeros((datax.shape[1],1)) 
    ws = [w]
    losses = [f_loss(w,datax, datay).mean()]
    for _ in range(iter):
        w = w - eps*f_grad(w, datax, datay).mean(-1).reshape(-1, 1)
        ws.append(w)
        losses.append(f_loss(w,datax, datay).mean())
    return w, np.array(ws).reshape(-1,2), np.array(losses)

def check_fonctions():
    ## On fixe la seed de l'aléatoire pour vérifier les fonctions
    np.random.seed(0)
    datax, datay = gen_arti(epsilon=0.1)
    wrandom = np.random.randn(datax.shape[1],1)
    assert(np.isclose(mse(wrandom,datax,datay).mean(),0.54731,rtol=1e-4))
    assert(np.isclose(reglog(wrandom,datax,datay).mean(), 0.57053,rtol=1e-4))
    assert(np.isclose(mse_grad(wrandom,datax,datay).mean(),-1.43120,rtol=1e-4))
    assert(np.isclose(reglog_grad(wrandom,datax,datay).mean(),-0.42714,rtol=1e-4))
    np.random.seed()


if __name__=="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax, datay = gen_arti(epsilon=0.1)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)
    
    plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    w  = np.random.randn(datax.shape[1],1)
    plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
    plot_data(datax,datay)

    ## Visualisation de la fonction de coût en 2D
    plt.figure()
    plt.contourf(x_grid,y_grid,np.array([mse(w,datax,datay).mean() for w in grid]).reshape(x_grid.shape),levels=20)
    
