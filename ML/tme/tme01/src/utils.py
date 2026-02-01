import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import pandas as pd
from typing import Callable

POI_FILENAME = "../data/poi-paris.pkl"
parismap = mpimg.imread('../data/paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin, xmax = 2.23, 2.48  # coord_x min et max
ymin, ymax = 48.806, 48.916  # coord_y min et max
coords = [xmin, xmax, ymin, ymax]

class Density(object):
    def fit(self,data):
        raise NotImplementedError
    def predict(self,data):
        raise NotImplementedError
    def score(self,data):
        #A compléter : retourne la log-vraisemblance
        # add a small number to avoid log(0) explosion
        return np.log(self.predict(data)+1e-10).sum() # type: ignore

class Histogramme(Density):
    def __init__(self,steps=10):
        Density.__init__(self)
        self.steps = steps

    def fit(self,x):
        #A compléter : apprend l'histogramme de la densité sur x
        hist, edges =  np.histogramdd(x, bins=self.steps)
        self.edges = np.array(edges)
        self.bin_size = np.diff(self.edges[:,:2], axis=-1).reshape(-1) 
        bin_area = np.prod(self.bin_size)
        self.density =  hist/hist.sum()/bin_area
    
    def to_bins(self, x):
        idx = np.floor( (x - self.edges[:,0]) / self.bin_size).astype(int)
        for d in range(idx.shape[1]):
            idx[:,d] = np.clip(idx[:,d], 0, self.density.shape[d]-1)
        return idx 
    
    def predict(self,x):
        #A compléter : retourne la densité associée à chaque point de x
        idx = self.to_bins(x)
        return self.density[idx[:,0], idx[:,1]]

def kernel_uniform(x):
    #return np.array([np.all(np.abs(row) <= 1/2) for row in x ], dtype=int)
    return (np.max(np.abs(x), axis=1) <= 1/2).astype(float)

def kernel_gaussian(x):
    d = x.shape[1]
    return (2*np.pi)**(-d/2)* np.exp(-1/2 * np.sum(x**2, axis=1))

class KernelDensity(Density):
    def __init__(self,sigma,kernel):
        Density.__init__(self)
        self.kernel = kernel
        self.sigma = sigma
    def fit(self,x):
        self.x = x
    def predict(self,data):
        N, d = self.x.shape
        data = np.atleast_2d(data) # reshapes a single sample vector into a matrix (if it's the case)
        #A compléter : retourne la densité associée à chaque point de data
        sm = np.array([np.sum(self.kernel((x0-self.x) / self.sigma)) for x0 in data]) # type: ignore
        return sm / self.sigma ** d / N 
    
        # Proposed by ChatGPT (as fast for this dataset, but more elegant):
        # diff = (data[:, None, :] - self.x[None, :, :]) / self.sigma
        # K = self.kernel(diff.reshape(-1, d)).reshape(diff.shape[:2])
        # return K.sum(axis=1) / (N * self.sigma**d)

class Nadaraya(Density):
    def __init__(self, sigma, kernel) -> None:
        super().__init__()
        self.sigma = sigma
        self.kernel = kernel
    
    def fit(self, x, y):
        self.x = x
        self.y = y
    
    def predict(self, data):
        diff = (data[:,None, :] - self.x[None, :, :]) / self.sigma
        K = self.kernel(diff.reshape(-1, diff.shape[-1])).reshape(diff.shape[:2])
        weighted_sum = (K*self.y).sum(axis=1)
        sum_weights = K.sum(axis=1)
        return weighted_sum / sum_weights
    
    def score(self, data, y):
        y_hat = self.predict(data)
        return np.sum((y_hat - y)**2) / len(y)


def cross_validate(estimator,data,params, y=None, param_name="hyperparameter", title = "", folds=5):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    fold_bounds = np.linspace(0,len(idx)-1, folds+1).astype(int)
    scores_train = np.empty((len(params), folds))
    scores_test = np.empty((len(params), folds))
    for i, param in enumerate(params):
        for fold in range(folds):
            start = fold_bounds[fold]
            end = fold_bounds[fold+1]
            test_idx = idx[start:end]
            train_idx = np.concatenate([idx[:start], idx[end:]])
            x_test = data[test_idx,:]
            x_train = data[train_idx,:]
            f = estimator(param)
            if y is None:
                f.fit(x_train)
                scores_test[i, fold] = f.score(x_test)
                scores_train[i, fold] = f.score(x_train)
            else:
                f.fit(x_train, y[train_idx])
                scores_test[i, fold] = f.score(x_test, y[test_idx])
                scores_train[i, fold] = f.score(x_train, y[train_idx])
            
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes = axes.flatten()

    fig.suptitle(title+f"\n{folds} fold cross validation")

    axes[0].set_xlabel(param_name)
    if y is None:
        score_name = "log likelihood" 
    else:
        score_name = "MSE" 
    axes[0].set_ylabel(score_name + " (mean $\\pm$ std)")

    mean, std = scores_test.mean(1), scores_test.std(1)

    axes[0].set_title("test")
    axes[0].plot(params, mean, label="test")
    axes[0].fill_between(params, mean - std, mean +  std, alpha = 0.2)

    axes[1].set_xlabel(param_name)
    # axes[1].set_ylabel("train log likelihood (mean $\\pm$ std)")

    mean, std = scores_train.mean(1), scores_train.std(1)
    axes[1].set_title("train")
    axes[1].plot(params, mean, label="train")
    axes[1].fill_between(params, mean - std, mean +  std, alpha = 0.2)
    plt.show()
    
def get_density2D(f,data,steps=100):
    """ Calcule la densité en chaque case d'une grille steps x steps dont les bornes sont calculées à partir du min/max de data. Renvoie la grille estimée et la discrétisation sur chaque axe.
    """
    xmin, xmax = data[:,0].min(), data[:,0].max()
    ymin, ymax = data[:,1].min(), data[:,1].max()
    xlin,ylin = np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps)
    xx, yy = np.meshgrid(xlin,ylin)
    grid = np.c_[xx.ravel(), yy.ravel()]
    res = f.predict(grid).reshape(steps, steps)
    return res, xlin, ylin



def show_density(f, data, steps=100, log=False):
    """ Dessine la densité f et ses courbes de niveau sur une grille 2D calculée à partir de data, avec un pas de discrétisation de steps. Le paramètre log permet d'afficher la log densité plutôt que la densité brute
    """
    res, xlin, ylin = get_density2D(f, data, steps)
    xx, yy = np.meshgrid(xlin, ylin)
    plt.figure()
    show_img()
    if log:
        res = np.log(res+1e-10)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8, s=3)
    show_img(res)
    plt.colorbar()
    plt.contour(xx, yy, res, 20)


def show_img(img=parismap):
    """ Affiche une matrice ou une image selon les coordonnées de la carte de Paris.
    """
    origin = "lower" if len(img.shape) == 2 else "upper"
    alpha = 0.3 if len(img.shape) == 2 else 1.
    plt.imshow(img, extent=coords, aspect=1.5, origin=origin, alpha=alpha)
    ## extent pour controler l'echelle du plan


def load_poi(typepoi,fn=POI_FILENAME):
    """ Dictionaire POI, clé : type de POI, valeur : dictionnaire des POIs de ce type : (id_POI, [coordonnées, note, nom, type, prix])
    
    Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, 
    clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    """
    poidata = pickle.load(open(fn, "rb"))
    data = np.array([[v[1][0][1],v[1][0][0]] for v in sorted(poidata[typepoi].items())])
    note = np.array([v[1][1] for v in sorted(poidata[typepoi].items())])
    return data,note

if __name__ == "__main__":
    plt.ion()
    # Liste des POIs : furniture_store, laundry, bakery, cafe, home_goods_store, clothing_store, atm, lodging, night_club, convenience_store, restaurant, bar
    # La fonction charge la localisation des POIs dans geo_mat et leur note.
    geo_mat, notes = load_poi("bar")

    # Affiche la carte de Paris
    show_img()
    # Affiche les POIs
    plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.8,s=3)



