# som implementation using minisom module

from minisom import MiniSom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")
##import cursor_funtionality as cf


def best_q_params(model_error, model_param, best_n=5):
    errors = np.asarray(model_error)
    q_error_sort_index = errors[:,0].argsort()
    #printing best n params based on q_errors
    for i, ele in enumerate(q_error_sort_index):
        if i == best_n:
            print("-"*80, "\n", "-"*80)
            break
        params = model_param[ele]
        print(("qunatization error: %.4f,  topographic error:%.4f" %
               model_error[q_error_sort_index[i]]))
        print("parameters: %s\n" % params)

def best_t_params(model_error, model_param, best_n=5):
    errors = np.asarray(model_error)
    t_error_sort_index = errors[:,1].argsort()
    #printing best n params based on q_errors
    for i, ele in enumerate(t_error_sort_index):
        if i == best_n:
            print("-"*80, "\n", "-"*80)
            break
        params = model_param[ele]
        print(("Quantization Error: %.4f,  Topographic Error:%.4f" %
               model_error[t_error_sort_index[i]]))
        print("Parameters: %s\n" % params)

def get_best_params(X, param_grid):
    model_error = []
    model_param = []
    for i, param_dict in enumerate(list(ParameterGrid(param_grid))):
        print("using parameters ", param_dict)
        print('\n')
        som = MiniSom(x=param_dict['grid_size'], y=param_dict['grid_size'],
                      input_len = X.shape[1], sigma=param_dict['sigma'],
                      learning_rate=param_dict['learning_rate'], random_seed=49)
        som.random_weights_init(X)
        som.train_random(data=X, num_iteration=param_dict['iter'])
        q_error = som.quantization_error(X)
        t_error = som.topographic_error(X)
        model_error.append((q_error, t_error))
        model_param.append(param_dict)

    best_q_params(model_error, model_param)
    best_t_params(model_error, model_param)

def train_som(X, param_dict):
    som = MiniSom(x=param_dict['grid_size'], y=param_dict['grid_size'],
                      input_len = X.shape[1], sigma=param_dict['sigma'],
                      learning_rate=param_dict['learning_rate'], random_seed=0)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=param_dict['iter'])
    q_error = som.quantization_error(X)
    t_error = som.topographic_error(X)
    print(("Quantization Error: %0.4f  Topographic Error: %0.4f" %
           (q_error, t_error)))
    return som

##    with open('som.p', 'wb') as outfile:
##        pickle.dump(som, outfile)

def train_batch(X, param_dict):
    pass

def plot_iter(X, param_dict):
    som = MiniSom(x=param_dict['grid_size'], y=param_dict['grid_size'],
                  input_len = X.shape[1], sigma=param_dict['sigma'],
                  learning_rate=param_dict['learning_rate'], random_seed=49)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=param_dict['iter'])
    errors_dict = som.get_errors()
    
    fig, ax = plt.subplots()
    errors = np.asarray(list(errors_dict.values()))
    ax.plot(errors[:,0], linewidth=0.7, gid='quantization error',
            label='Quantization Error')
    ax.plot(errors[:,1], linewidth=0.7, gid='topographic error',
            label='Topographic Error')
    ax.set_title('Model Error for No. of Iterations', {'fontsize':20})
    ax.set_xlabel('No. of Iterations')
    ax.set_ylabel('Error')
    ax.legend()

##    ax_cf = cf.SnaptoCursor(ax1, df1,annotate_onplot=False)
    
    plt.show()
    
def plot_grid_size(X, param_dict, start, stop, step):
    errors_dict = {}
    for grid_size in range(start, stop, step):
        som = MiniSom(x=grid_size, y=grid_size, input_len = X.shape[1],
                      sigma=param_dict['sigma'],
                      learning_rate=param_dict['learning_rate'], random_seed=49)
        som.random_weights_init(X)
        som.train_random(data=X, num_iteration=param_dict['iter'])
        errors_dict[grid_size] = (som.quantization_error(X),
                                  som.topographic_error(X))
    
    fig, ax = plt.subplots()
    errors = np.asarray(list(errors_dict.values()))
    ax.plot(list(errors_dict.keys()), errors[:,0], linewidth=0.7,
            gid='quantization error', label='Quantization Error')
    ax.plot(list(errors_dict.keys()), errors[:,1], linewidth=0.7,
            gid='topographic error', label='Topographic Error')
    ax.set_title('Model Error for Grid Size', {'fontsize':20})
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

def plot_sigma(X, param_dict, start, stop, step):
    sigmas = []
    errors = []
    for sig in np.arange(start, stop, step):
        som = MiniSom(x=param_dict['grid_size'], y=param_dict['grid_size'],
                      input_len = X.shape[1], sigma=sig,
                      learning_rate=param_dict['learning_rate'], random_seed=49)
        som.random_weights_init(X)
        som.train_random(data=X, num_iteration=param_dict['iter'])
        sigmas.append(sig)
        errors.append((som.quantization_error(X), som.topographic_error(X)))
    
    fig, ax = plt.subplots()
    errors = np.asarray(errors)
    ax.plot(sigmas, errors[:,0], linewidth=0.7,
            gid='quantization error', label='Quantization Error')
    ax.plot(sigmas, errors[:,1], linewidth=0.7,
            gid='topographic error', label='Topographic Error')
    ax.set_title('Model Error for Sigma', {'fontsize':20})
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

def plot_learning_rate(X, param_dict, start, stop, step):
    l_rates = []
    errors = []
    for l_rate in np.arange(start, stop, step):
        som = MiniSom(x=param_dict['grid_size'], y=param_dict['grid_size'],
                      input_len = X.shape[1], sigma=param_dict['sigma'],
                      learning_rate=l_rate, random_seed=49)
        som.random_weights_init(X)
        som.train_random(data=X, num_iteration=param_dict['iter'])
        l_rates.append(l_rate)
        errors.append((som.quantization_error(X), som.topographic_error(X)))
    
    fig, ax = plt.subplots()
    errors = np.asarray(errors)
    ax.plot(l_rates, errors[:,0], linewidth=0.7,
            gid='quantization error', label='Quantization Error')
    ax.plot(l_rates, errors[:,1], linewidth=0.7,
            gid='topographic error', label='Topographic Error')
    ax.set_title('Model Error for Learning Rate', {'fontsize':20})
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

def main():
    # Importing the dataset
    dataset = pd.read_csv('Credit_Card_Applications.csv')
    print(dataset.columns)
    X = dataset.iloc[:, :-1]
##    X.drop(X.columns[0], axis=1, inplace=True)
    X = np.asarray(X)
    print(X.shape)
    y = dataset.iloc[:, -1].values

    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)

    #finding best parameters
    param_grid = {'grid_size':[10,20,50], 'iter':[100,1000,10000],
                  'sigma':[0.25,0.5,0.75,1.0],
                  'learning_rate':[1.0,0.5,0.1,0.01]}
##    get_best_params(X, param_grid)

    #training som using the best parameters found
##    best_param = {'grid_size': 10, 'iter': 100, 'learning_rate': 1.05,
##                  'sigma': 1.26}
    best_param = {'grid_size': 10, 'iter': 100, 'learning_rate': 1,
                  'sigma': 1.0}
    som = train_som(X, best_param)
    print(som.activation_response(X))
    print(som.distance_map())
    print(som.get_weights().shape)
##    print(len(som.win_map(X)[(6,1)]))
##    winner_coordinates = np.array([som.winner(x) for x in X])
##    print(winner_coordinates)
    
    plot_iter(X, best_param)
##    plot_grid_size(X, best_param, 10, 100, 20)
    plot_grid_size(X, best_param, 5, 30, 5)
    plot_sigma(X, best_param, 0.001, 4.9, 0.15)
    plot_learning_rate(X, best_param, 0.001, 4.9, 0.15)

##    grid_size = int(np.sqrt(5*np.sqrt(X.shape[0])))

    # Visualizing the results
    from matplotlib import cm
    fig, ax = plt.subplots()
    ax.pcolor(som.distance_map(), cmap='bone')
    fig.colorbar(cm.ScalarMappable(cmap='bone'))
    markers = ['o', 's']
    colors = ['r', 'g']
    for i, x in enumerate(X):
        w = som.winner(x)
        plt.plot(w[0] + 0.5,
              w[1] + 0.5,
              markers[y[i]],
              markeredgecolor = colors[y[i]],
              markerfacecolor = 'None',
              markersize = 10,
              markeredgewidth = 2)
    plt.gca().invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xlabel('yoo')
    ax.xaxis.set_label_position('top')
    plt.show()

##    quantization_errors = np.linalg.norm(som.quantization(X) - X, axis=1)
##    error_treshold = np.percentile(quantization_errors,
##                                   100*(1-0.35)+5)
##    frauds_bool = quantization_errors > error_treshold
##    final = pd.DataFrame(sc.inverse_transform(X[frauds_bool, :]))
##    check = final.loc[:, 0].astype(int).values
##    print(dataset.loc[check, dataset.columns[0]])
    
main()




