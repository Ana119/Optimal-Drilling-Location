

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, brute, shgo, basinhopping
# from niapy.algorithms.basic import GreyWolfOptimizer
# from niapy.task import Task
# import pygad
import pickle

def train_depth_model():
    X, y = get_depth_data()
    model = DecisionTreeRegressor()
    model.fit(X, y)
    filename = 'train_depth_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model

def load_depth_model():
     model = pickle.load(open('train_depth_model.sav', 'rb'))
     return model
def load_water_model():
    model = pickle.load(open('train_water_model.sav', 'rb'))
    return model
def load_soil_model():
    model = pickle.load(open('train_soil_model.sav', 'rb'))
    return model
    
def train_water_model():
    X, y = get_water_data()
    model = DecisionTreeRegressor()
    model.fit(X, y)
    filename = 'train_water_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model

def train_soil_model():
    X, y = get_soil_data()
    model = DecisionTreeRegressor()
    model.fit(X, y)
    filename = 'train_soil_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model

def get_data():
     data = pd.read_csv('D:\\Projects\\Drilling Project\\Classification Project\\Data\\all-features.csv')
     max_value = data['total_depth'].max()
     min_value = data['total_depth'].min()
     data['total_depth'] = (data['total_depth'] - min_value) / (max_value - min_value)
     return data
     
def get_depth_data():
    data = get_data()
    X = pd.DataFrame()
    X['X'] = data['X']
    X['Y'] = data['Y']
    y = data['total_depth']
    return X,y

def get_water_data():
    data = pd.read_csv('D:\\Projects\\Drilling Project\\Classification Project\\Data\\water_depth_data.csv')
    X = pd.DataFrame()
    X['X'] = data['X']
    X['Y'] = data['Y']
    y = data['Water_level']
    return X,y

def get_soil_data():
    data = get_data()
    X = pd.DataFrame()
    X['X'] = data['X']
    X['Y'] = data['Y']
    soil_color = data['soil_color']
    soil_color = np.array(soil_color)
    soil_color = soil_color.reshape(-1,1)
    oe = OrdinalEncoder()
    oe.fit(soil_color)
    soil_color = oe.transform(soil_color)
    y = soil_color
    data['soil_encoded'] = soil_color
    return X,y

def get_softness(position):
    softness = 0.001
    soil_color = model_soil.predict(position)
    if soil_color == 0: # Bitumen
        softness = 2.86
    elif soil_color == 1: # Brown
        softness = 3.13
    elif soil_color == 2: # Dark Brown
        softness = 4.34
    elif soil_color == 3: # Dark Gray
        softness = 3.45
    elif soil_color == 4: # Gray
        softness = 2.64
    elif soil_color == 5: # Light Brown
        softness = 3.43
    elif soil_color == 6: # Light Gray
        softness = 3.85
    elif soil_color == 7: # Partridge
        softness = 2.37
    elif soil_color == 8: # Tan
        softness = 3.09
    elif soil_color == 9: # Taupe
        softness = 3.58
    return softness

def get_depth(position):
    depth = model_depth.predict(position)
    return depth

def get_water_level(position):
    water_level = model_water.predict(position)
    return water_level

def plot_Cost(depths,levels, softness, costs):
      x_axis=[x for x in range(0,len(depths))]
      fig = plt.figure(figsize=(8,4))
      plt.plot(x_axis, softness,marker='.', label="Depths",color = 'red')
      plt.plot(x_axis, levels, 'g', label="Test Water level", color = '#234518')
      plt.ylabel("Cost", size=15)
      plt.xlabel('Iterations', size=15)
      plt.legend(fontsize=12)
      fig = plt.figure(figsize=(8,4))
      
      # plt.plot(x_axis, levels, 'r', label="Test Water level", color = '#234518')
      # plt.ylabel("X", size=15)
      # plt.xlabel('Y', size=15)
      # plt.legend(fontsize=12)
      # fig = plt.figure(figsize=(8,4))
      
      # plt.plot(x_axis, softness, marker='.', label="Softness", color = 'red')
      # plt.ylabel("Cost", size=15)
      # plt.xlabel('Time step', size=15)
      # plt.legend(fontsize=12)
      fig = plt.figure(figsize=(8,4))
      
      plt.plot(x_axis,  costs, marker='.', label="Cost", c = 'r')
      plt.tight_layout()

      plt.subplots_adjust(left=0.07)
      
      #plt.ylabel('Energy Consumption (KW)',size=15)
      
      plt.ylabel("Cost", size=15)
      plt.xlabel('Iterations', size=15)
      plt.legend(fontsize=12)
      
      plt.title('Optimal Drilling Point Search based on Enhanced FA')
      #plt.title("TF Lite DNN for  ")
      plt.show()
      fig.savefig("Convergence.png", dpi = 600,bbox_inches='tight')
def plot2Axis(depths, levels):
    x_axis=[x for x in range(0,len(depths))]
    fig, ax1 = plt.subplots(figsize=(15,4))
    color = 'tab:red'
    ax1.set_xlabel('time step')
    ax1.set_ylabel('Depth', color=color)
    ax1.plot(x_axis, depths, color=color, marker = '*')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Water Level', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_axis, levels, color=color, marker = '*')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Water Level and Depth of Drilling Point During Search')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    fig.savefig('multiaxis.pdf', dpi = 600)
def plot_positions(positions):
      fig = plt.figure(figsize=(8,4))
      positions = np.array(positions)
      plt.scatter(positions[:,0], positions[:,1],marker='*', label="Population",color = 'red')
      positions = positions[-1]
      plt.scatter(positions[0], positions[1], marker='o', s = 50, label="Optimal Location",color = 'green')
      plt.ylabel("Y", size=15)
      plt.xlabel('X', size=15)
      #plt.legend(fontsize=12)
      plt.tight_layout()
      plt.subplots_adjust(left=0.07)
      #plt.ylabel('Energy Consumption (KW)',size=15)
     # plt.ylabel("Cost", size=15)
      plt.title('Optimal Drilling Point Search using Firefly')
      #plt.xlabel('Time step', size=15)
      fig.tight_layout()
      plt.legend(fontsize=12)
      plt.show()
      fig.savefig('location finding.pdf', dpi = 600)

depths = []
levels = []
softness = []
costs = []
positions = []

def objective(position):
    #print('position', position)
    positions.append(position)
    position = [position]
    depth = get_depth(position)
    depth = 1/depth
    water_level = get_water_level(position)
    #water_level = 1 - (1/water_level)
    soil_softness = get_softness(position)
   # soil_softness = 1 - (1/soil_softness)
    cost = depth[0] + water_level[0] + soil_softness
    if(water_level[0]<10):
        depths.append(depth[0])
        levels.append(water_level[0])
        softness.append(soil_softness)
        costs.append(cost)
    return cost

model_depth = load_depth_model()
model_water = load_water_model()
model_soil = load_soil_model()

x_range = [198733.589 , 202258.3361]
y_range = [539535.0948, 545936.3696]
bounds = [x_range, y_range]

result  = differential_evolution(objective,bounds)
plot_Cost(depths,levels,softness,costs)
# plot2Axis(depths, levels)
# plot_positions(positions)











