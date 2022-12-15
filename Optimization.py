# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:43:58 2022

@author: user
"""



from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, brute, shgo, basinhopping
# from niapy.algorithms.basic import GreyWolfOptimizer
# from niapy.task import Task
# import pygad
from fireflyalgorithm import FireflyAlgorithm
import pickle



def load_depth_model():
     model = pickle.load(open('train_depth_model.sav', 'rb'))
     return model
def load_water_model():
    model = pickle.load(open('train_water_model.sav', 'rb'))
    return model
def load_soil_model():
    model = pickle.load(open('train_soil_model.sav', 'rb'))
    return model

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
    water_level = 1 - (1/water_level)
    soil_softness = get_softness(position)
    soil_softness = 1 - (1/soil_softness)
    cost = depth[0] + water_level[0] + soil_softness
    if(water_level[0]<10):   # User Preferences 
        depths.append(depth[0])
        levels.append(water_level[0])
        softness.append(soil_softness)
        costs.append(cost)
    return cost

model_depth = load_depth_model()
model_water = load_water_model()
model_soil = load_soil_model()

x_range = [198733.589 , 202258.3361] # Sample Test
y_range = [539535.0948, 545936.3696] # Sample Test
bounds = [x_range, y_range]   # Sample Test bounds

FA = FireflyAlgorithm()
result = FA.run(function=objective, dim=2, lb=[x_range[0],y_range[0]], ub=[x_range[1],y_range[1]], max_evals=10000)
plot_Cost(depths,levels,softness,costs)
# plot2Axis(depths, levels)
# plot_positions(positions)











