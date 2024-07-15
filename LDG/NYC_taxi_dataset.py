import csv
import torch
import numpy as np
import pandas as pd
import copy
import networkx as nx
import random
import matplotlib.pyplot as plt
from math import sqrt 


# Load the CSV file into a DataFrame
data = pd.read_csv("train.csv")

# Display the first five rows of the DataFrame
print(data.shape)
data.head()

data2=copy.deepcopy(data)
data2['pickup_longitude'] = data['pickup_longitude'].round(2)
data2['pickup_latitude'] = data['pickup_latitude'].round(2)
data2['dropoff_longitude'] = data['dropoff_longitude'].round(2)
data2['dropoff_latitude'] = data['dropoff_latitude'].round(2)

data2.head()


# Create a directed graph
G2 = nx.DiGraph()
selfloops = 0
# Add edges to the graph
for row in data2.itertuples():
    source = (row.pickup_longitude, row.pickup_latitude)
    destination = (row.dropoff_longitude, row.dropoff_latitude)
    weight = row.trip_duration
    if (source != destination):
      G2.add_edge(source, destination, weight=weight)
    else :
      selfloops+=1

print(f'num of self loops in data2 = {selfloops}')