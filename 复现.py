'''
author:Liushk
'''

#Data
import numpy as np
import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

# preprocessing
from sklearn.preprocessing import MinMaxScaler

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error

#loader
train_df = pd.read_csv('DailyDelhiClimateTrain.csv')
test_df = pd.read_csv('DailyDelhiClimateTest.csv')

# 数据可视化，绘制特征分布
class plotly_graph:
    def __init__(self, data, date):
        self.data = data
        self.date = date
        self.name_lst = ['Mean Temp', 'Humidity', 'Wind Speed', 'Mean Pressure']
        self.box_title = 'Multiple Box Plots'
        self.line_title = 'Multiple Line Plots'

    # subplots
    def make_subplot(self, graphs):
        fig = make_subplots(rows=2, cols=2, subplot_titles=(self.name_lst))
        for i in range(4): fig.add_trace(graphs[i], row=i // 2 + 1, col=i % 2 + 1)
        return fig

    # boxplot
    def box_plot(self):
        graph_lsts = []
        for i, element in enumerate(self.data.transpose()):
            graph_lst = go.Box(y = element,
                               name = self.box_title,
                               boxpoints = 'outliers',
                               line = dict(width=1))
            graph_lsts.append(graph_lst)
        fig = self.make_subplot(graph_lsts)
        fig.update_layout(title=self.box_title,
                          xaxis_title='Columns',
                          yaxis_title='Values',
                          template = 'simple_white')
        fig.show()
    # line plot
    def line_plot(self):
        line_lsts = []
        for i, element in enumerate(self.data.transpose()):
            line = go.Scatter(x = self.date,
                               y = element,
                               mode = 'lines',
                               name = self.line_title)
            line_lsts.append(line)
        fig = self.make_subplot(line_lsts)
        fig.update_layout(title=self.line_title,
                          xaxis_title='Columns',
                          yaxis_title='Values',
                          template = 'simple_white')
        fig.show()

data_ = train_df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].values
graph = plotly_graph(data_, train_df['date'])

# graph
graph.box_plot()
graph.line_plot()


# Feature Engineering
# 增加列 湿度压力比

def humidity_pressure_ratio(df):
    df['humidity_pressure_ratio'] = df ['humidity'] / ['meanpressure']
    return df

# 增加日期列：年，月，日
def get_date_colums(date):
    year, month, day = date.spilt('-')
    return (year, month, day)


