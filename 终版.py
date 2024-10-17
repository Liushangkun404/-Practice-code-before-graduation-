'''
author:Liushk
'''

# Data
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

# loader
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
            graph_lst = go.Box(y=element,
                               name=self.box_title,
                               boxpoints='outliers',
                               line=dict(width=1))
            graph_lsts.append(graph_lst)
        fig = self.make_subplot(graph_lsts)
        fig.update_layout(title=self.box_title,
                          xaxis_title='Columns',
                          yaxis_title='Values',
                          template='simple_white')
        fig.show()

    # line plot
    def line_plot(self):
        line_lsts = []
        for i, element in enumerate(self.data.transpose()):
            line = go.Scatter(x=self.date,
                              y=element,
                              mode='lines',
                              name=self.line_title)
            line_lsts.append(line)
        fig = self.make_subplot(line_lsts)
        fig.update_layout(title=self.line_title,
                          xaxis_title='Columns',
                          yaxis_title='Values',
                          template='simple_white')
        fig.show()


data_ = train_df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].values
graph = plotly_graph(data_, train_df['date'])

# graph
'''
graph.box_plot()
graph.line_plot()
'''

# Feature Engineering
# 增加列 湿度压力比

def humidity_pressure_ratio(df):
    df['humidity_pressure_ratio'] = df['humidity'] / df['meanpressure']
    return df

# 增加日期列：年，月，日
def get_date_columns(date):
    year, month, day = date.split('-')
    return (year, month, day)


# apply func
train_df = humidity_pressure_ratio(train_df)
test_df = humidity_pressure_ratio(test_df)

# apply func
tr_date_cols = train_df['date'].apply(get_date_columns)
te_date_cols = test_df['date'].apply(get_date_columns)

train_df[['year', 'month', 'day']] = pd.DataFrame(tr_date_cols.tolist(), index=train_df.index)
test_df[['year', 'month', 'day']] = pd.DataFrame(te_date_cols.tolist(), index=test_df.index)

print('Train set \n\n')
#print(train_df.head())

#feature selection
tr_timeseries = train_df[['month', 'day', 'humidity', 'wind_speed', 'meanpressure', 'humidity_pressure_ratio', 'meantemp']].values.astype('float32')
te_timeseries = test_df[['month', 'day',  'humidity', 'wind_speed', 'meanpressure', 'humidity_pressure_ratio', 'meantemp']].values.astype('float32')

new = pd.concat([train_df, test_df], axis=0).reset_index().drop('index', axis=1)
new_timeseries = new[['month', 'day',  'humidity', 'wind_speed', 'meanpressure',  'humidity_pressure_ratio', 'meantemp']].values.astype('float32')

# scaling
scaler = MinMaxScaler()
tr_timeseries = scaler.fit_transform(tr_timeseries)
te_timeseries = scaler.transform(te_timeseries)

# set loolback(window size,sequence length)
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[:,:6][i:i+lookback]
        target = dataset[:, 6][i:i+lookback]
        X.append(feature)
        y.append(target)
    
    # 将列表转换为 NumPy 数组
    X = np.array(X)
    y = np.array(y)
    
    # 将 NumPy 数组转换为 PyTorch 张量
    return torch.from_numpy(X), torch.from_numpy(y)

lookback = 7

train, test = tr_timeseries, te_timeseries
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

# modify shape of train and test
X_train, X_test = X_train, X_test
y_train, y_test = y_train, y_test

loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                         batch_size = 8, shuffle = True) #这段代码定义了一个 PyTorch 的 DataLoader，用于将训练数据按批次（batch）加载，并在每个 epoch 中随机打乱数据。

#LSTM建模
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6,
                            num_layers=2,
                            hidden_size=128,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 8)
        self.output_linear = nn.Linear(8, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output_linear(x)
        return x

# 调用模块并定义优化器与损失函数
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = LSTMModel()
optimizer = optim.Adam(model.parameters(),lr = 1e-3 ,weight_decay = 1e-5)
loss_fn = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode = 'min',factor= 0.5, patience=5, verbose=True)

# 定义提前停止函数
class CustomEarlyStopping:
    def __init__(self, patience=20, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        self.best_y_pred = None

    def __call__(self, val_loss, model, X):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            with torch.no_grad():
                self.best_y_pred = model(X)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {self.best_score}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            with torch.no_grad():
                self.best_y_pred = model(X)
            self.counter = 0


early_stopping = CustomEarlyStopping(patience=15, verbose=True)

# 使用预定义的函数进行训练
best_score = None
best_weights = None
best_train_preds = None
best_test_preds = None

n_epochs = 200

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train.unsqueeze(2)))
        train_preds = y_pred.clone().detach().cpu().numpy()

        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test.unsqueeze(2)))
        test_preds = y_pred.clone().detach().cpu().numpy()

        # Update the learning rate scheduler and early stopping
        scheduler.step(test_rmse)

        if best_score is None or test_rmse < best_score:
            best_score = test_rmse
            best_weights = model.state_dict()
            best_train_preds = train_preds
            best_test_preds = test_preds

        early_stopping(test_rmse, model, X_test)

        # Check if early stopping criterion is met
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if epoch % 10 == 0:
        print('*' * 10, 'Epoch: ', epoch, '\ train RMSE: ', train_rmse, '\ test RMSE', test_rmse)

# 评估
if best_weights is not None:
    model.load_state_dict(best_weights)

    # using the best weights to generate predictions
    with torch.no_grad():
        y_pred_train = model(X_train).clone().detach().cpu().numpy()
        y_pred_test = model(X_test).clone().detach().cpu().numpy()

with torch.no_grad():
    train_plot = np.ones_like(new_timeseries) * np.nan
    train_plot[lookback: len(train)] = y_pred_train[:, -1, :]

    test_plot = np.ones_like(new_timeseries) * np.nan
    test_plot[len(train) + lookback:len(new_timeseries)] = y_pred_test[:, -1, :]

train_predictions = scaler.inverse_transform(train_plot)
test_predictions = scaler.inverse_transform(test_plot)


    # 绘图
import plotly.express as px
import plotly.graph_objects as go

# plot
plt.figure(figsize=(20,10))
plt.plot(new_timeseries[:,6], c = 'b')
plt.plot(train_predictions[:,6], c='r')
plt.plot(test_predictions[:,6], c='g')


# plt.xlim([500,1000])
# plt.ylim([100000, 7000ㅋ00])
plt.show()

eval_df = pd.concat([test_df['meantemp'].reset_index(),
                  pd.Series(test_predictions[:,6][len(train):].reshape(-1).tolist())],axis=1).drop('index',axis=1)

eval_df.columns = ['real_meantemp', 'pred_meantemp']

fig = go.Figure(data = [
    go.Scatter(x = eval_df.index, y = eval_df['real_meantemp'], name = "Actual", mode='lines'),
    go.Scatter(x = eval_df.index, y = eval_df['pred_meantemp'], name="Predict", mode='lines'),
])

fig.update_layout(
    font = dict(size=17,family="Franklin Gothic"),
    template = 'simple_white',
    title = 'Real & Predicted Temp')

fig.show()

np.sqrt(mean_squared_error(eval_df.iloc[7:]['real_meantemp'], eval_df.iloc[7:]['pred_meantemp']))
'''
# 计算评估指标
mae = mean_absolute_error(eval_df['real_meantemp'], eval_df['pred_meantemp'])
mse = mean_squared_error(eval_df['real_meantemp'], eval_df['pred_meantemp'])
rmse = np.sqrt(mse)

print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
'''
