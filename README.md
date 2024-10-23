# -Practice-code-before-graduation-
This project mainly covers predictions on time series data across different fields using models like LSTM and Transformer. It includes ideas on data visualization, feature engineering, and model framework design (though not all of them are successful yet). And of course, there are some notes for myself as well.

Tks

20241016 
完成了数据可视化与新指标计算部分。遇到几个书写错误，无大碍。

20241017 
完成了lookback、LSTM建模、earlystop等模块的实现。bug之余，出现两个警告，分别如下：
1. lookback模块部分运行中报了警告：UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\actions runner\_work\pytorch\pytorch\builder\windows\pytorch\torch\csrc\utils\tensor_new.cpp:233.) return torch.tensor(X), torch.tensor(y)
老问题，使用列表效率较低，一般的解决方法是将列表转换为单个numpy数组，再转为张量即可。解决代码：
    将列表转换为 NumPy 数组
    X = np.array(X)
    y = np.array(y)
    将 NumPy 数组转换为 PyTorch 张量
    return torch.from_numpy(X), torch.from_numpy(y)
解决问题
2. 数据可视化部分参考的实现方法有些老，给了警告。使用的Plotly库来创建交互式图表，但是警告提示plotly.graph_objs.Line已被弃用。所以寻找其他新方法，将 go.Line 替换为 go.Scatter，并设置 mode='lines' 来创建线图。这样可以避免使用已弃用的 Line 对象，同时保持相同的图表效果。

20241018
前两日复现了单层LSTM对天气的时序数据预测,今日想实现多层LSTM预测,碍于盲审和档案等学校的事情，无新文件上传。
目前只做了数据集描述和读取，LSTM建模部刚起步，打算实现多层LSTM预测，根据预测效果决定是否加上注意力机制。理论上多层LSTM更能缓解梯度消失问题，改善预测效果。后期优化，保存最佳权重文件。

20241019-22 
工作技术笔试

20241023
上传了 整合预测.py
完成了18号预定的目标，多层LSTM建模加新绘图拟合，与上一个模型相比变化不是很大，备注会更详细一些。下一阶段的目标是把transformer和LSTM结合，想尝试更高纬和复杂的数据集，作为下一个模型的训练数据。如果模型设计顺利，下一步就是研究多模态模型。
