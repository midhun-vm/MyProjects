
import pandas as pd
import datetime
import operator
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff
import numpy as np
import sklearn.metrics as me

# Data Import And Labeling
print('######### Importing DataSet #########')
data = pd.read_csv("E:/Study/Python/pytorch/voice.csv")
importData = data
data.label = [1 if each == "male" else 0 for each in data.label]

# Check Missing Values
print("######### Checking For Missing Values #########")
data.isnull().sum()
# Split Predictor And Response Variable
x = data.drop(["label"], axis=1)
y = data.label

# Correlation Heat Map
g = data.iloc[:, :-1].astype(float).corr()
layout = go.Layout(
    title='Correlation Heat Map'
)
gdata = go.Figure(data=[go.Heatmap(z=g.values, x=g.columns,
                                   y=g.columns,
                                   colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'],
                                               [0.2222222222222222, 'rgb(244,109,67)'],
                                               [0.3333333333333333, 'rgb(253,174,97)'],
                                               [0.4444444444444444, 'rgb(254,224,144)'],
                                               [0.5555555555555556, 'rgb(224,243,248)'],
                                               [0.6666666666666666, 'rgb(171,217,233)'],
                                               [0.7777777777777778, 'rgb(116,173,209)'],
                                               [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']])],
                  layout=layout)
plotly.offline.plot(gdata,filename = "Correlation-Heat-Map.html")

# Scatter Plot
axis = dict(showline=False,
            zeroline=False,
            gridcolor='#fff',
            ticklen=5,
            titlefont=dict(size=14))
layout = go.Layout(
    title='Corelation Pair Plot',
    dragmode='select',
    width=1500,
    height=1500,
    autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',xaxis1=dict(axis),xaxis2=dict(axis),xaxis3=dict(axis),xaxis4=dict(axis),xaxis5=dict(axis),xaxis6=dict(axis),
    xaxis7=dict(axis),xaxis8=dict(axis),xaxis9=dict(axis),xaxis10=dict(axis),xaxis11=dict(axis),xaxis12=dict(axis),yaxis1=dict(axis),yaxis2=dict(axis),
    yaxis3=dict(axis),yaxis4=dict(axis),yaxis5=dict(axis),yaxis6=dict(axis),yaxis7=dict(axis),yaxis8=dict(axis),yaxis9=dict(axis),yaxis10=dict(axis),
    yaxis11=dict(axis),yaxis12=dict(axis)
)
trace1 = go.Splom(dimensions=[dict(label='meanfreq',values=data['meanfreq']),dict(label='sd',values=data['sd']),
                              dict(label='median',values=data['median']),dict(label='Q25',values=data['Q25']),
                              dict(label='Q75',values=data['Q75']),dict(label='IQR',values=data['IQR']),
                              dict(label='skew',values=data['skew']),dict(label='kurt',values=data['kurt']),
                              dict(label='sp.ent',values=data['sp.ent']), dict(label='sfm',values=data['sfm']),
                              dict(label='mode',values=data['mode']),dict(label='centroid',values=data['centroid'])])
fig1 = dict(data=[trace1], layout=layout)
plotly.offline.plot(fig1,filename="Corelation-Pair-Plot.html")

# Split To Training And Testing DataSet
print("######### Splitting DataSet To Training And Testing #########")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=74)

# Standardization Of Training And Testing Data
print("######### Standardizing Data #########")
scaler = StandardScaler()
transformed = scaler.fit_transform(x_train)
train = data_utils.TensorDataset(torch.from_numpy(transformed).float(),torch.from_numpy(y_train.values).float())
dataloader = data_utils.DataLoader(train, batch_size=120, shuffle=False)
scaler = StandardScaler()
transformed = scaler.fit_transform(x_test)
test_set = torch.from_numpy(transformed).float()
test_valid = torch.from_numpy(y_test.values).float()


# Model Creation Function
def create_model(model_dims):
    model = torch.nn.Sequential()
    for index, dim in enumerate(model_dims):
        if (index < len(model_dims) - 1):
            module = torch.nn.Linear(dim, model_dims[index + 1])
            init.xavier_uniform_(module.weight)
            model.add_module("linear" + str(index), module)
            torch.nn.Dropout(p=0.1)

        else:
            model.add_module("sig" + str(index), torch.nn.Sigmoid())
        if (index < len(model_dims) - 2):
            model.add_module("relu" + str(index), torch.nn.ReLU())


    return model

# Create Model And HyperParameters
print("######### Model Creation And Initialization #########")
in_shape = x_train.shape[1]
out_shape = 1
layer_dims = [in_shape, 10, 5, out_shape]   #  Hidden Layer Dimensions
model = create_model(layer_dims)
loss_fn = torch.nn.BCELoss()
learning_rate = 0.15
lambda1 = lambda epoch: .95 ** epoch
n_epochs = 100
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
# Training Model
history = {"loss": [], "accuracy": [], "loss_val": [], "accuracy_val": []}
best_accuracy=0
best_model=None
print("######### Entering Epoch Loop #########")
print("Start Pytorch Training")
start = datetime.datetime.now()
print("Start Time ",start)
for epoch in range(n_epochs):
    loss = None
    scheduler.step()
    for idx, (minibatch, target) in enumerate(dataloader):
        y_pred = model(Variable(minibatch))
        model.train()
        loss = loss_fn(y_pred, Variable(target.float()).reshape(len(Variable(target.float())), 1))
        prediction = [1 if x > 0.5 else 0 for x in y_pred.data.numpy()]
        correct = (prediction == target.numpy()).sum()
        model.eval()
        y_val_pred = model(Variable(test_set))
        loss_val = loss_fn(y_val_pred, Variable(test_valid.float()).reshape(len(test_valid.float()), 1))
        prediction_val = [1 if x > 0.5 else 0 for x in y_val_pred.data.numpy()]
        correct_val = (prediction_val == test_valid.numpy()).sum()
        model.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history["loss"].append(loss.item())
    history["accuracy"].append(100 * correct / len(prediction))
    history["loss_val"].append(loss_val.item())
    accuracy=100 * correct_val / len(prediction_val)
    if(accuracy>best_accuracy):
        best_accuracy=accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
    history["accuracy_val"].append(best_accuracy)
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1, history["loss"][-1],
          history["accuracy"][-1], history["loss_val"][-1], history["accuracy_val"][-1])
end = datetime.datetime.now()
print("End Time ",end,"\n")
print("Total Time Taken")
pytorchTotalTime=end - start
print(pytorchTotalTime,"\n")
index, value = max(enumerate(history["accuracy_val"]), key=operator.itemgetter(1))

print("Best accuracy was {} at iteration {}".format(value, index))
model.load_state_dict(best_model_wts)


#Plot Loss vs Loss_val

random_x = list(range(0,n_epochs))
random_y0 = history['loss']
random_y1 = history['loss_val']
layout= go.Layout(
    title= 'Loss Comparison',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Epoch',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Loss',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)
# Create traces
trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'lines+markers',
    name = 'loss'
)
trace1 = go.Scatter(
    x = random_x,
    y = random_y1,
    mode = 'lines+markers',
    name = 'loss_val'
)


data = [trace0, trace1]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig,filename="LossVsLoss_val.html")

# Plot Accuracy vs Accuracy_Loss
N = len(test_valid)
random_x =  list(range(0,n_epochs))
random_y0 = history['accuracy']
random_y1 = history['accuracy_val']


# Create traces
trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'lines+markers',
    name = 'accuracy'
)
trace1 = go.Scatter(
    x = random_x,
    y = random_y1,
    mode = 'lines+markers',
    name = 'accuracy_val'
)
layout= go.Layout(
    title= 'Accuracy Comparison',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Epoch',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Accuracy',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)

data = [trace0, trace1]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig,filename="AccuracyVsAccuracy_Loss.html")

# Confusion Matrix Plot
y_val_pred = model(Variable(test_set))
loss_val = loss_fn(y_val_pred, Variable(test_valid.float()).reshape(len(test_valid.float()), 1))
prediction_val = [1 if x > 0.5 else 0 for x in y_val_pred.data.numpy()]
df_ans = pd.DataFrame({'Label': test_valid.numpy().astype(int)}) # Create a dataframe for prediction and correct answer
df_ans['Prediction'] = prediction_val
cols = ['Male Actual', 'Female Actual']  # Gold standard
rows = ['Male Pred', 'Female Pred']  # diagnostic tool (our prediction)
colorscale = [[0, '#d9d9d9'], [1, '#F1C40F']]
A1P1 = len(df_ans[(df_ans['Prediction'] == df_ans['Label']) & (df_ans['Label'] == 1)])  # Actual 1 Predicted 1
A1P0 = len(df_ans[(df_ans['Prediction'] != df_ans['Label']) & (df_ans['Label'] == 1)])  # Actual 1 Predicted 0
A0P1 = len(df_ans[(df_ans['Prediction'] != df_ans['Label']) & (df_ans['Label'] == 0)])  # Actual 0 Predicted 1
A0P0 = len(df_ans[(df_ans['Prediction'] == df_ans['Label']) & (df_ans['Label'] == 0)])  # Actual 0 Predicted 0
conf = np.array([[A1P1, A0P1], [A1P0, A0P0]])
df_cm = pd.DataFrame(conf, columns=[i for i in cols], index=[i for i in rows])
z = df_cm.values
z_text = np.around(z, decimals=2)
fig = ff.create_annotated_heatmap(z, x=cols, y=rows, annotation_text=z_text, colorscale=colorscale)
fig.layout.title = 'Confusion Matrix'
fig.layout.width = 500
fig.layout.height = 500
fig.layout.yaxis.automargin = True
plotly.offline.plot(fig, filename="Confusion-Matrix.html")

def model_efficacy(conf):
    total_num = np.sum(conf)
    sen = conf[0][0] / (conf[0][0] + conf[1][0])
    spe = conf[1][1] / (conf[1][0] + conf[1][1])
    false_positive_rate = conf[0][1] / (conf[0][1] + conf[1][1])
    false_negative_rate = conf[1][0] / (conf[0][0] + conf[1][0])

    print('sensitivity: ', sen)
    print('specificity: ', spe)
    print('false_positive_rate: ', false_positive_rate)
    print('false_negative_rate: ', false_negative_rate)

    return total_num, sen, spe, false_positive_rate, false_negative_rate
print("\n")
print("######### Model Efficiency #########\n")
model_efficacy(conf)

print("######### Comparison With Keras #########\n")

import torch.nn as nn
import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing
import plotly
import plotly.graph_objs as go
from torch.utils.data import TensorDataset, DataLoader

df= pd.read_csv("E:/Study/Python/Concrete_Data_Yeh.csv")
df.columns = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age', 'strength']

sc=preprocessing.MinMaxScaler()
train, test = np.split(df.sample(frac=1), [int(.8*len(df))])
train=sc.fit_transform(train)
test=sc.fit_transform(test)

inputs = torch.Tensor(train[:,0:8])
targets = torch.Tensor(train[:,8:9])
testInputs = torch.Tensor(test[:,0:8])
testTargets = torch.Tensor(test[:,8:9])

train = TensorDataset(inputs, targets)
batch_size = 100
train_loader = DataLoader(train, batch_size, shuffle=True)
model = torch.nn.Sequential(
 torch.nn.Linear(inputs.shape[1], 20),
 torch.nn.ReLU(),
 torch.nn.Linear(20, 50),
 torch.nn.ReLU(),
 torch.nn.Linear(50, 1),
 nn.Sigmoid()
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Start Keras Training")
start = datetime.datetime.now()
print("Start Time ",start)
for i in range(1000):
    for b, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)

        if b % 100:
            print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
        # reset gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
end = datetime.datetime.now()
print("End Time ",end,"\n")
print("Total Time Taken")
kerasTotalTime=end - start
print(kerasTotalTime,"\n")
prediction=model(testInputs)
testTargetsList=[]
for i in testTargets.detach().numpy():
    testTargetsList.append(i[0])
predictKeras=[]
for i in prediction.tolist():
    predictKeras.append(i[0])

print("Mean Squared Error Pytorch ")
print(me.mean_squared_error(testTargetsList,predictKeras),"\n")
# Create a trace
trace1 = go.Scatter(
    x = testTargetsList,
    y = predictKeras,
    mode = 'markers'
)
trace2 = go.Scatter(
    x = testTargetsList,
    y = testTargetsList,
    mode = 'lines'
)
layout= go.Layout(
    title= 'Fitted Line',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Measured',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Predicted',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)
data = [trace1,trace2]
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='bar-line.html')