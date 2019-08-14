import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
tf.logging.set_verbosity(tf.logging.INFO)
import datetime
from bokeh.plotting import figure, output_file, show
import bokeh.io.output as out
from bokeh.layouts import gridplot
from bokeh.colors import RGB
from bokeh.layouts import row

# Importing DataSet
print("Importing DataSet")
df= pd.read_csv("E:/Study/Python/Keras/Concrete_Data_Yeh.csv")
df.columns = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age', 'strength']

# Checking For Null Values
print("Check Null Values")
print(df.isnull().sum().sort_values(ascending=False))
print("No Null Values Are Found")

# Scatter Plot Function
def mscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=15,
              line_color="navy", fill_color="orange", alpha=0.5)

# Scatter Plot
scatterList = []
tools="pan,box_zoom,reset,save"
for i in df.columns:
    for j in df.columns:
        p = figure(title="SactterPlot : " + i + " Vs " + j, tools=tools)
        p.grid.grid_line_color = None
        p.background_fill_color = "#eeeeee"
        mscatter(p, df[i], df[j], "asterisk")
        p.xaxis.axis_label = i
        p.yaxis.axis_label = j
        scatterList.append(p)
output_file("scatterPlot.html")
show(gridplot(scatterList, ncols=9, plot_width=250, plot_height=250))
out.reset_output()

# Histogram Function
def histPlot(data,edge,title,tools,xlab,ylab):
    p=figure(title=title,tools=tools)
    p.quad(top=his,bottom=0,left=edge[:-1],right=edge[1:])
    p.xaxis.axis_label=xlab
    p.yaxis.axis_label=ylab
    return p

# Histogram Plot
histList=[]
for i in df.columns:
    his,edg=np.histogram(df[i],density=True,bins=50)
    p=histPlot(his,edg,i,tools,i,"Frequency")
    histList.append(p)
output_file("histogramPlot.html")
show(gridplot(histList,ncols=4, plot_width=250, plot_height=250, toolbar_location=None))
out.reset_output()

# Corelation Plot Function
def myCorrelationPlot(df, pw=300, ph=300, tt="correlations"):
    colNames = df.columns.tolist()
    rownames = df.index.tolist()
    x = [c for r in rownames for c in colNames]
    y = [r for r in rownames for c in colNames]
    corarr = [df[c][r] for r in rownames for c in colNames]
    colors = [RGB(255*(1-x)/2,255*(1+x)/2,0,0.7) for x in corarr]
    p = figure(title=tt, x_range=colNames, y_range=rownames, plot_width=pw, plot_height=ph, toolbar_location="right")
    p.rect(x, y, color=colors, width=1, height=1)
    p.xaxis.major_label_orientation = 3.14159/2
    c = myColorBar(75, ph)
    output_file("correlation.html")
    show((row(p,c)))
    out.reset_output()

# Correlation Color Bar Function
def myColorBar(pw=75, ph=300, tt="colors"):
    from bokeh.models import CategoricalAxis
    corarr = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = [RGB(255*(1-x)/2,255*(1+x)/2,0,0.7) for x in corarr]
    colNames = ["color"]
    rownames = [str(x) for x in corarr]
    x = [c for r in rownames for c in colNames]
    y = [r for r in rownames for c in colNames]
    p = figure(title=tt, x_range=colNames, y_range=rownames, plot_width=pw, plot_height=ph, toolbar_location=None)
    p.rect(x, y, color=colors, width=1, height=1)
    p.xaxis.major_label_orientation = 3.14159/2
    return p

myCorrelationPlot(df.corr())

# Split DataSet To Train And Test
train, test = np.split(df.sample(frac=1), [int(.9*len(df))])
train=preprocessing.normalize(train, norm='l2')
test=preprocessing.normalize(test, norm='l2')

# Standardization Of Data
sc=preprocessing.MinMaxScaler()
train, test = np.split(df.sample(frac=1), [int(.9*len(df))])
train=sc.fit_transform(train)
test=sc.fit_transform(test)

# Slicing XTrain And YTrain Values
yTrain=train[:,8]
yTest=test[:,8]
xTest=test[:,0:8]
xTrain=train[:,0:8]

# Model Creation Function
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=xTrain.shape[1], activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error',
              metrics=['mae'])
    return model


# Model Training
model = create_model()
n_epoch=1000
print("Start Keras Training")
start = datetime.datetime.now()
print("Start Time ",start)
history = model.fit(xTrain, yTrain, validation_data=(xTest,yTest), epochs=n_epoch, batch_size=100)

end = datetime.datetime.now()
print("End Time ",end,"\n")
print("Total Time Taken")
pytorchTotalTime=end - start
print(pytorchTotalTime,"\n")

# evaluate the model
print("Model Evaluation")
scores = model.evaluate(xTrain, yTrain)
print(model.metrics_names[1], scores[1]*100)
print(model.metrics_names[0], scores[0]*100)

# MAE Plot
p = figure(
    tools="pan,box_zoom,reset,save",
    x_axis_label='Epoch', y_axis_label='MAE',title="Model MAE"
)
p.line(range(0, n_epoch), history.history['mean_absolute_error'], legend="MAE", line_color="blue")
p.line(range(0, n_epoch), history.history['val_mean_absolute_error'], legend="Predicted MAE", line_color="red")
output_file("mae.html")
show(p)
out.reset_output()

# MSE Plot
p = figure(
    tools="pan,box_zoom,reset,save",
    x_axis_label='Epoch', y_axis_label='MSE', title="Model MSE"
)
p.line(range(0, n_epoch), history.history['loss'], legend="MSE", line_color="blue")
p.line(range(0, n_epoch), history.history['val_loss'], legend="Predicted MSE", line_color="red")
output_file("mse.html")
show(p)
out.reset_output()

#Prediction
prediction = model.predict(xTest)
scores = model.evaluate(xTest, yTest, verbose=1)
print(scores)

# Actual Vs Prediction Plot
p = figure(
    tools="pan,box_zoom,reset,save",
    x_axis_label='Epoch', y_axis_label='MSE', title="Model MSE"
)
p.line(range(0, 103), yTest.tolist(), legend="Actual Value", line_color="blue")
p.line(range(0, 103), prediction.tolist(), legend="Predicted Value", line_color="red")
output_file("predictionActual.html")
show(p)
out.reset_output()

# Comparison Plot
predictList=[]
for i in prediction.tolist():
    predictList.append(i[0])
p = figure(title = type(model).__name__, plot_width=400, plot_height=400)
p.circle(yTest.tolist(), predictList, size=3, color="red", alpha=1)
p.line(yTest.tolist(), yTest.tolist(), color="blue")
p.xaxis.axis_label = "measured"
p.yaxis.axis_label = "predicted"
output_file("bestfit.html")
show(p)
out.reset_output()

# MSE And MAE
for i, j in enumerate(model.metrics_names):
    print(model.metrics_names[i], ": ", scores[i] * 100, "%")
