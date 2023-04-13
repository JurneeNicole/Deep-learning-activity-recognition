import torch
from torch.utils.data import DataLoader
from data_preprocess import load_data, ToTensor
from network import Network
import gradio as gr
import numpy as np
from torch.utils.data import TensorDataset
import warnings
warnings.filterwarnings('ignore')

# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY
def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]

# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data(data_folder):
    import os
    print(data_folder)
    if os.path.isfile(data_folder + '/data_har.npz') == True:
        data = np.load(data_folder + '/data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = data_folder + 'UCI HAR Dataset/UCI HAR Dataset/'
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
        
        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                          item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'


        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(Y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(Y_test))

    return train_dataset, test_dataset


# Load the trained model weights
model = Network()
model.load_state_dict(torch.load('model.pth'))

# Load the preprocessed UCI HAR dataset
train_dataset, test_dataset = load_data('')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

activity_mapping = {
    0: "Downstairs",
    1: "Jogging",
    2: "Sitting",
    3: "Standing",
    4: "Upstairs",
    5: "Walking"
}

def predict(x):
    # Repeat the input sequence 9 times along the first dimension to match the input channels
    x = [float(i) for i in x.split(",")]
    x = torch.Tensor(x).repeat(9, 1).reshape(1, 9, 1, 128)
    output = model(x)
    _, predicted = torch.max(output.data, 1)
    return activity_mapping[int(predicted)]

iface = gr.Interface(fn=predict, 
                     inputs=gr.inputs.Textbox(label="Sensor Data", lines=128),
                     outputs=gr.outputs.Textbox(label="Activity Prediction"))


iface.launch()
