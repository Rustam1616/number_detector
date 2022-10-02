# %%
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.markdown(""" <style> .font {
font-size:30px ; font-family: 'Tahoma'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Please, write two numbers on the board.</p>', unsafe_allow_html=True)

# Model loading
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(r"Modelconv_net_model.ckpt",map_location=torch.device(device)))
model.to(device)

# Canvas parameters
stroke_width = st.sidebar.slider("Marker width: ", 10, 45, 20)
stroke_color = st.sidebar.color_picker("Marker color: ", "#eee")
bg_color = st.sidebar.color_picker("Background color: ")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Canvas
canvas_result = st_canvas(
    fill_color="rgb(0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=200,
    width = 400,
    drawing_mode="freedraw",
    key="canvas",
)
# Prediction function
def pre_image(image_path,model):
    img = image_path
    transform_norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Resize((28,28)),transforms.Normalize((0.1307,), (0.3081,))])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad():
        model.eval()  
        output = model(img_normalized)
        mi = output.tolist()[0].index(max(output.tolist()[0]))
        return mi
        
# Converting to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# Finding the treshold between numbers
if canvas_result.image_data is not None:
    p = 0.
    c = 0
    f1,f2 = 22,22
    for i in rgb2gray(canvas_result.image_data).T:
        if sum(i) == 0 and p != 0:
            f1 = c
            break
        c += 1
        p = sum(i)
    for i in rgb2gray(canvas_result.image_data).T[c:]:
        if sum(i) != 0 and p == 0:
            f2 = c
            break
        c += 1
        p = sum(i)
    cntr_ind = round((f1 + f2) / 2)
    
    # Result :)
    st.subheader('This numbers are: {} and {} '.format(pre_image(rgb2gray(canvas_result.image_data)[:, :cntr_ind],model),
                                    pre_image(rgb2gray(canvas_result.image_data)[:, cntr_ind:max(cntr_ind+200,400)],model)))
