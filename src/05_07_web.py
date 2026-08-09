# streamlit run 05_07_web.py
# 載入套件
import numpy as np
import streamlit as st
import torch
from torch import nn
from skimage import io
from skimage.transform import resize

# 模型載入
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
device


@st.cache_resource
def load_model() -> nn.Module:
    return torch.load('model.pt').to(device)


model = load_model()

# 標題
st.title("上傳圖片(0~9)辨識")

# 上傳圖檔
uploaded_file = st.file_uploader("上傳圖片(.png)", type="png")
if uploaded_file is not None:
    # 讀取上傳圖檔
    image1 = io.imread(uploaded_file, as_gray=True)

    # 縮為 (28, 28) 大小的影像
    image_resized = np.asarray(resize(image1, (28, 28), anti_aliasing=True))
    X1 = image_resized.reshape(1, 28, 28)  # / 255.0

    # 反轉顏色，顏色0為白色，與 RGB 色碼不同，它的 0 為黑色
    X1 = torch.FloatTensor(1 - X1).to(device)

    # 預測
    predictions = torch.softmax(model(X1), dim=1)

    # 顯示預測結果
    st.write(f'### 預測結果:{np.argmax(predictions.detach().cpu().numpy())}')

    # 顯示上傳圖檔
    st.image(image1)
