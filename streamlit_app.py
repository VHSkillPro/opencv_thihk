import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from services import X, y, grad, cost

st.set_page_config(
    page_title="Trực quan hóa Linear Regression + Gradient Descent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Init session state
if "w" not in st.session_state:
    st.session_state.w = [np.array([[0, 0]]).T]
if "epoch" not in st.session_state:
    st.session_state.epoch = 0
if "costs" not in st.session_state:
    st.session_state.costs = []

st.title("Trực quan hóa Linear Regression + Gradient Descent")

st.header("1. Dữ liệu")
cols = st.columns(2)
cols[0].table({"X": X.flatten(), "y": y.flatten()})
cols[1].subheader("Trực quan hóa dữ liệu")
cols[1].scatter_chart(
    {"x": X.flatten(), "y": y.flatten()}, use_container_width=True, x="x", y="y"
)

st.header("2. Linear Regression + Gradient Descent")

st.write(
    """
    - **Bước 1**: Chọn **learning rate** và **số lần lặp** để thực hiện thuật toán Gradient Descent.
    - **Bước 2**: Ấn nút **Thực thi** để thực hiện thuật toán.
    - **Bước 3**: Ấn nút **Tiếp tục** để thực hiện lần lặp tiếp theo.
    - Nếu muốn thực hiện lại từ đầu, ấn nút **Thực thi** để thực hiện lại từ đầu.
    """
)

with st.form("my_form"):
    learning_rate = st.slider("Chọn learning rate", 0.01, 0.1, 0.05, 0.01)
    num_iter = st.slider("Chọn số lần lặp", 1, 100, 10, 1)
    submit = st.form_submit_button(label="Thực thi")

if submit or st.session_state.epoch > 0:
    continue_btn = st.button("Tiếp tục", disabled=(st.session_state.epoch >= num_iter))


@st.fragment()
def show_result():
    if continue_btn and st.session_state.epoch <= num_iter:
        w = st.session_state.w[-1] - learning_rate * grad(st.session_state.w[-1])
        st.session_state.costs.append(cost(w))
        st.session_state.w.append(w)

    fig, ax = plt.subplots()
    y_predicts = []
    for w in st.session_state.w:
        y_predict = w[0][0] + w[1][0] * X
        y_predicts.append(y_predict)
        ax.plot(X, y_predict)
    ax.scatter(X, y)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(st.session_state.costs)), st.session_state.costs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    cols = st.columns(3)
    cols[0].pyplot(fig, clear_figure=True)
    cols[1].pyplot(fig2, clear_figure=True)
    cols[2].table(
        {"X": X.flatten(), "y": y.flatten(), "y_predict": y_predicts[-1].flatten()}
    )


if submit:
    st.session_state.costs = [cost(st.session_state.w[-1])]
    st.session_state.epoch = 0
    st.session_state.w = [np.array([[0, 0]]).T]

if submit or st.session_state.epoch > 0:
    st.write(f"Epoch: {min(st.session_state.epoch, num_iter)} / {num_iter}")
    show_result()
    st.session_state.epoch += 1
