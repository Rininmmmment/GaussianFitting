import streamlit as st
import matplotlib
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

st.title("Data Analyze")
file = st.file_uploader("File Upload", type='csv')
# アップロード部分
try:
    data = pd.read_csv(file)
except:
    data = pd.read_csv("Ba.csv")
    
ch = data['CH']
element = data['count']  # 完成後Ba→countに変更

#データ表示部分
st.markdown("---")
st.subheader("Input Data")
st.dataframe(data, width=500, height=200)

#G初期値
plt.scatter(ch, element, s=1)
plt.xlabel('CH')
plt.ylabel('count')

st.markdown("---")
st.subheader("Initial Value")

shoki = ch.min() + (ch.max() - ch.min()) / 2
count = st.sidebar.number_input("Number of Functions", 1, 4, 1)
st.sidebar.markdown("---")
amp1 = st.sidebar.slider("A(green)", 0, int(element.max()), int(element.max()/2))
ctr1 = st.sidebar.slider("μ(green)", 0, ch.max(), int(shoki))
wid1 = st.sidebar.slider("width(green)", 0, 30, int(shoki/10))
g1 = amp1 * np.exp( -((ch - ctr1)/wid1)**2)
plt.fill_between(ch, g1, 0, facecolor='lime', alpha=0.3)
y = g1

if count > 1:
    st.sidebar.markdown("---")
    amp2 = st.sidebar.slider("A(blue)", 0, int(element.max()), int(element.max()/2))
    ctr2 = st.sidebar.slider("μ(blue)", 0, ch.max(), int(shoki)+5)
    wid2 = st.sidebar.slider("width(blue)", 0, 30, int(shoki/10))
    g2 = amp2 * np.exp( -((ch - ctr2)/wid2)**2)
    plt.fill_between(ch, g2, 0, facecolor='blue', alpha=0.3)
    y = g1 + g2

if count > 2:
    st.sidebar.markdown("---")
    amp3 = st.sidebar.slider("A(pink)", 0, int(element.max()), int(element.max()/2))
    ctr3 = st.sidebar.slider("μ(pink)", 0, ch.max(), int(shoki)+10)
    wid3 = st.sidebar.slider("width(pink)", 0, 30, int(shoki/10))
    g3 = amp3 * np.exp( -((ch - ctr3)/wid3)**2)
    plt.fill_between(ch, g3, 0, facecolor='pink', alpha=0.3)
    y = g1 + g2 + g3

if count > 3:
    st.sidebar.markdown("---")
    amp4 = st.sidebar.slider("A(red)", 0, int(element.max()), int(element.max()/2))
    ctr4 = st.sidebar.slider("μ(red)", 0, ch.max(), int(shoki)+15)
    wid4 = st.sidebar.slider("width(red)", 0, 30, int(shoki/10))
    g4 = amp4 * np.exp( -((ch - ctr4)/wid4)**2)
    plt.fill_between(ch, g4, 0, facecolor='red', alpha=0.3)
    y = g1 + g2 + g3 + g4

plt.plot(ch, y , ls='-', c='black', lw=1)
st.pyplot(plt)
    
#フィッティング部分
st.markdown("---")
st.subheader("Perform the Fitting")
start = st.button("fit!")

guess_total = []
if count == 1:
    guess_total = [amp1, ctr1, wid1]
elif count == 2:
    guess_total = [amp1, ctr1, wid1, amp2, ctr2, wid2]
elif count == 3:
    guess_total = [amp1, ctr1, wid1, amp2, ctr2, wid2, amp3, ctr3, wid3]
elif count == 4:
    guess_total = [amp1, ctr1, wid1, amp2, ctr2, wid2, amp3, ctr3, wid3, amp4, ctr4, wid4]

bounds = (0, np.inf)

def func(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i
#     y_sum = y_sum + params[-1]
    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)
    return y_list

if start:
    popt, pcov = curve_fit(func, ch, element, p0=guess_total, bounds=bounds)
    
#結果の確認
st.markdown("---")
st.subheader("Display Results")
st.markdown("**"+str(file)[13:59]+"**")
st.sidebar.latex(r'''
    f(x) = Aexp(-\frac{(x-μ)^2}{2σ^2})
    \\μ: average, 
    \\σ: standard\quad deviation
    ''')
st.write("")
try:
    if count > 0:
        sigma1 = popt[2]/np.sqrt(2)
        FWHM1 = 2 * sigma1 * np.sqrt(2 * np.log(2))
        st.latex(r'''
    f_1(x) = '''+str(round(popt[0], 3))+r'''*exp(-\frac{(x-'''+str(round(popt[1], 3))+r''')^2}{2*'''+str(round(popt[2]/np.sqrt(2), 5))+r'''^2})''')
        st.latex(r'''FWHM_1 = '''+str(FWHM1))
    if count > 1:
        sigma2 = popt[5]/np.sqrt(2)
        FWHM2 = 2 * sigma2 * np.sqrt(2 * np.log(2))
        st.latex(r'''
    f_2(x) = '''+str(round(popt[3], 3))+r'''*exp(-\frac{(x-'''+str(round(popt[4], 3))+r''')^2}{2*'''+str(round(popt[5]/np.sqrt(2), 5))+r'''^2})
    ''')
        st.latex(r'''FWHM_2 = '''+str(FWHM2))
    if count > 2:
        sigma3 = popt[8]/np.sqrt(2)
        FWHM3 = 2 * sigma1 * np.sqrt(2 * np.log(2))
        st.latex(r'''
    f_3(x) = '''+str(round(popt[6], 3))+r'''*exp(-\frac{(x-'''+str(round(popt[7], 3))+r''')^2}{2*'''+str(round(popt[8]/np.sqrt(2), 5))+r'''^2})
    ''')
        st.latex(r'''FWHM_3 = '''+str(FWHM3))
    if count > 3:
        sigma4 = popt[11]/np.sqrt(2)
        FWHM4 = 2 * sigma2 * np.sqrt(2 * np.log(2))
        st.latex(r'''
    f_4(x) = '''+str(round(popt[9], 3))+r'''*exp(-\frac{(x-'''+str(round(popt[10], 3))+r''')^2}{2*'''+str(round(popt[11]/np.sqrt(2), 5))+r'''^2})
    ''')
        st.latex(r'''FWHM_4 = '''+str(FWHM4))
#     st.dataframe(popt, width=500, height=200)
    plt.clf()
    fit = func(ch, *popt)
    plt.scatter(ch, element, s=1)
    plt.plot(ch, fit , ls='-', c='black', lw=0.7)
    plt.xlabel('CH')
    plt.ylabel('count')
    y_list = fit_plot(ch, *popt)
    for n,i in enumerate(y_list):
        plt.fill_between(ch, i, 0, facecolor=cm.rainbow(n/len(y_list)), alpha=0.3)
    st.pyplot(plt)
except:
    print('変数が存在しません')