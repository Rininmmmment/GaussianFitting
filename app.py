import streamlit as st
import matplotlib
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# タイトルとアップローダー
st.title("Data Analyze")
file = st.file_uploader("File Upload", type='csv')
try:
    data = pd.read_csv(file)
except:
    data = pd.read_csv("Sample.csv")


# --------定義部分--------------
ch = data['CH']
element = data['count']  # 完成後Ba→countに変更
element_err = list(map(float, np.sqrt(element)))
gauss_max = 7 # 関数の最大個数
count = st.sidebar.number_input("Number of Functions", 1, gauss_max, 1) # 選択された関数の個数
amp = [0]*(gauss_max+1)
ctr = [0]*(gauss_max+1)
wid = [0]*(gauss_max+1)
g = [[] for i in range(gauss_max+1)]
color_list = ["", "lime", "blue", "pink", "red", "silver", "cyan", "green"]
y = 0
ctr_shoki = ch.min() + (ch.max() - ch.min()) / 2 
wid_shoki = ctr_shoki / 3
guess_total = [] # 全ての推測結果を格納するリスト
bounds = (0, np.inf)
sigma = [0]*gauss_max
FWHM = [0]*gauss_max

def sidebar(number, color):
    st.sidebar.markdown("---")
    amp[number] = st.sidebar.slider("A("+color+")", 0, int(element.max()), int(element.max()/2))
    ctr[number] = st.sidebar.slider("μ("+color+")", 0, int(ch.max()), int(ctr_shoki))
    wid[number] = st.sidebar.slider("width("+color+")", 0, 30, int(wid_shoki))
    g[number] = amp[number] * np.exp( -((ch - ctr[number])/wid[number])**2)
    plt.fill_between(ch, g[number], 0, facecolor=color, alpha=0.3)

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

def result(num): # 0=<num<count+1
    st.latex(
        r'''f_'''+str(num+1)+r'''(x) = '''+str(round(popt[num*3], 3))+r'''*\mathrm{exp}(-\frac{(x-'''+str(round(popt[1+num*3], 3))+r''')^2}{2*'''+str(round(popt[2+num*3]/np.sqrt(2), 5))+r'''^2}) \quad FWHM_'''+str(num+1)+r''' = '''+str(FWHM[num])
    )

# --------定義部分--------------

# データ表示部分
st.markdown("---")
st.subheader("Input Data")
st.dataframe(data, width=500, height=200)


# 初期値
st.markdown("---")
st.subheader("Initial Value")
for i in range(1, count+1):
    sidebar(i, color_list[i])
    y += g[i]
# plt.scatter(ch, element, s=1)
plt.errorbar(ch, element, yerr = element_err, fmt='o', markersize=2, ecolor='red', color="black")
plt.xlabel('CH')
plt.ylabel('count')
plt.plot(ch, y, ls='-', c='black', lw=1)
st.pyplot(plt)
    

#フィッティング部分
st.markdown("---")
st.subheader("Perform the Fitting")
start = st.button("fit!")
for i in range(1, count+1):
    guess_total.append(amp[i])
    guess_total.append(ctr[i])
    guess_total.append(wid[i])
if start:
    popt, pcov = curve_fit(func, ch, element, p0=guess_total, bounds=bounds)
    fit = func(ch, *popt)
    # 決定係数
    avg = np.average(element) / len(element)
    s1, s2 = 0, 0
    for i in range(len(element)):
        s1 += (element[i]- fit[i]) ** 2 # 残差変動
        s2 += (fit[i]-avg) ** 2 # 回帰による変動
        rr = s2 / (s1 + s2)

#結果の確認
st.markdown("---")
st.subheader("Display Results")
st.markdown("**"+str(file)[13:59]+"**")
st.sidebar.latex(r'''
    f(x) = A\mathrm{exp}(-\frac{(x-μ)^2}{2σ^2})
    \\μ: \mathrm{average}
    \\σ: \mathrm{standard\quad deviation}
    ''')
try:
    # 関数表示
    for i in range(count):
        sigma[i] = popt[2+i*3]/np.sqrt(2)
        FWHM[i] = 2 * sigma[i] * np.sqrt(2 * np.log(2))
        result(i)
    # 決定関数表示
    st.latex(r'''
    R^{2}='''+str(rr)+'''
    ''')

    # グラフ表示
    plt.clf()
    fig = plt.figure(figsize = (10,8))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2,sharex=ax1)
    plt.subplots_adjust(hspace=.0)

    # グラフ詳細設定
    ax1.plot(ch, fit, color="black")
    ax1.errorbar(ch, element, yerr = element_err, fmt='.', markersize=5, ecolor='red', color="blue")
    y_list = fit_plot(ch, *popt)
    for n,i in enumerate(y_list):
        ax1.fill_between(ch, i, 0, facecolor=cm.rainbow(n/len(y_list)), alpha=0.3)
    ax1.set_ylabel("count")

    ax2.scatter(ch, element-fit, s=5)
    ax2.grid(which = 'both', color='gray', linestyle='--')
    ax2.axhline(0, color='black')
    ax2.set_xlabel("CH")
    ax2.set_ylabel("ε")

    st.pyplot(plt)

except:
    print('aaa')