import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from libsvm.svmutil import *
import streamlit as st


# load model vào cache, không cần reload lại mỗi lần thực hiên predict
@st.cache(allow_output_mutation=True)
def reload_model():
    filename = './libsvm.model'
    loaded_model = svm_load_model(filename)
    return loaded_model

# lấy danh sách các thuộc tính
def load_zoo():
    zoo_path = '../datasets/zoo.csv'
    df = pd.read_csv(zoo_path)
    return [col for col in df.columns if col not in ['animal_name','class_type']]

# lấy danh sách các lớp phân loại
def load_class():
    class_path = '../datasets/class.csv'
    df = pd.read_csv(class_path)

    d = df.set_index('Class_Number')['Class_Type'].to_dict()
    return d #df[['Class_Number','Class_Type']]

# tạo check list
def check_list():
    list = load_zoo()
    X = [0]*len(list)
    newlist = st.sidebar.multiselect('select properties',[i for i in list if i != 'legs'])
    legs = st.sidebar.slider('legs?', 0, 8, 0, 2)
    if st.sidebar.button('submit'):
        for index, i in enumerate(list):
            if i in newlist:
                X[index] = 1
            elif i == 'legs':
                X[index] = legs
            else:
                X[index] = 0
        return X

# thực hiện dự đoán
def predtion(model, arr):
    return svm_predict([],[np.array(arr)],model,"-b 1")

# hiển thị biểu đồ
def show_plot(list,list2):
    #for s in plt.style.available:
    plt.style.use('seaborn-pastel')
    #st.write(s)
    fig1, ax1 = plt.subplots(figsize=(20, 12))
    ax1.pie(list, autopct='%1.1f%%', wedgeprops = {'linewidth': 1, 'edgecolor' :'gray'})
    ax1.axis('equal')
    ax1.legend(list2, fontsize = 17)

    st.pyplot(fig1)

def main():
    model = reload_model()
    cl = load_class()
    X = check_list()
    
    if X:
        pred = predtion(model, X)
        list1, list2 = zip(*sorted(zip(model.get_labels(), pred[2][0])))
        show_plot(list2,list(cl.values()))
        
if __name__ == '__main__':
    main()