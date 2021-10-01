#importing the necessary libraries
#go through the code from the last to understand it
from tkinter import*
from tkinter import filedialog
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Creation function

global x,y
#to get the value from the user
entry_address = []
input_values = []
final_prediction_value={}
degree_box1=None

def getting():
    global output_column,x_poly
    global output_column
    index = 0
    for address in entry_address:
        adding_value = address.get()
        input_values.append(float(adding_value))
    for column in prediction_column:
        final_prediction_value[column]=pd.Series(input_values[index])
        index+=1
    df=pd.DataFrame(final_prediction_value)
    window6 = Tk()
    window5.destroy()
    for x in selected_dep_variables_name:
        output_column=x
    if Ml_Type == "Regression":
        if final_alg == 'Polynomial Regression':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            polyreg = PolynomialFeatures(degree=int(degre))
            x_poly = polyreg.fit_transform(df)
            y_predict_test = model.predict(x_poly)
        else:
            y_predict_test = model.predict(df)
        result_text = Label(window6, text="THE PREDICTED OUTPUT AFTER DOING THE REGRESSION IS AS FOLLOWS", font=("", 12)).place(x=100, y=200)
        result_name = Label(window6,text=output_column,font=("",12)).place(x=200,y=300,width=120)
        result=Label(window6,text=y_predict_test[0][0],font=("",12)).place(x=340,y=300)
    elif Ml_Type == "Classification":
        y_predict_test = (model.predict(df))
        result_text = Label(window6, text="THE GIVEN VALUE FALLS UNDER THE FOLLOWING CATEGORY",
                            font=("", 10)).place(x=105, y=200,width=500)
        result_name = Label(window6, text=output_column, font=("", 11)).place(x=150, y=300,width=100)

        result = Label(window6, text=str(y_predict_test[0]), font=("", 11)).place(x=280, y=300,width=190)
    window6.geometry("800x600+20+20")
    window6.configure(background="#9ba9bf")
    window6.mainloop()


selected_std_variables_index = []
selected_std_variables_name = []

def accuracy():
    y_pred = model.predict(x_test)
    from sklearn import metrics
    global accuracy_value
    accuracy_value=metrics.accuracy_score(y_test, y_pred)
def standardization():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
def advanced_option():
    global model
    if Ml_Type == "Regression":
        if final_alg == 'Simple Linear Regression' or 'Multiple Regression' :
            # for building a model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x_train, y_train)
        elif final_alg == 'Polynomial Regression':

            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            polyreg = PolynomialFeatures(degree=int(degre))
            x_poly = polyreg.fit_transform(x)
            model = LinearRegression()
            model.fit(x_poly, y)
    elif Ml_Type == "Classification":
        if final_alg == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0)
            model.fit(x_train, y_train)
            accuracy()

        elif final_alg=='Navie Bayes':

            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(x_train, y_train)
            accuracy()
        elif final_alg=='KNN':
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
            model.fit(x_train, y_train)
            accuracy()
        elif final_alg=='SVC':
            from sklearn.svm import SVC
            model = SVC(kernel=("linear"), random_state=0)
            model.fit(x_train, y_train)
            accuracy()
        elif final_alg=='Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=19, criterion="entropy", random_state=0)
            model.fit(x_train, y_train)
            accuracy()
        elif final_alg=='Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(criterion="entropy", random_state=0)
            model.fit(x_train, y_train)
            accuracy()
def getting_regression_type():
    global degre,window5,i_y,degree_box,prediction_column
    if final_alg == 'Polynomial Regression':
        degre=e12.get()
    window5=Tk()
    advanced_option()
    i_y = 200
    prediction_column = selected_indep_variables_name
    for column_name in prediction_column:
        entry = Entry(window5)
        entry.place(x=330, y=i_y)
        entry_address.append(entry)
        label = Label(window5, text=column_name).place(x=220, y=i_y)
        i_y += 50
    b = Button(window5, text="Click Here to Go for prediction", command=getting).place(x=300, y=i_y + 50)
    window5.geometry("800x600+20+20")
    window5.configure(background="#9ba9bf")
    window4.destroy()
def enhancement():
    global window4
    global e12
    entry1=10
    window4 = Tk()
    window4.geometry("800x600+20+20")
    global final_alg
    window4.configure(background="#9ba9bf")
    final_alg=select_algorithm.get()
    if select_algorithm.get() == 'Polynomial Regression':
        lab = Label(window4, text="Give the degree value  ").place(x=100, y=200)
        e12=Entry(window4)
        e12.place(x=250,y=200)


    else:
        lab = Label(window4, text=("type of algorithm                   " + select_algorithm.get())).place(x=100, y=200,width=390)

    assign = Button(window4, text="Assign",command=getting_regression_type).place(x=220, y=300, height=20, width=100)


    window3.destroy()
    window4.mainloop()
def train_model():
    global select_algorithm
    global window3
    window3 = Tk()
    window3.geometry("800x600+20+20")
    window3.configure(background="#9ba9bf")

    #spearating dep and indep variables
    dep = list(map(int, selected_dep_variables_index))
    indep = list(map(int, selected_indep_variables_index))
    global x,y
    x = dataset.iloc[:, indep].values
    y = dataset.iloc[:, dep].values

    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=1)


    name_algorithm = Label(window3, text="SELECT THE ALGORITHM TYPE", font=('', 10)).place(x=40, y=150, height=40,
                                                                                           width=300)
    if Ml_Type == "Regression":
        select_algorithm = ttk.Combobox(window3, value=['Simple Linear Regression', 'Multiple Linear', 'Polynomial Regression'])
        select_algorithm.place(x=350, y=150, height=40, width=230)
    elif Ml_Type == "Classification":
        select_algorithm = ttk.Combobox(window3, value=['Logistic Regression', 'Navie Bayes', 'KNN', 'SVC', 'Random Forest', 'Decision Tree'])
        select_algorithm.place(x=350, y=150, height=40, width=230)
    elif Ml_Type == "Clustering":
        select_algorithm = ttk.Combobox(window3, value=['kmeans', 'Herirerichal'])
        select_algorithm.place(x=350, y=150, height=40, width=230)

    type_ml_button = Button(window3, text="Next", command=enhancement).place(x=375, y=250,
                                                                           height=30, width=130)
    window2.destroy()
    window3.mainloop()

global selected_dep_variables_index,selected_dep_variables_name,selected_indep_variables_name,selected_dep_variables_index
selected_dep_variables_index=[]
selected_dep_variables_name=[]




def selecting_dep_var():
    d=my_tree.selection()
    for m in d:
        values = my_tree.item(m, 'values')
        selected_dep_variables_name.append(values)
    for D in d:
        selected_dep_variables_index.append(D)
        my_tree.delete(D)

def showing_dep_var():
    dep_var_label=Label(window2,text="DEPENDENT VARIABLE")
    dep_var_label.place(x=23,y=310)
    my_tree_dep = ttk.Treeview(window2)
    my_tree_dep.place(x=20, y=333)
    my_tree_dep["columns"] = ("Variables")
    my_tree_dep.column('#0', width=50)
    my_tree_dep.column("Variables", width=80)
    my_tree_dep.heading('#0', text="No")
    my_tree_dep.heading("Variables", text='Variables')
    sno = 1
    id = 0
    col = selected_dep_variables_name
    for co in col:
        my_tree_dep.insert(parent='', index="end", iid=id, text=sno, values=(co))
        id += 1
        sno += 1
selected_indep_variables_index=[]
selected_indep_variables_name=[]
def selecting_indep_var():
    d = my_tree.selection()
    M = my_tree.focus()
    for m in d:
        values = my_tree.item(m, 'values')
        selected_indep_variables_name.append(values)
    for D in d:
        selected_indep_variables_index.append(D)
        my_tree.delete(D)
def showing_indep_var():
    indep_var_label = Label(window2, text="INDEPENDENT VARIABLE")
    indep_var_label.place(x=174, y=310)
    my_tree_dep = ttk.Treeview(window2)
    my_tree_dep.place(x=175, y=333)
    my_tree_dep["columns"] = ("Variables")
    my_tree_dep.column('#0', width=50)
    my_tree_dep.column("Variables", width=80)
    my_tree_dep.heading('#0', text="No")
    my_tree_dep.heading("Variables", text='Variables')
    sno = 1
    id = 0
    col = selected_indep_variables_name
    for co in col:
        my_tree_dep.insert(parent='', index="end", iid=id, text=sno, values=(co))
        id += 1
        sno += 1
def re_main_page():
    window1.destroy()
    main_page()
# function to be used for confirm button

def type_ml():
    global windowtype
    windowtype = Tk()
    windowtype.geometry("800x600+20+20")
    windowtype.configure(background="#9ba9bf")
    global list_type
    lab = Label(windowtype, text="ML IMPLEMENTATION").place(x=100, y=154,width=210,height=30)
    list_type = ttk.Combobox(windowtype, value=['Regression', 'Classification'])
    list_type.place(x=350, y=150, height=40, width=230)
    type_ml_button = Button(windowtype, text="Confirm", command=Confirm).place(x=300, y=250,
                                                                                             height=30, width=130)
    window1.destroy()
    windowtype.mainloop()
def Confirm():
    global window2
    window2 = Tk()
    window2.geometry("800x600+20+20")
    window2.configure(background="#9ba9bf")
    global Ml_Type
    Ml_Type = list_type.get()
    # creating tree view
    global my_tree
    my_tree = ttk.Treeview(window2)
    my_tree.place(x=20, y=30)
    my_tree["columns"] = ("Variables")
    my_tree.column('#0', width=50)
    my_tree.column("Variables", width=80)
    my_tree.heading('#0', text="No")
    my_tree.heading("Variables", text='Variables')
    sno = 1
    id = 0
    col = dataset.columns
    for co in col:
        my_tree.insert(parent='', index="end", iid=id, text=sno, values=(co))
        id += 1
        sno += 1
    dependent_var_button = Button(window2, text="ASSIGN DEPENDENT VARIABLE", command=selecting_dep_var, width=28).place(
        x=300, y=55)
    show_dependent_var_button = Button(window2, text="CHECK DEPENDENT VARIABLE", command=showing_dep_var,
                                       width=28).place(x=300, y=105)
    independent_var_button = Button(window2, text="ASSIGN INDEPENDENT VARIABLE", command=selecting_indep_var,
                                    width=28).place(x=300, y=155)
    show_independent_var_button = Button(window2, text="CHECK INDEPENDENT VARIABLE", command=showing_indep_var,
                                         width=28).place(x=300, y=205)
    Train_Button=Button(window2,text="Next",command=train_model).place(x=400,y=400)
    windowtype.destroy()
    window2.mainloop()

def Tree_View():
    tree_frame = Frame(window1, height=300, width=300)
    tree_frame.pack(pady=100)
    # scroll bar
    tree_scroll = Scrollbar(tree_frame)
    tree_scroll.pack(side=RIGHT, fill=Y)
    tree_scroll_x = Scrollbar(tree_frame)
    tree_scroll_x.pack(side=BOTTOM, fill=X)
    # inserting scroll bar into the tree view
    tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, xscrollcommand=tree_scroll_x.set, height=20)
    tree.pack()
    # configure the scroll bar
    tree_scroll.configure(command=tree.yview)
    tree_scroll_x.configure(command=tree.xview)
    # to get the column vslues
    c = tuple(dataset.columns)
    tree["columns"] = c
    # formating the column
    tree.column('#0', width=40)
    for x in c:
        tree.column(x, width=120)
        # for creating column nmae
    tree.heading('#0', text="id")
    for z in c:
        tree.heading(z, text=z)
    global sno
    sno = 0
    w = ((dataset.index))
    for lm in w:
        tree.insert(parent='', index="end", text=sno, values=tuple(dataset.iloc[lm, :]))
        sno += 1
    tree.insert(parent='', index="end", text=sno, values=tuple(dataset.iloc[-1, :]))
def rset():
    global dataset
    input = filedialog.askopenfile(initialdir='/', title="select A file")
    dataset = pd.read_csv(input)
    global window1
    window1 = Tk()
    window1.title("Tree view")
    window1.geometry("800x600+20+20")
    window1.configure(background="#9ba9bf")
    Tree_View()
    confirm = Button(window1, text="Confirm", command=type_ml).place(x=300, y=530, height=40, width=100)
    not_confirm = Button(window1, text="Not Confirm", command=re_main_page).place(x=420, y=530, height=40, width=100)
    root.destroy()



#creating the main frame
#Main Function
def main_page():
    global root
    root=Tk()
    head=Label(root,text="PREDICTING THE VALUE FOR THE GIVEN DATSET",bg="red").place(x=220,y=200,height=35,width=300)
    root.geometry("800x600+20+20")
    root.configure(background="#9ba9bf")
    #root.configure(bg="Blue")
    e=Entry(root)
    e.place(x=150,y=400,height=40,width=280)
    button1 = Button(root, text="Browse", command=rset).place(x=550, y=400, height=40, width=100)
    root.mainloop()
main_page()
