from tkinter import *


top = Tk ()
frame= Frame(top)
frame.pack()
top.title("Servis cancellation prediction gui")
top.minsize(800,600)
first_label =Label(text="methodology")



# Load libraries
import pandas as pd
# load dataset
pima =pd.read_csv("CustomersDataset.csv")

#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- Data Cleaning ---------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

for x in pima.index:
    #gender
    if pima.loc[x , "gender"] == "Male":
        pima.loc[x , "gender"] = 1
    else:
        pima.loc[x , "gender"] = 0

    #Partner
    if pima.loc[x , "Partner"] == "Yes":
        pima.loc[x , "Partner"] = 1
    else:
        pima.loc[x , "Partner"] = 0
    
    #Dependents
    if pima.loc[x , "Dependents"] == "Yes":
        pima.loc[x , "Dependents"] = 1
    else:
        pima.loc[x , "Dependents"] = 0
    
    #PhoneService
    if pima.loc[x , "PhoneService"] == "Yes":
        pima.loc[x , "PhoneService"] = 1
    else:
        pima.loc[x , "PhoneService"] = 0
    
    #MultipleLines
    if pima.loc[x , "MultipleLines"] == "Yes":
        pima.loc[x , "MultipleLines"] = 1
    else:
        pima.loc[x , "MultipleLines"] = 0
        
    #InternetService
    if pima.loc[x , "InternetService"] == "DSL":
        pima.loc[x , "InternetService"] = 2
    elif pima.loc[x , "InternetService"] == "Fiber optic":
        pima.loc[x , "InternetService"] = 1
    else:
        pima.loc[x , "InternetService"] = 0
      
    #OnlineSecurity
    if pima.loc[x , "OnlineSecurity"] == "Yes":
        pima.loc[x , "OnlineSecurity"] = 1
    else:
        pima.loc[x , "OnlineSecurity"] = 0
    
    #OnlineBackup
    if pima.loc[x , "OnlineBackup"] == "Yes":
        pima.loc[x , "OnlineBackup"] = 1
    else:
        pima.loc[x , "OnlineBackup"] = 0 
         
    #DeviceProtection
    if pima.loc[x , "DeviceProtection"] == "Yes":
        pima.loc[x , "DeviceProtection"] = 1
    else:
        pima.loc[x , "DeviceProtection"] = 0
        
    #TechSupport
    if pima.loc[x , "TechSupport"] == "Yes":
        pima.loc[x , "TechSupport"] = 1
    else:
        pima.loc[x , "TechSupport"] = 0
         
    #StreamingTV
    if pima.loc[x , "StreamingTV"] == "Yes":
        pima.loc[x , "StreamingTV"] = 1
    else:
        pima.loc[x , "StreamingTV"] = 0
        
    #StreamingMovies
    if pima.loc[x , "StreamingMovies"] == "Yes":
        pima.loc[x , "StreamingMovies"] = 1
    else:
        pima.loc[x , "StreamingMovies"] = 0
         
    #Contract
    if pima.loc[x , "Contract"] == "Two year":
        pima.loc[x , "Contract"] = 2
    elif pima.loc[x , "Contract"] == "One year":
        pima.loc[x , "Contract"] = 1
    else:
        pima.loc[x , "Contract"] = 0
         
    #PaperlessBilling
    if pima.loc[x , "PaperlessBilling"] == "Yes":
        pima.loc[x , "PaperlessBilling"] = 1
    else:
        pima.loc[x , "PaperlessBilling"] = 0
         
    #PaymentMethod
    if pima.loc[x , "PaymentMethod"] == "Credit card (automatic)":
        pima.loc[x , "PaymentMethod"] = 3
    elif pima.loc[x , "PaymentMethod"] == "Bank transfer (automatic)":
        pima.loc[x , "PaymentMethod"] = 2
    elif pima.loc[x , "PaymentMethod"] == "Electronic check":
        pima.loc[x , "PaymentMethod"] = 1
    else:
        pima.loc[x , "PaymentMethod"] = 0
        
    #TotalCharges
    if pima.loc[x , "TotalCharges"] == " ":
        pima.drop(x, inplace=True)
         

#print(pima)


#split dataset in features and target variable
feature_cols = [ 'SeniorCitizen',  'Dependents','tenure','MultipleLines','InternetService',
                'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'MonthlyCharges','PhoneService']

X = pima[feature_cols] # Features
y = pima.Churn # Target variable


#dinput.insert(17, 33.33)

#y = pima.Churn # Target variable

from sklearn.model_selection import train_test_split # Import train_test_split function
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# Model Accuracy, how often is the classifier correct?

#---------------------------------------------- tree ---------------------------------------------#
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# Create Decision Tree classifer object
treeClf = DecisionTreeClassifier()
# Train Decision Tree Classifer
treeClf = treeClf.fit(X_train,y_train)

#------------------------------------------- svm -------------------------------------------------#
#Import svm model
from sklearn import svm
#Create a svm Classifier
svmClf = svm.SVC() 
#Train the model using the training sets
svmClf.fit(X_train, y_train)

#------------------------------------ logistic regression -----------------------------------------#
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
lrClf = LogisticRegression()
# fit the model with data
lrClf.fit(X_train,y_train)

#------------------------------------------------- GUI ------------------------------------------------#
clf=0
def sel():
    selection = "You selected the option " + str(var.get())
    label.config(text = selection)
    print("variable = ", var.get())
        # # # # CHOOSE # # # # 
    if var.get() == 2 :
        clfg=treeClf
        print("tree")
    elif var.get() == 1 :
        clfg=svmClf
        print("svm")
    elif var.get() == 3 :
        clfg=lrClf
        print("lr")
    global clf
    clf = clfg

var = IntVar()
R1 = Radiobutton(frame, text="SVM", variable=var, value=1,command=sel)
R1.pack(side= LEFT)
R2 = Radiobutton(frame, text="ID3", variable=var, value=2, command=sel)
R2.pack(side=LEFT)
R3 = Radiobutton(frame, text="logistec regression", variable=var, value=3,command=sel)
R3.pack(side= LEFT)

label = Label(top)
first_label.pack()
label.pack()

label2= Label(text="customer data")
label2.pack( anchor = W)

L1 = Label(top, text="Customer ID")
L1.pack( anchor = W)

E1 = Entry(top, bd =3)
E1.pack(anchor = W)

L2 = Label(top, text="Gender")
L2.pack( anchor = W)

E2 = Entry(top, bd =3)
E2.pack(anchor = W)

L3 = Label(top, text="senior citizen")
L3.pack( anchor = W)

E3 = Entry(top, bd =3)
E3.pack(anchor = W)

L4 = Label(top, text="partner")
L4.pack( anchor = W)

E4 = Entry(top, bd =3)
E4.pack(anchor = W)

L5 = Label(top, text="dependence")
L5.pack( anchor = W)

E5 = Entry(top, bd =3)
E5.pack(anchor = W)

L6 = Label(top, text="tenure")
L6.pack( anchor = W)

E6 = Entry(top, bd =3)
E6.pack(anchor = W)

L7 = Label(top, text="phone service")
L7.pack( anchor = W)

E7 = Entry(top, bd =3)
E7.pack(anchor = W)

L8 = Label(top, text="multiple lines")
L8.pack( anchor = W)

E8 = Entry(top, bd =3)
E8.pack(anchor = W)

L9 = Label(top, text="internet service")
L9.pack( anchor = W)

E9 = Entry(top, bd =3)
E9.pack(anchor = W)

L10 = Label(top, text="online security")
L10.pack( anchor = E)
L10.place(x=300,y=90)

E10 = Entry(top, bd =3)
E10.pack(anchor = E)
E10.place(x=300,y=110)

L11 = Label(top, text="online backup")
L11.pack( anchor = E)
L11.place(x=300,y=135)

E11 = Entry(top, bd =3)
E11.pack(anchor = E)
E11.place(x=300,y=155)

L12 = Label(top, text="device protection")
L12.pack( anchor = E)
L12.place(x=300,y=180)

E12 = Entry(top, bd =3)
E12.pack(anchor = E)
E12.place(x=300,y=200)

L13 = Label(top, text="tech support")
L13.pack( anchor = E)
L13.place(x=300,y=225)

E13 = Entry(top, bd =3)
E13.pack(anchor = E)
E13.place(x=300,y=245)

L14 = Label(top, text="streaming tv")
L14.pack( anchor = E)
L14.place(x=300,y=270)

E14 = Entry(top, bd =3)
E14.pack(anchor = E)
E14.place(x=300,y=290)

L15 = Label(top, text="streaming movies")
L15.pack( anchor = E)
L15.place(x=300,y=315)

E15= Entry(top, bd =3)
E15.pack(anchor = E)
E15.place(x=300,y=335)

L16 = Label(top, text="contract")
L16.pack( anchor = E)
L16.place(x=300,y=360)

E16 = Entry(top, bd =3)
E16.pack(anchor = E)
E16.place(x=300,y=380)

L17 = Label(top, text="paperless billing")
L17.pack( anchor = E)
L17.place(x=300,y=405)

E17 = Entry(top, bd =3)
E17.pack(anchor = E)
E17.place(x=300,y=425)

L18 = Label(top, text="payment method")
L18.pack( anchor = E)
L18.place(x=540,y=90)

E18 = Entry(top, bd =3)
E18.pack(anchor = E)
E18.place(x=540,y=110)

L19 = Label(top, text="monthly charges")
L19.pack( anchor = E)
L19.place(x=540,y=135)

E19 = Entry(top, bd =3)
E19.pack(anchor = E)
E19.place(x=540,y=155)

L20 = Label(top, text="total charges")
L20.pack( anchor = E)
L20.place(x=540,y=180)

E20 = Entry(top, bd =3)
E20.pack(anchor = E)
E20.place(x=540,y=200)



dinput=[]

def getAcc():
    y_pred=clf.predict(X_test)
    # import the metrics class
    from sklearn import metrics
    print("Accuracy :",metrics.accuracy_score(y_test, y_pred))
    print(" ")
    labeltst=Label(text="Accuracy:"+str(metrics.accuracy_score(y_test,y_pred)))
    labeltst.pack()
    labeltst.place(x=393,y=470)

def getPredection():
    
    dinput.clear()
    #senion citizen
    dinput.insert(1, E3.get())    
    #Dependents
    if E5.get() == "Yes":
        dinput.insert(3, 1)
    else:
        dinput.insert(3, 0)
    #tenur
    dinput.insert(4, E6.get())
    #PhoneService
    if E7.get()== "Yes":
        dinput.insert(5, 1)
    else:
        dinput.insert(5, 0)
    #MultipleLines
    if E8.get() == "Yes":
        dinput.insert(6, 1)
    else:
        dinput.insert(6, 0)
    #InternetService
    if E9.get()== "DSL":
        dinput.insert(7, 2)
    elif E9.get() == "Fiber optic":
        dinput.insert(7, 1)
    else:
        dinput.insert(7, 0)
    #OnlineSecurity
    if E10.get() == "Yes":
        dinput.insert(8, 1)
    else:
        dinput.insert(8, 0)
    #OnlineBackup
    if E11.get()== "Yes":
        dinput.insert(9, 1)
    else:
        dinput.insert(9, 0) 
    #DeviceProtection
    if E12.get()== "Yes":
        dinput.insert(10, 1)
    else:
        dinput.insert(10, 0)
    #TechSupport
    if E13.get() == "Yes":
        dinput.insert(11, 1)
    else:
        dinput.insert(11, 0)
    #StreamingTV
    if E14.get()== "Yes":
        dinput.insert(12, 1)
    else:
        dinput.insert(12, 0)
    #StreamingMovies
    if E15.get() == "Yes":
        dinput.insert(13, 1)
    else:
        dinput.insert(13, 0)
    #Contract
    if E16.get()== "Two year":
        dinput.insert(14, 2)
    elif  E16.get() == "One year":
        dinput.insert(14, 1)
    else:
        dinput.insert(14, 0)
    #PaperlessBilling
    if E17.get() == "Yes":
        dinput.insert(15, 1)
    else:
        dinput.insert(15, 0)
    #monthly charges
    dinput.insert(17, E19.get())    
    
    ourpred = clf.predict([dinput])
    resultLabel =Label(text="the predection is"+str(ourpred))
    resultLabel.pack()
    resultLabel.place(x=393,y=530)

    print(dinput)
    print(str(ourpred))

B = Button(top, text ="predict",command=getPredection)
B.pack()
B.place(x =393,y=500)

B3 = Button(top, text ="test",command=getAcc)
B3.pack()
B3.place(x=393,y=450)

#----------------------------------------
top.mainloop()