import streamlit as st 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
data={
    "Studyhours":[2,3,4,5,6,7,8,1,9,10],
    "Attendence":[60,65,70,75,80,85,90,55,95,98],
    "Assignments":[2,3,4,5,6,7,8,1,9,10],
    "Marks":[45,50,55,60,70,75,85,40,90,95]
}
df=pd.DataFrame(data)
print(df)

x=df[["Studyhours","Attendence","Assignments"]]
y=df["Marks"]
model=LinearRegression()
model.fit(x,y)

#Title
st.title("Stuent Performance Prediction System")

#Show dataset
st.subheader("Student Dataset")
st.dataframe(df)

#Inputs
st.subheader("Enter Student Details")
Studyhours=st.number_input("Studyhours",min_value=0,max_value=15,value=5)
Attendence=st.number_input("Attendence",min_value=0,max_value=100,value=80)
Assignments=st.number_input("Assignments",min_value=0,max_value=10,value=5)

#prediction
if st.button("Predict Marks"):
    prediction=model.predict([Studyhours,Attendence,Assignments])
    st.success(f"Predicated Marks:{prediction[0]:.2f}")
#chart
st.subheader("Study Hours vs Marks")
fig,ax=plt.subplots()
ax.scatter(df["Studyhours"],df["Marks"])
ax.set_xlabel("StudyHours")
ax.set_ylabel("Marks")
ax.set_title("Studyhours vs Marks")
st.pyplot(fig)
