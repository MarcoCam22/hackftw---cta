import streamlit as st
import pickle
import numpy as np
import math
import os
# data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.impute import SimpleImputer
import joblib

# Asteroid Danger Prediction###########################
def yearDanger(df, year):
    df = df.sort_values("Year").reset_index(drop=True)
    if(int(year) < 2024):
        return "This is in the past. Prediction failed. Sorry!"
    else:
        max = -1e16
        max_name = ""
        max_year = 2024
        i = 0
        while i < len(df) and df["Year"][i] <= year:
            if df["Palermo Scale (cum.)"][i] > max:
                max = df["Palermo Scale (cum.)"][i]
                max_name = df["Object Designation"][i]
                max_year = df["Year"][i]
            i=i+1
        if max_name == "":
            return "Sorry, no asteroids found in this time range! Stay Safe!"
        else:
            return "You are in danger! " + str(max) + " is the recorded Palermo Scale value of the most dangerous asteroid in the time period from 2024 to " + str(year) + " , designated " + max_name + ". At the earliest, it could collide with Earth during the year of " + str(max_year) + "."
        
# Importing our dataset from the NASA CNEOS
def get_data():
    try:
        impact_data = pd.read_csv('cneos_sentry_summary_data.csv')
        return impact_data
    except FileNotFoundError:
        st.error('cneos_sentry_summary_data.csv file not found. Please check the file path.')
        return pd.DataFrame()
    except Exception as e:
        st.error(f'Error occurred while loading the data: {e}')
        return pd.DataFrame()

impact_data = get_data()

# Exploratory Data Analysis
# Visualizing the first rows of dataset, the dimensions, summary statistics

# Cleaning our Dataset
impact_data['Year'] = impact_data['Year Range'].str[:-5]
impact_data['Year'] = impact_data['Year'].astype(int)
impact_data = impact_data.dropna(subset=["Year", "Impact Probability (cumulative)", "Vinfinity (km/s)","H (mag)","Estimated Diameter (km)", "Palermo Scale (cum.)"])

# Designating the variables used for training, variables used for predicting
X = impact_data[["Year", "Impact Probability (cumulative)", "Vinfinity (km/s)","H (mag)","Estimated Diameter (km)"]]
y = impact_data["Palermo Scale (cum.)"]

x_train, x_test, y_train, y_test=train_test_split(X, y, train_size=0.8,random_state=1)
LR = LinearRegression()
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)

GBR = GradientBoostingRegressor()
GBR.fit(x_train,y_train)
y_pred = GBR.predict(x_test)

pickle.dump(GBR,open('model.pkl','wb'))

# Impact Measurements ###########################

# Define constants
PI = math.pi
EARTH_GRAVITY = 9.81

def calculate_impact(projectile_diameter, projectile_density, impact_velocity, impact_angle, target_density):
  """
  This function calculates the estimated crater diameter of an asteroid impact on Earth, as well as the energy of impact and danger level.

  Args:
      projectile_diameter (float): Diameter of the asteroid in meters.
      projectile_density (float): Density of the asteroid material in kg/m^3.
      impact_velocity (float): Speed of the asteroid at impact in m/s.
      impact_angle (float): Angle between the asteroid's trajectory and the horizontal plane at impact in degrees.
      target_density (float): Density of the Earth's material at the impact site in kg/m^3.

  Returns:
      float: Estimated crater diameter in meters.
  """

  # Convert impact angle to radians
  impact_angle_rad = impact_angle * PI / 180

  # Calculate impact momentum
  momentum = (PI / 4) * projectile_density * (projectile_diameter ** 3) * impact_velocity

  # Calculate effective gravity
  effective_gravity = EARTH_GRAVITY * math.cos(impact_angle_rad)

  # Apply the scaling law to estimate crater radius
  crater_radius = 0.5 * ((2 / 5) * momentum / (effective_gravity * target_density))**(1/3)

  # Calculate crater diameter
  crater_diameter = 2 * crater_radius
 # Calculate impact energy (kinetic energy)
  impact_energy = 0.5 * projectile_density * (projectile_diameter**3) * impact_velocity**2

  danger_level = "Low"  # Default danger level

  if impact_energy < 1e14:
    danger_level = "Minimal"
  elif impact_energy < 1e16:
    danger_level = "Regional" 
  elif impact_energy < 1e19:
    danger_level = "Global"
  else:
    danger_level = "Civilization Ending"
  return crater_diameter, impact_energy, danger_level


# App Structure
# first model
# first one is a machine learning that predicts palermo index
# second one is given a year, it provides future asteroid with
# second model


def main_page():
    impact_data = get_data()
    import streamlit as st
    st.header("Home")
    st.write("Project Description: Assessing Asteroid Impact Risk")
    st.write("This project aims to develop a tool for understanding the potential threat posed by asteroid impacts on Earth. It has two key components:")
    st.write("The Parmelo Index is a logarithmic scale that quantifies the potential danger level of an asteroid impact.")
    st.write("")
    st.write("Machine Learning Model:")
    st.write("A Gradient Boost Regressor model is trained on historical data from the Center for Near-Earth Object Studies (CNEOS JPL) to predict the Parmelo Index.")
    st.write("By inputting various physical characteristics of an asteroid, the model can estimate its corresponding Parmelo Index, offering a quick and informative risk assessment.")
    st.write("The entire project utilizes data from CNEOS JPL, a reputable source for information on near-Earth objects, including asteroids.")
    st.write("")
    st.write("Impact Analysis Calculations:")
    st.write("In addition to the machine learning model, the project incorporates separate calculations to assess the impact of an asteroid on Earth.")
    st.write("These calculations estimate the impact crater diameter, impact energy, and danger level of collision based on the provided asteroid parameters.")
    st.write("")
    st.write("Time analysis of Threat Level")
    st.write("The project incorporates an interesting functionality that allows users to identify the most dangerous asteroid within a specific timeframe, given a starting date and an end date. This feature utilizes the trained model and impact calculations to analyze potential threats within the specified period.")
    st.write("")
    st.write("In Conclusion:")
    st.write("Overall, this project provides a valuable tool for understanding and assessing the potential risks associated with asteroid impacts. By combining machine learning techniques with classical calculations, it offers a comprehensive and informative approach to analyzing this critical topic.")
    st.write()
    st.write(impact_data.head(5))
    st.write(impact_data.shape)
    st.write(impact_data.describe())

    st.write("Linear Regression Accuracy Score: ", LR.score(x_train,y_train))
    st.write("Linear Regression Correlation: ", r2_score(y_test,y_pred))
    st.write("Mean Squared Error: ", mean_squared_error(y_pred,y_test), "\n")

    st.write("GradientBoostRegressor Accuracy Score: ", GBR.fit(x_train,y_train).score(x_train,y_train))
    st.write("GradientBoostRegressor Mean Squared Error: ", mean_squared_error(y_test,y_pred))
    st.write("GradientBoostRegressor Correlation Score: ", r2_score(y_test,y_pred))



model = joblib.load('model.pkl')


def prediction_page():
    
    st.header("Predict Model")
    year = st.number_input("Enter Year:", min_value=2025, max_value=5000)
    impact_probability = st.number_input("Enter Impact Probability (cumulative):")
    v_infinity = st.number_input("Enter V infinity (km/s):")
    h_magnitude = st.number_input("Enter H (magnitude):")
    estimated_diameter = st.number_input("Enter Estimated Diameter (km):")
    if st.button("Predict Palermo Scale"):
    # Get the values from the input fields
        features = np.array([[year, impact_probability, v_infinity, h_magnitude, estimated_diameter]])
        prediction = model.predict(features)

        st.write("Predicted Palermo Scale:", prediction[0])

    
    if "palermo_year" not in st.session_state:
        st.session_state["palermo_year"] = 2024

    palermo_year = st.number_input("Input Year for Palermo Index estimation", st.session_state["palermo_year"])
    submit = st.button("Submit")

    if submit:
        st.session_state["palermo_year"] = palermo_year
        st.write("You have entered", palermo_year)
    st.write(yearDanger(impact_data, palermo_year))
        

def impact_page():
    st.header("Impact Calculations")
    projectile_diameter = st.number_input("Enter Projectile Diameter (m):", min_value=0.01)
    projectile_density = st.number_input("Enter Projectile Density (kg/m^3):", min_value=0.01)
    impact_velocity = st.number_input("Enter Impact Velocity (m/s):", min_value=0.01)
    impact_angle = st.number_input("Enter Impact Angle (degrees):", min_value=0.01)
    target_density = st.number_input("Enter Target Density (cumulative):", min_value=0.01)  # 5500 kg/m^3 (Earth's density)

    if st.button("Predict Palermo Scale"):
        st.write("Predicted Palermo Scale:", calculate_impact(projectile_diameter, projectile_density, impact_velocity, impact_angle, target_density))

    # Calculate the impact parameters

page_names_to_funcs = {
    "Home Page": main_page,
    "Prediction": prediction_page,
    "Impact Calculations": impact_page,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

