# Traffic_Monitoring_Streamlit

This project is a **traffic monitoring application** built with Python and **Streamlit**. The goal is to visualize and analyze traffic data in real time, leveraging the simplicity and interactivity of the Streamlit framework. The app can be used for monitoring traffic trends, analyzing vehicle counts, and more.

![Traffic Monitoring](path_to_screenshot.png)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- 📊 **Real-time Traffic Monitoring**: Displays traffic data with live updates.
- 📈 **Data Visualization**: Offers various visualizations to better understand traffic patterns.
- 🛠️ **User-Friendly Interface**: Built with Streamlit for easy interaction.
- 📅 **Historical Analysis**: Supports visualization of historical traffic data trends.

## Installation

###

###Streamlit

To run the project locally, follow these steps:

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/DarrenCheww/Traffic_Monitoring_Streamlit.git
2. **Install requirements**:

   ```bash
   pip install -r requirements.txt
3. Fill in API keys in secrets.toml file
   -Consists of the following
   1) MY_GEO_API_KEY 
   2) MY_GOOGLE_API_KEY 
   3) MY_LOCATION_IQ_KEY 
   4) MY_TOM_TOM_API_KEY 

   [database]
   5) MY_MONGO_DB_USER_NAME 
   6) MY_MONGO_DB_PASSWORD 
   7) MY_MONGO_CLUSTER 
   
4.**Run Streamlit Application**
   ```bash
   streamlit run streamlittesting.py


