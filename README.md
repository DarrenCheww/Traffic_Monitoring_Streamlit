# Traffic Monitoring Streamlit

## Overview

This Traffic Monitoring application is a powerful Python-based tool built with Streamlit, designed to provide real-time insights into traffic data in Singapore's Context. The application offers comprehensive visualization and analysis capabilities, making it easy to understand traffic patterns and trends.

![Traffic Monitoring Dashboard](path_to_screenshot.png)

## 🌟 Features
- **📊 Real-time Traffic Monitoring**
  - Live updates of traffic data
  - Instant insights into current traffic conditions

- **📈 Advanced Data Visualization**
  - Multiple chart types and graphical representations
  - Comprehensive traffic pattern analysis

- **🛠️ Intuitive User Interface**
  - Seamless interaction with Streamlit
  - User-friendly design for easy navigation

- **📅 Historical Data Analysis**
  - In-depth trend visualization
  - Comparative traffic data analysis

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DarrenCheww/Traffic_Monitoring_Streamlit.git
   cd Traffic_Monitoring_Streamlit
   ```

2. **Create Virtual Environment** (Optional but Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys and Credentials**
   Create a `.streamlit/secrets.toml` file with the following structure:
   ```toml
   # Geocoding and Map APIs
   MY_GEO_API_KEY = "your_geo_api_key"
   MY_GOOGLE_API_KEY = "your_google_api_key"
   MY_LOCATION_IQ_KEY = "your_location_iq_key"
   MY_TOM_TOM_API_KEY = "your_tom_tom_api_key"

   # Database Credentials
   [database]
   MY_MONGO_DB_USER_NAME = "your_mongodb_username"
   MY_MONGO_DB_PASSWORD = "your_mongodb_password"
   MY_MONGO_CLUSTER = "your_mongodb_cluster"
   ```

5. **Run the Streamlit Application**
   ```bash
   streamlit run streamlittesting.py
   ```

## 🔧 Technologies Used

- **Language**: Python
- **Web Framework**: Streamlit
- **Database**: MongoDB
- **APIs**: 
  - Geocoding APIs
  - Google Maps API
  - Location IQ
  - Tom Tom API
  - Roboflow

- **BackEnd**
  - Azure Functions
  - YoloV8
    

## 📂 Project Structure

```
Traffic_Monitoring_Streamlit/
│
├── streamlittesting.py      # Main Streamlit application
├── requirements.txt         # Project dependencies
├── .streamlit/
│   └── secrets.toml         # API keys and credentials
```

```
Azure Functions/
│
├── function_app.py      # Main Azure function
├── requirements.txt     # Project dependencies
├── host.json            # Azure Function Parameters
├── ID_location.json     # CameraID to Location Name in place of Google API
└── sample2.json         # Annotated Constraint Polygons File
```

```
Machine Learning/
│
├── ML Traffic Training.py      # Main Azure function
├── best.pt                     # Best Training File
```


## 📞 Contact

[Darren Chew] - [darrenchew123@gmail.com]

Project Link: [https://github.com/DarrenCheww/Traffic_Monitoring_Streamlit](https://github.com/DarrenCheww/Traffic_Monitoring_Streamlit)
