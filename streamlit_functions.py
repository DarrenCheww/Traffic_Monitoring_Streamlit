import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv, dotenv_values
import pandas as pd
import pydeck as pdk
import math
import random
import numpy as np
from numpy import sin, cos, arcsin, sqrt
from streamlit_folium import st_folium
import folium
import branca
import base64
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
import polyline
from collections import Counter
import statistics
import plotly.express as px
from datetime import time, timedelta
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import OrderedDict
import asyncio
import aiofiles
import aiohttp
import time
from rtree import index
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
load_dotenv()

camera_icon_url =  "https://cdn-icons-png.flaticon.com/128/685/685655.png"
car_speed_icon_url = "https://static.thenounproject.com/png/2603319-200.png"
graph_icon_url = "https://cdn-icons-png.flaticon.com/128/404/404723.png"

def most_frequent_element(lst):
    # Count the frequency of each element
    print("Check Most Freq List:", lst)
    frequency = Counter(lst)
    
    # Find the most common element
    try:
        most_common_element, _ = frequency.most_common(1)[0]
    except:
        return None
    
    return most_common_element

def randomcolorvalue():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    color_code = "#{:02x}{:02x}{:02x}".format(red, green, blue)
    return color_code


def all_coords_selected():
    if len(st.session_state["option_selected_coordinate_arr"])<st.session_state.num_of_dest+1 :
        # st.session_state.all_selected = False
        print("Triggered 1")
        return False

    for i in range(st.session_state.num_of_dest+1):
        if not st.session_state["option_selected_coordinate_arr"][i]:
            # st.session_state.all_selected = False
            print("triggered 2")
            return False
    # if st.session_state.option_selected_coordinate_arr == []:
    #     return True

    if len(st.session_state.option_selected_coordinate_arr) <=1:
        print("triggered 3")
        return False

    if st.session_state.prev_option_selected_coordinate_arr != st.session_state.option_selected_coordinate_arr:
        st.session_state.prev_option_selected_coordinate_arr = st.session_state.option_selected_coordinate_arr.copy()
        print("triggered 4")
        return True 
        
def check_duplicate_coords():
    '''
    
    '''
    print("Duplicate Check")
    for i in range(len(st.session_state["option_selected_coordinate_arr"])):
        for j in range(len(st.session_state["option_selected_coordinate_arr"])):
            print(st.session_state["option_selected_coordinate_arr"][i],  st.session_state["option_selected_coordinate_arr"][j])
            if i == j:
                continue
            if st.session_state["option_selected_coordinate_arr"][i] == st.session_state["option_selected_coordinate_arr"][j]:
                return True


def getsuggestions_v2(indx):
    key = st.secrets["MY_GEO_API_KEY"]
    location = st.session_state["entered_loc_{}".format(indx)]
    if location == "":
        return {}
    url = "https://api.geoapify.com/v1/geocode/autocomplete?text={}&lang=en&filter=circle:103.82293524258239,1.3008977700046245,25000&format=json&apiKey={}".format(location,key)
    response = requests.get(url)
    json_object = json.loads(response.text)
    value = {}

    for i in json_object["results"]:
        value[i["formatted"]] = [i["lon"],i["lat"]]
    return value

def trysuggestions_v2(indx):
    temp = getsuggestions_v2(indx)
    if len(st.session_state["locations_arr"])<=(indx+1):
        st.session_state["locations_arr"].append(temp)
    else:
        (st.session_state["locations_arr"])[indx] = temp
        

def Routing_locationIQ_v2():
    geometries_duration_pair = []
    st.session_state.various_routes = []
    #https://us1.locationiq.com/v1/reverse?key=pk.22aac5a23e1adbeebe8e84fe015ae488&lat=1.28036584335876&lon=103.830451146503&format=json

    #ACCEPTS LONGITUDE,  LATITUDE 
    key = st.secrets["MY_LOCATION_IQ_KEY"]
    # key = os.getenv("MY_LOCATION_IQ_KEY")
    # print(st.session_state.option_selected_coordinate_arr)
    coordinates = ";".join([f"{lat},{lon}" for lat, lon in st.session_state.option_selected_coordinate_arr])
    print(coordinates)
    url = "https://us1.locationiq.com/v1/directions/driving/{}?key={}&steps=true&alternatives=3&geometries=polyline&overview=full".format(coordinates, key)
    print(url)
    #   url = "https://us1.locationiq.com/v1/reverse?key={}&lat={}&lon={}&format=json".format(key,lat,long)

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)
    json_object = json.loads(response.text)
    # with open("RoutingLocationIQreturnvalue.json", "w") as outfile:
    #     json.dump(response.json(), outfile)
    for i in json_object["routes"]:
        geometries_duration_pair.append({
            "geometry":polyline.decode(i["geometry"]),
            "duration":i["duration"]
        })
    st.session_state.various_routes = geometries_duration_pair



def Routing_locationIQ(first_coords, second_coords):
    temp_arr = []
    #ACCEPTS LONGITUDE,  LATITUDE 

    key = st.secrets["MY_LOCATION_IQ_KEY"]
    url = "https://us1.locationiq.com/v1/directions/driving/{};{}?key={}&steps=true&alternatives=3&geometries=polyline&overview=full".format(first_coords, second_coords,key, )

    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    json_object = json.loads(response.text)
    with open("RoutingLocationIQreturnvalue.json", "w") as outfile:
        json.dump(response.json(), outfile)
    geometry = json_object["routes"][0]["geometry"]
    coordinates = polyline.decode(geometry)
    return coordinates

def haversine_formula(path):
    #This function makes use of the haversine_formula to find another parallel path.
    #This function enables us to create boundaries to find cameras within the path.
    #R is defined as the radius of earth
    #source: https://stackoverflow.com/questions/54873868/python-calculate-bearing-between-two-lat-long
    #This function returns pathing for left and right boundaries based on a given path as an argument
    #Argument accepts a 2D array [[lat1,long1], [lat2,long2]...]
    #Function returns a 2 variables, consisting of left boundary and right map boundary 

    
    parallel_path_1 =[]
    parallel_path_2 = []
    distance  = 0.035 #in km
    R = 6371
    for i in range(len(path)-1):
        lon1 = path[i][1]
        lat1 = path[i][0]
        lon2 = path[i+1][1]
        lat2 = path[i+1][0]
        
        
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        long_diff = lon2 - lon1
        x=  math.sin(long_diff) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(long_diff)
        bearing = math.atan2(x,y)


        #adding 90 degrees to the bearing achieved. 
        lat_offset = distance * math.cos(bearing + math.pi/2) / R
        lon_offset = distance * math.sin(bearing + math.pi/2) / (R * math.cos(lat1))


        
        left_first_coord = [math.degrees(i) for i in [lat1+lat_offset, lon1+lon_offset]]
        left_second_coord = [math.degrees(i) for i in [ lat2+lat_offset, lon2+lon_offset]]
        
        right_first_coord = [math.degrees(i) for i in [lat1-lat_offset, lon1-lon_offset]]
        right_second_coord = [math.degrees(i) for i in [lat2-lat_offset, lon2-lon_offset]]
        parallel_path_1.append(left_first_coord)
        parallel_path_1.append(left_second_coord)

        parallel_path_2.append(right_first_coord)
        parallel_path_2.append(right_second_coord)

    return parallel_path_1, parallel_path_2



def lat_lon_to_cartesian(lat, lon):
    # Convert latitude and longitude to radians
    lat, lon = np.radians(lat), np.radians(lon)
    R = 6371  # Earth's radius in kilometers
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])



async def point_between_parallel_lines(path1, path2):
    # Load camera data asynchronously
    async with aiofiles.open('AllCameraCoords.json', mode='r') as f:
        camera_data = json.loads(await f.read())  # async file read

    positive_camera = OrderedDict()

    async def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * arcsin(sqrt(a))
        return R * c

    async def project_point_to_line(p, a, b):
        ap = np.array(p) - np.array(a)
        ab = np.array(b) - np.array(a)
        denominator = np.dot(ab, ab)
        if denominator == 0:
            raise ValueError("Division by zero encountered in projection calculation. Point: {}, a:{}, b:{}".format(p, a, b))
        projection = np.array(a) + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return tuple(projection)

    async def is_point_on_line_segment(p, a, b):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    # Iterate through each line segment of path1 and path2
    for j in range(len(path1) - 1):
        for cameraID, point in camera_data.items():
            try:
                # Project points asynchronously
                proj1 = await project_point_to_line(point, path1[j], path1[j + 1])
                proj2 = await project_point_to_line(point, path2[j], path2[j + 1])
            except:
                continue

            # Check if projections are on the line segments asynchronously
            on_line1 = await is_point_on_line_segment(proj1, path1[j], path1[j + 1])
            on_line2 = await is_point_on_line_segment(proj2, path2[j], path2[j + 1])
            if not (on_line1 and on_line2):
                continue

            # Calculate distances
            d1 = await haversine_distance(*point, *proj1)  # Ensure point and proj1 are tuples or lists
            d2 = await haversine_distance(*point, *proj2)
            d_lines = await haversine_distance(*proj1, *proj2)

            # Check if point is between lines
            if abs(d1 + d2 - d_lines) < 1e-9:
                positive_camera[cameraID] = point

    return positive_camera

async def optimized_points_between_parallel_lines(path1, path2):
    '''
    This function takes in 2 parallel paths and checks if the point lies between the 2 parallel paths
    path 1 and path 2 are parallel to each other. 
    RD tree type
    '''
    async with aiofiles.open('AllCameraCoords.json', mode='r') as f:
        camera_data = json.loads(await f.read())  # async file read
    positive_camera = OrderedDict()
    
    async def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    async def project_point_to_line(p, a, b):
        ap = np.array(p) - np.array(a)
        ab = np.array(b) - np.array(a)
        denominator = np.dot(ab, ab)
        if denominator == 0:
            raise ValueError(f"Division by zero encountered in projection calculation. Point: {p}, a: {a}, b: {b}")
        projection = np.array(a) + np.dot(ap, ab) / denominator * ab
        return tuple(projection)

    async def is_point_on_line_segment(p, a, b):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and 
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    tree1 = create_spatial_index(path1)
    tree2 = create_spatial_index(path2)
    for cameraID, point in camera_data.items():
    # Find nearest line segments
        _, idx1 = tree1.query(point, k=3)
        _, idx2 = tree2.query(point, k=3)
        

        #lets say I  only need 1 tree.
        # Ensure idx1 and idx2 are 1D arrays
        idx1 = np.atleast_1d(idx1)
        idx2 = np.atleast_1d(idx2)
        
        concatenated_array = np.concatenate((idx1, idx2))
        flag = 0
        for i in concatenated_array:
            # segment1 = (tuple(path1[idx1[0]]), tuple(path1[idx1[0]+1]))
            # segment2 = (tuple(path2[idx1[0]]), tuple(path2[idx1[0]+1]))
            try:
                segment1 = (tuple(path1[i]), tuple(path1[i+1]))
                segment2 = (tuple(path2[i]), tuple(path2[i+1]))
            except:
                segment1 = (tuple(path1[i-1]), tuple(path1[i]))
                segment2 = (tuple(path2[i-1]), tuple(path2[i]))

            try:
                # Project point onto segments
                proj1 = await project_point_to_line(point, segment1[0], segment1[1])
                proj2 = await project_point_to_line(point, segment2[0], segment2[1])
            except:
                continue
            # Check if projections are on the line segments
            if (await is_point_on_line_segment(proj1, segment1[0], segment1[1]) and 
                    await is_point_on_line_segment(proj2, segment2[0], segment2[1])):
                flag= 1
                break
        if flag== 0:
            continue
        # Calculate distances
        d1 = await haversine_distance(point[0], point[1], proj1[0], proj1[1])
        d2 = await haversine_distance(point[0], point[1], proj2[0], proj2[1])
        d_lines = await haversine_distance(proj1[0], proj1[1], proj2[0], proj2[1])
        
        # Check if point is between lines
        if abs(d1 + d2 - d_lines) < 1e-9:
            positive_camera[cameraID] = point
    return positive_camera


def find_time_window(current_time):
    # Subtract 15 minutes from the current time
    adjusted_time = current_time - timedelta(minutes=15)
    
    # Extract hour and minute
    hour = adjusted_time.hour
    minute = adjusted_time.minute
    
    # Determine the start of the window
    if 0 <= minute < 15:
        window_start = datetime(adjusted_time.year, adjusted_time.month, adjusted_time.day, hour, 0)
    elif 15 <= minute < 30:
        window_start = datetime(adjusted_time.year, adjusted_time.month, adjusted_time.day, hour, 15)
    elif 30 <= minute < 45:
        window_start = datetime(adjusted_time.year, adjusted_time.month, adjusted_time.day, hour, 30)
    else:  # 45 <= minute < 60
        window_start = datetime(adjusted_time.year, adjusted_time.month, adjusted_time.day, hour, 45)
    
    # Calculate window end (30 minutes after start)
    window_end = window_start + timedelta(minutes=30)
    return window_start, window_end



def create_spatial_index(path):
    return cKDTree(np.array(path))


async def optimized_speed_points_in_between_lines(tree1, tree2, path1, path2, point):
    '''
    This function takes in 2 parallel paths and checks if the point lies between the 2 parallel paths
    path 1 and path 2 are parallel to each other. 
    RD tree type
    '''
    
    async def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    async def project_point_to_line(p, a, b):
        ap = np.array(p) - np.array(a)
        ab = np.array(b) - np.array(a)
        denominator = np.dot(ab, ab)
        if denominator == 0:
            raise ValueError(f"Division by zero encountered in projection calculation. Point: {p}, a: {a}, b: {b}")
        projection = np.array(a) + np.dot(ap, ab) / denominator * ab
        return tuple(projection)

    async def is_point_on_line_segment(p, a, b):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and 
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

    # Find nearest line segments
    _, idx1 = tree1.query(point, k=3)
    _, idx2 = tree2.query(point, k=3)
    

    #lets say I  only need 1 tree.
    # Ensure idx1 and idx2 are 1D arrays
    idx1 = np.atleast_1d(idx1)
    idx2 = np.atleast_1d(idx2)
    
    concatenated_array = np.concatenate((idx1, idx2))
    flag = 0
    for i in concatenated_array:
        # segment1 = (tuple(path1[idx1[0]]), tuple(path1[idx1[0]+1]))
        # segment2 = (tuple(path2[idx1[0]]), tuple(path2[idx1[0]+1]))
        segment1 = (tuple(path1[i]), tuple(path1[i+1]))
        segment2 = (tuple(path2[i]), tuple(path2[i+1]))
        # print("Segment1: ", segment1)
        # print("Segment2: ", segment2)
        try:
            # Project point onto segments
            proj1 = await project_point_to_line(point, segment1[0], segment1[1])
            proj2 = await project_point_to_line(point, segment2[0], segment2[1])

            
        except ValueError:
            # If we encounter a division by zero, skip this point
            return None

        # Check if projections are on the line segments
        if (await is_point_on_line_segment(proj1, segment1[0], segment1[1]) and 
                await is_point_on_line_segment(proj2, segment2[0], segment2[1])):
            flag= 1
            break
    if flag == 0 :
        return None
    # Calculate distances
    d1 = await haversine_distance(point[0], point[1], proj1[0], proj1[1])
    d2 = await haversine_distance(point[0], point[1], proj2[0], proj2[1])
    d_lines = await haversine_distance(proj1[0], proj1[1], proj2[0], proj2[1])
    
    # Check if point is between lines
    if abs(d1 + d2 - d_lines) < 1e-9:
        return point




async def get_speed_paths(i, parallel_1, parallel_2,session):
    key = st.secrets["MY_TOM_TOM_API_KEY"]
    baseURL = "api.tomtom.com"
    versionNumber = 4
    style = "reduced-sensitivity"  # Use this directly
    zoom = "11"
    format = "json"
    
    point = f"{i[0]},{i[1]}"
    url = f"https://{baseURL}/traffic/services/{versionNumber}/flowSegmentData/{style}/{zoom}/{format}?key={key}&point={point}"
    print(url)

    # async with aiohttp.ClientSession() as session:
    start_time = time.time()
    async with session.get(url) as response:
        tom_tom = await response.json()  # Async JSON loading
    print("--- %s tomtom I/O seconds ---" % (time.time() - start_time))
    coordinates = [
        [coord["latitude"], coord["longitude"]]
        for coord in tom_tom["flowSegmentData"]["coordinates"]["coordinate"]
    ]

    correct = []
    #async here
    start_time = time.time()
    tree1 = create_spatial_index(parallel_1)
    tree2 = create_spatial_index(parallel_2)
    # correct = await asyncio.gather(*(speed_points_in_between_lines(parallel_1, parallel_2, k) for k in coordinates))
    correct = await asyncio.gather(*(optimized_speed_points_in_between_lines(tree1, tree2, parallel_1, parallel_2, k) for k in coordinates))
    while None in correct:
        correct.remove(None)
    print("--- %s tomtom calculations seconds ---" % (time.time() - start_time))
    # for k in coordinates:
    #     # Await the asynchronous speed_points_in_between_lines function
    #     if await speed_points_in_between_lines(parallel_1, parallel_2, k):
    #         correct.append(k)
        

    speed = tom_tom['flowSegmentData']["currentSpeed"]
    return correct, speed



async def speed_labelling(route, parallel_1, parallel_2):
    n = len(route)
    num_breaks = 5
    segment_size = math.ceil(n / num_breaks)
    first_indices = [route[i * segment_size] for i in range(num_breaks)]
    async with aiohttp.ClientSession() as session:
        result = await asyncio.gather(*(get_speed_paths(i, parallel_1, parallel_2, session) for i in first_indices), return_exceptions=True)

    for i, item in enumerate(result):
        if isinstance(item, Exception):
            print(f"Error occurred for index {first_indices[i]}: {item}")
            result.pop(i)
    #Now need to transform array into zip
    #now result is [[correct, speed],[correct, speed]]
    print(result)
    correct_val, speed_val = zip(*result)
    result = zip(list(correct_val), list(speed_val))
    return result


def flatten(lst):
    """Flattens a list of lists into a single list."""
    return [item for sublist in lst for item in sublist if sublist]

def polygon_density_plotting(cameraID):
    mydb = client['clusterfyp']
    density_data = mydb['Clustering_Density']
    document= density_data.find_one({"_id": cameraID})
    document = document["cluster_properties"]
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(document)))

    # Collect all unique time points across all documents
    all_time_points = sorted(set(time for doc in document for time in doc.keys() if time != "polygon_coords"))

    # Create a mapping of time points to x-axis positions
    time_to_position = {time: i for i, time in enumerate(all_time_points)}

    for count, doc in enumerate(document):
        del doc["polygon_coords"]
        time_window_arr = []
        density_arr = []
        median_arr = []
        positions = []
        
        for time in sorted(doc.keys()):  # Sort the times to ensure correct order
            time_window_arr.append(time)
            doc[time] = [0 if sublist == [] else sublist for sublist in doc[time]]
            density_arr.append(doc[time])
            median_arr.append(np.median(doc[time]))
            positions.append(time_to_position[time])

        # Plot boxplot
        bp = ax.boxplot(density_arr,
                        positions=positions,
                        widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[count], alpha=0.6),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color=colors[count]),
                        capprops=dict(color=colors[count]),
                        flierprops=dict(color=colors[count], markeredgecolor=colors[count]))
        
        # Plot median line
        ax.plot(positions, median_arr, color=colors[count], marker='o', linestyle='-', linewidth=2, 
                label=f'Polygon {count+1} line')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(all_time_points)))
    ax.set_xticklabels(all_time_points, rotation=45, ha='right')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Density')
    ax.set_title('Density Distribution Across Time Categories for Various Polygon per Camera')

    # Add legend
    ax.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert BytesIO to base64 string
    image_string = base64.b64encode(buffer.getvalue()).decode()
    plt.clf()
    return image_string

def veh_count_plotting(cameraID):
    mydb = client['clusterfyp']
    density_data = mydb['Clustering_Density']
    document= density_data.find_one({"_id": cameraID})
    document = document["cluster_properties"]
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(document)))

    # Collect all unique time points across all documents
    all_time_points = sorted(set(time for doc in document for time in doc.keys() if time != "polygon_coords"))

    # Create a mapping of time points to x-axis positions
    time_to_position = {time: i for i, time in enumerate(all_time_points)}

    for count, doc in enumerate(document):
        del doc["polygon_coords"]
        time_window_arr = []
        density_arr = []
        median_arr = []
        positions = []
        
        for time in sorted(doc.keys()):  # Sort the times to ensure correct order
            time_window_arr.append(time)
            doc[time] = [0 if sublist == [] else sublist for sublist in doc[time]]
            density_arr.append(doc[time])
            median_arr.append(np.median(doc[time]))
            positions.append(time_to_position[time])

        # Plot boxplot
        bp = ax.boxplot(density_arr,
                        positions=positions,
                        widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[count], alpha=0.6),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color=colors[count]),
                        capprops=dict(color=colors[count]),
                        flierprops=dict(color=colors[count], markeredgecolor=colors[count]))
        
        # Plot median line
        ax.plot(positions, median_arr, color=colors[count], marker='o', linestyle='-', linewidth=2, 
                label=f'Polygon {count+1} line')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(all_time_points)))
    ax.set_xticklabels(all_time_points, rotation=45, ha='right')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Density')
    ax.set_title('Density Distribution Across Time Categories for Various Polygon per Camera')

    # Add legend
    ax.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)



    
# @st.cache_data(ttl=900)
async def route_poly_plotting_df(camera_ID_arr):
    #Accepts all the cameraID based on the chosen route. 
    #Query with cameraID. 
    #For each polygon in cameraID, calculate the average density.
    
    #Question: What if the database does not have that timing for that cameraID. (Solved)
    
    #How to know which histogram belongs to which colour on the histogram?
    #Plot uses colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters) - (-1 in unique_clusters)))
    #When colouring the polygons and colouring the barchart in the histogram, use the same colouring array. 
    #Seems like need to use VEGA plot
    # print("Arr:",camera_ID_arr)
    mydb = client['clusterfyp']
    density_data = mydb['Clustering_Density']
    documents = density_data.find({"_id": {"$in": camera_ID_arr}},)
    


    # Prepare to store the results
    results = {}
    # timing = st.session_state["timing_{}".format(st.session_state['selected_route'])].strftime("%H:%M")
    timing=st.session_state["timing"].strftime("%H:%M")
    print("timing: ",timing)
    # Process each document and compute the average cluster density for the specified time
    for doc in documents:
        camera_id = doc["_id"]
        for cluster in doc["cluster_properties"]:
            if timing in cluster:
                # Calculate the average of the values for the given time
                densities = cluster[timing]
                print("Densities:", densities)
                while [] in densities:
                    densities.remove([])
                if densities and densities!=[[]]:  # Check if densities list is not empty
                    average_density = statistics.mean(densities)
                else:
                    average_density = 0 # Handle empty list case
                if camera_id not in results.keys():
                    results[camera_id] = [average_density]
                else:
                    results[camera_id].append(average_density)
            else:
                results[camera_id] =[None]

    
    # Ensure all lists have the same length by padding with zeros
    # print("results:", results)
    highest_len = max((len(v) for v in results.values()), default=0)
    for key in results:
        if len(results[key]) != highest_len:
            results[key] = results[key] + (highest_len - len(results[key])) * [0]

    df_bar = pd.DataFrame.from_dict(results)
    df_bar= df_bar.T
    df_bar =df_bar.fillna(0)
    # print(df.index)

    temp_arr = [int(x) for x in camera_ID_arr]
    df_bar.index = df_bar.index.map(int)
    valid_rows = [int(col) for col in temp_arr if col in df_bar.index.tolist()]
    df_bar = df_bar.reindex(valid_rows)

    print("Checking:",df_bar)

    #Other than columns, 
    num_of_cols = df_bar.shape[1]
    num_of_rows = df_bar.shape[0]
    
    try:
        off_val = (num_of_rows*0.8)/(num_of_cols*num_of_rows)
    except:
        off_val= 1
    
    # off_val = 

    #when number of cols = even number, 
    if num_of_cols % 2 == 0:  # Even number of columns
        offsets = [(i - (num_of_cols / 2 - 0.5)) * off_val for i in range(num_of_cols)]
    else:  # Odd number of columns
        offsets = [(i - num_of_cols // 2) * off_val for i in range(num_of_cols)]
    

    print(offsets)
    x_label= list(range(len(df_bar.index)))
    x_labels_str = ["ID:{}".format(x) for x in df_bar.index]
    fig =  go.Figure()
    colors = plt.cm.rainbow(np.linspace(0, 1, df_bar.shape[1]))
    plotly_colors = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})' for c in colors]
    for i,col in enumerate(df_bar.columns):
        fig.add_trace(go.Bar(
        x=x_label,
        y=df_bar[col],
        name=f'Bar {col}',
        marker_color=  plotly_colors[i],
    ))


    current_info = mydb["Current Info"]
    current_documents = current_info.find({"_id": {"$in": camera_ID_arr}}, {"cluster_density": 1})
    current_results = {}
    for doc in current_documents:
        camera_id = doc["_id"]
        if doc["cluster_density"]:
            for cluster in doc["cluster_density"]:
                density = cluster["current_window_density"]
                if camera_id not in current_results.keys():
                    current_results[camera_id] = [density]
                else:
                    current_results[camera_id].append(density)
        else:
            current_results[camera_id] = [None]

    highest_len = max((len(v) for v in current_results.values()), default=0)
    for key in current_results:
        if len(current_results[key]) != highest_len:
            current_results[key] =current_results[key] + (highest_len - len(current_results[key])) * [0]
    
    cur_df = pd.DataFrame.from_dict(current_results)
    cur_df= cur_df.T
    cur_df =cur_df.fillna(0)
    cur_df.index = cur_df.index.map(int)
    cur_df = cur_df.reindex(valid_rows)

    for i, col in enumerate(cur_df.columns):
        if col in df_bar.columns:
            fig.add_trace(go.Scatter(
                x=[index + offsets[col] for index in x_label],
                y=cur_df[col],
                mode='markers',
                marker=dict(size=10, symbol='circle', color='red', line=dict(width=3,
                                        color='DarkSlateGrey')),
                name=f'Scatter {col}',
                marker_color=  plotly_colors[i]
            ))

    fig.update_layout(
    title="Current Density vs Average Density through Selected Journey",
    xaxis_title="Index",
    yaxis_title="Values",
    barmode='group',  # Group bars side by side,
    xaxis = dict(
        tickvals=x_label,  # Original numeric positions for bars
        ticktext= x_labels_str,  # Corresponding categorical labels
        title="Camera IDs"
        )
    )

    st.session_state.route_cluster_density_fig = fig


@st.cache_resource
def connect_to_db():
    # Encode the username and password
    password = st.secrets["database"]["MY_MONGO_DB_PASSWORD"]
    username = st.secrets["database"]["MY_MONGO_DB_USER_NAME"]
    cluster = st.secrets["database"]["MY_MONGO_CLUSTER"]

    uri = f"mongodb+srv://{username}:{password}@{cluster}.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=500000"
    client = MongoClient(uri, serverSelectionTimeoutMS=30000,
                     socketTimeoutMS=30000,
                     maxIdleTimeMS=60000,
                     connectTimeoutMS=20000)
    return client



@st.cache_data(ttl =900)
def get_camera_data(cameraID):
    mydb = client['clusterfyp']
    current_info = mydb["Current Info"]
    document = current_info.find_one({"_id": cameraID}, {"yolo_image":1, 
                                                         '_id':0, 
                                                         "location":1, 
                                                         "predicted_count":1, 
                                                         "cluster_img":1, 
                                                         "plot_img":1, 
                                                         "orig_image":1, 
                                                         "veh_plot":1})
    blob = document.get("yolo_image")
    location=document.get("location")
    vehicle_count =  document.get("predicted_count")
    cluster_img = document.get("cluster_img")
    plot_img = document.get("plot_img")
    orig_img = document.get("orig_image")
    veh_plot =  document.get("veh_plot")
    return blob, location, vehicle_count, cluster_img, plot_img, orig_img, veh_plot

client = connect_to_db()


html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
</head>
<body>
    <h1 style="font-size: 20px;"> {}</h1>
    <div style="background-color: white; padding: 2px; border-radius: 5px;">
        <img src="data:image/png;base64,{}" style="width:350px; height:auto;">
    </div>
</body>
<div>{}</div>
<div style="background-color: white; padding: 2px; border-radius: 5px;">
    <img src="data:image/png;base64,{}" style="width:450px; height:auto;">
</div>
<div style="background-color: white; padding: 2px; border-radius: 5px;">
    <img src="data:image/png;base64,{}" style="width:550px; height:auto;">
</div>
</html>
""".format


clust_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
</head>
<body>
    <h1 style="font-size: 20px;"> {}</h1>
    <div style="background-color: white; padding: 2px; border-radius: 5px;">
        <img src="data:image/png;base64,{}" style="width:450px; height:auto;">
    </div>
    <div style="background-color: white; padding: 2px; border-radius: 5px;">
        <img src="data:image/png;base64,{}" style="width:550px; height:auto;">
    </div>
</body>
</html>
""".format
