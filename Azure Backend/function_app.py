import logging
import azure.functions as func
import googlemaps
import json
from datetime import datetime
import requests
import json
import googlemaps
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import base64
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Polygon, box, Point
from shapely.validation import explain_validity
import numpy as np
from sklearn.cluster import DBSCAN,HDBSCAN
import matplotlib.pyplot as plt
from matplotlib.image import imread
from collections import Counter
import time
import threading
import queue
from pymongo import MongoClient
from datetime import datetime, timedelta
from azure.storage.fileshare import (
    ShareServiceClient,
    ShareClient,
    ShareDirectoryClient,
    ShareFileClient
)
import pytz
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()

app = func.FunctionApp()

@app.function_name(name = "first_trigger")
@app.timer_trigger(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 
def first_trigger(myTimer: func.TimerRequest) -> None:
    '''
    Main function call for Azure initialization.

    Function call properties stored in host.json

    Comment functions below are for respective function apps.  

    '''
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python first timer trigger function executed.')
    resp = get_current_directory()
    logging.warning(resp)
    # connect_to_azure_files()
    try:
        acquire_TrafficImages(1)
    except Exception as e:
        logging.critical(e, exc_info=True)



# @app.timer_trigger(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=False,
#               use_monitor=False) 
# def second_trigger(myTimer: func.TimerRequest) -> None:
#     if myTimer.past_due:
#         logging.info('The timer is past due!')

#     logging.info('Python 2nd timer trigger function executed.')
#     resp = get_current_directory()
#     logging.warning(resp)
#     # connect_to_azure_files()
#     try:
#         acquire_TrafficImages(2)
#     except Exception as e:
#         logging.critical(e, exc_info=True)

# @app.timer_trigger(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=False,
#               use_monitor=False) 
# def third_trigger(myTimer: func.TimerRequest) -> None:
#     if myTimer.past_due:
#         logging.info('The timer is past due!')

#     logging.info('Python 3rd timer trigger function executed.')
#     resp = get_current_directory()
#     logging.warning(resp)
#     # connect_to_azure_files()
#     try:
#         acquire_TrafficImages(3)
#     except Exception as e:
#         logging.critical(e, exc_info=True)

# @app.timer_trigger(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=False,
#               use_monitor=False) 
# def fourth_trigger(myTimer: func.TimerRequest) -> None:
#     if myTimer.past_due:
#         logging.info('The timer is past due!')

#     logging.info('Python 4th timer trigger function executed.')
#     resp = get_current_directory()
#     logging.warning(resp)
#     # connect_to_azure_files()
#     try:
#         acquire_TrafficImages(4)
#     except Exception as e:
#         logging.critical(e, exc_info=True)




def get_current_directory():
    '''
    Get current directory of Azure function. Folder is "/home/site/wwwroot/function_app.py"
    '''

    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # List all files and directories
    all_items = os.listdir(current_dir)
    
    # Separate files and directories
    files = []
    directories = []
    for item in all_items:
        full_path = os.path.join(current_dir, item)
        if os.path.isfile(full_path):
            files.append(item)
        elif os.path.isdir(full_path):
            directories.append(item)
    response = "Files:\n" + "\n".join(files) + "\n\nDirectories:\n" + "\n".join(directories)
    return (response)


def connect_to_db():
    # Encode the username and password
    password = os.getenv("MY_MONGO_DB_PASSWORD")
    username = os.getenv("MY_MONGO_DB_USER_NAME")
    cluster = os.getenv("MY_MONGO_CLUSTER")
    # Construct the URI with the encoded username and password
    uri = f"mongodb+srv://{username}:{password}@{cluster}.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=500000"

    client = MongoClient(uri, serverSelectionTimeoutMS=30000,
                     socketTimeoutMS=30000,
                     maxIdleTimeMS=60000,
                     connectTimeoutMS=20000)
    return client
    

def reduce_img_quality(image_blob):
    '''
    Reduce Image quality to save space in the database. 
    '''
    img = blob_to_image(image_blob)
    output_stream  = io.BytesIO()
    img.save(output_stream, format=img.format, optimize=True, quality=0.5)
    output_stream.seek(0)  # Go to the beginning of the stream
    output_blob = output_stream.read()
    return output_blob


def reduce_img_quality_64(input_data, quality = 50):
    '''
    Reduce Image quality to save space in the database. 
    '''

    #check if it is an image or base64 string
    #if string, then need to convert to image type and reduce quality then convert it back to base64 string and return base64 string
    #else just reduce quality then convert it back to base64 string and returnbase64 string
    def pil_to_base64(image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

    # Helper function to convert base64 string to PIL image
    def base64_to_pil(b64_string):
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data))
        return img

    # Check if input is base64 string (assuming it's base64 if it's a string type)
    if isinstance(input_data, str):
        # Convert base64 string to PIL image
        image = base64_to_pil(input_data)
        # Reduce quality and convert back to base64 string
        return pil_to_base64(image)
    else:
        # If it's not a string, assume it's already an image and reduce quality
        return pil_to_base64(input_data)


def find_time_window(current_time):
    '''
    Args: Current Time based on Singapore 
    
    Objective: Find the earliest timing for the sliding window of 30 mins 

    Returns:
        Window Start Time
        Window End Time

    Example: 
        Current Time: 22.50
        Sliding window start time: 22.30
        Ending Window Start time: 23.00

    '''

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

def clustering(eps_val, coords):
    '''
    Objective:
        To perform DBscan on the given coordinates. 

    Args: 
        eps_val: Minimum distance between the core point and the points to be considered to be in the cluster.
        coords: Coordinates taken in for clustering

    Return:
        Array of integers indicating respective cluster assignment to each coords in args. 
    '''
    coords = np.array(coords)
    hdb= DBSCAN(eps=eps_val, min_samples=3)
    clusters =hdb.fit_predict(coords)
    return clusters

def Reverse_GoogleMap(lat, long, gmaps):
    '''
    Objective:
        Use GoogleMaps API to perform Reverse Geoding. Longitude and Latitude are passed into the API to return location. 
    '''

    reverse_geocode_result = gmaps.reverse_geocode((lat, long))
    location = None
    flag = 0
    for i in reverse_geocode_result:
        if flag ==1:
            break
        for j in range(len(i['address_components'])):
            if "route" in i['address_components'][j]["types"]:
                location = i['address_components'][j]["long_name"] if "Exp" in i['address_components'][j]["long_name"] else None
            if location !=None:
                flag  = 1
                break
    
    # with open("GoogleMapsAPIreturnvaluetemp.json", "w") as outfile:
    #     json.dump(reverse_geocode_result, outfile)
    return location

def Reverse_locationIQ(lat, long):
    #https://us1.locationiq.com/v1/reverse?key=pk.22aac5a23e1adbeebe8e84fe015ae488&lat=1.28036584335876&lon=103.830451146503&format=json
    key = os.getenv("MY_LOCATION_IQ_KEY")
    url = "https://us1.locationiq.com/v1/reverse?key={}&lat={}&lon={}&format=json".format(key,lat,long)
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    json_object = json.loads(response.text)
    return json_object["address"]["road"]

def two_sum(i, n_clusters):
    i = i+n_clusters
    return i

def split_outliers(eps_value, car_coordinates):


    #split outliers and clustered
    #check outliers coordinates to compare if same content
    #need to link clusters array to saved coordinates
    #print(car_coordinates)
    clusters = clustering(eps_value, car_coordinates)
    outliers = []
    saved_coordinates = []
    saved_clusters = []
    i = 0
    while(i<len(clusters)):
        #Removing noise
        if clusters[i] ==-1:
            # clusters = np.delete(clusters, i, axis=0)
            outliers.append(car_coordinates[i])
        else:
            saved_clusters.append(clusters[i])
            saved_coordinates.append(car_coordinates[i])
        i+=1
    return saved_coordinates, np.array(saved_clusters), outliers

def jaccard_similarity_polygons(poly1, poly2):
    """
    Calculate the Jaccard similarity (Intersection over Union) between two polygons.
    
    :param poly1: First Shapely Polygon object
    :param poly2: Second Shapely Polygon object
    :return: Jaccard similarity as a float between 0 and 1
    """
    if not isinstance(poly1, Polygon) or not isinstance(poly2, Polygon):
        raise ValueError("Both inputs must be Shapely Polygon objects")

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    if union == 0:
        return 0  # Return 0 if both polygons are empty or don't intersect
    
    return intersection / union

def find_polygon_centroid(coordinates):
    """
    Find the centroid of a polygon using Shapely.
    
    Parameters:
    coordinates: List of tuples containing (x,y) coordinates of polygon vertices
                in clockwise or counterclockwise order
    
    Returns:
    tuple: (x, y) coordinates of the centroid
    """
    # Create a Shapely polygon from coordinates
    poly = Polygon(coordinates)
    
    # Get the centroid
    centroid = poly.centroid
    
    return (centroid.x, centroid.y)

def jaccard_overlap_check(poly1, poly2,cameraID):
    '''
    Params: 
        poly1: unique_cluster
        poly2: new cluster
        cameraID

    Return: Merged/Unmerged Polygon , True/False if merged
    
    Objective:
        1) Check if each polygon exists in the same constraint by using the centroid of the polygons. 
        2) If yes, then intersect the 2 polygons together. If the intersect exceeds 0.2,
        merge the 2 polygons
        
    '''

    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "sample2.json")
    f = open(file_path)
    data = json.load(f)
    constraint_polygons = data[cameraID]

    # best_poly1_val= -1
    # best_poly2_val= -1
    
    best_poly1_poly= []
    best_poly2_poly = []




    for i in constraint_polygons:
        #find with constraint the polygon belongs to. 
        temp_i = Polygon(i)
        if not Polygon(i).is_valid:
            print(f"constraint is invalid: {explain_validity(temp_i)}")
            temp_i= temp_i.buffer(0)
        
        if not poly1.is_valid:
            print(f"poly1 is invalid: {explain_validity(poly1)}")
            poly1 = poly1.buffer(0)
        
        centroid = poly1.centroid
        centroid = Point((centroid.x, centroid.y))
        if centroid.within(temp_i) or centroid.touches(temp_i):
            best_poly1_poly = i
            break

        
        # poly_1_val = jaccard_similarity_polygons(temp_i, poly1)

    for i in constraint_polygons:
        temp_i = Polygon(i)
        if not Polygon(i).is_valid:
            print(f"constraint is invalid: {explain_validity(temp_i)}")
            temp_i= temp_i.buffer(0)
        if not poly2.is_valid:
            print(f"poly2 is invalid: {explain_validity(poly2)}")
            poly2 = poly2.buffer(0)

        centroid = poly2.centroid
        centroid = Point((centroid.x, centroid.y))
        if centroid.within(temp_i) or centroid.touches(temp_i):
            best_poly2_poly = i
            break
        # poly_2_val = jaccard_similarity_polygons(temp_i, poly2)
        # if poly_1_val> best_poly1_val:
        #     best_poly1_val = poly_1_val
        #     best_poly1_poly = i
        # if poly_2_val> best_poly2_val:
        #     best_poly2_val = poly_2_val
        #     best_poly2_poly = i

    if best_poly1_poly != best_poly2_poly:
        #means they dont belong to the same constraint
        return poly1, False
        


    intersection = poly1.intersection(poly2).area
    if intersection == 0:
        return poly1, False
    if (poly1.area)/intersection >= 0.3 or (poly2.area)/intersection>=0.3:
        return poly1.union(poly2), True
    else:
        return poly1, False








def update_time_window(camera_id, polygon_coords, time, value, density_data):
    '''
    Params:
        cameraID
        polygon_coords: current unique polygon to update
        time: time window to update
        value: density 
        density_data: mongodb document "Clustering Density"

    Objective: 
        Update mongodb "Clustering_density" with the latest density according to given polygon
        and time window. 
    
    '''
    # Convert time to the format used in your database if necessary
    # For example, if you're storing time as a string in "HH:MM" format:
    time_str = time.strftime("%H:%M") if isinstance(time, datetime) else time

    # Prepare the update operation
    update_operation = {
        "$push": {
            "cluster_properties.$[elem].{}".format(time_str): {
                "$each": [value],
                "$position": 0  # Add to the beginning of the array
            }
        }
    }
    
    # Prepare the array filter to match the specific element
    array_filters = [{
        "elem.polygon_coords": polygon_coords
    }]

    # Execute the update
    result = density_data.update_one(
        {"_id": camera_id},
        update_operation,
        array_filters=array_filters
    )
    return result


def update_veh_count(pre_update, cameraID, start_time, client):
    '''
    Objective: 
        This function is to update the veh_count in the Counting Database
        1) Acquire cameraID from the database


    Args:
        pre_update: 
            Sliding window demographics collected when full

        cameraID:
            Used as ID to update the database
            Used to check against constraint polygon

        start_time:
            Time of the sliding window

        client:
            Database object.
        
    
    '''

    time_window = start_time.strftime("%H:%M")
    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "sample2.json")
    f = open(file_path)
    data = json.load(f)
    constraint_polygons = data[cameraID]
    mydb = client['clusterfyp']
    counting_data = mydb['Counting']


    
    check_if_exists = counting_data.find_one({"_id":cameraID})
    if check_if_exists == None:
        return None

    projection = {"polygon_coords": {
                    "$map": {
                        "input": "$cluster_properties",
                        "as": "cluster",
                        "in": "$$cluster.polygon_coords"
                        }
                    }
                }
    
    #Get all the polygon coords. 
    # logging.warning(cameraID)

    doc = counting_data.find_one({"_id": cameraID}, projection)['polygon_coords']

    for c_poly in constraint_polygons:
        total_poly_count = []
        try:
            temp = Polygon(c_poly)
        except:
            print(cameraID)
            continue
        if not Polygon(c_poly).is_valid:
            temp= temp.buffer(0)
        for image in pre_update["metadata"]:
            poly_array  = [Polygon(coords) for coords in image["cluster_shapes"]]
            for j in range(len(poly_array)):
                _, assignment = jaccard_overlap_check(temp, Polygon(poly_array[j]), cameraID)
                if assignment == True:
                    total_poly_count.append(image["confirmed_clusters"].count(j))
        if len(total_poly_count) == 0:
            continue
        else:
            avg_poly_count = sum(total_poly_count)/len(total_poly_count)
        if (c_poly not in doc) or doc == None:
            to_push = {"polygon_coords": c_poly,
                        time_window: [avg_poly_count]}
            counting_data.update_one({"_id":cameraID}, {"$push":{"cluster_properties":to_push}}, upsert = True)
        else:
            update_time_window(cameraID, c_poly, time_window, avg_poly_count, counting_data)


def update_clustering_density_mongodb(pre_update, cameraID, start_time, client):
    '''
    Params:
        pre_update = window array before it gets flushed
        cameraID 
        start_time = Beginning of new window
        client = mongodb object
    
    Return:
        Dictionary to update current info document

    Objective:
        1) From the pre-update, intersect all the polygons
        2) Get all unique polygon_coords for the cameraID from Clustering Density Document
        3) If doesnt exist, assign the current polygons from preupdates as unique
        4) Else, attempt to merge intersecting polygon. If can merge, then remove polygon
        from pre-update and update the unique polygons
        5) For each cluster from the window from pre-update, find the closest polygon to unqiue polygon in 
        terms of jaccard coefficient and calculate cluster density
        6) Update document of updated polygons and new polygons
        7) Update Cluster Density
        
    '''

    current_cluster_to_density_pair_value= []
    time_window = start_time.strftime("%H:%M")
    mydb = client['clusterfyp']
    
    temp_intersections = [Polygon(coords) for coords in pre_update["metadata"][0]["cluster_shapes"]]
    for i in range(len(pre_update["metadata"])-1):
        temp_intersections = find_intersections(temp_intersections,[Polygon(coords) for coords in pre_update["metadata"][i+1]["cluster_shapes"]])


    density_data = mydb['Clustering_Density']
    projection = {"polygon_coords": {
                    "$map": {
                        "input": "$cluster_properties",
                        "as": "cluster",
                        "in": "$$cluster.polygon_coords"
                        }
                    }
                }
    density_result = density_data.find_one({"_id": cameraID}, projection)

    #Need to key here if temp_intersection == null and density_result == None, then update cluster density == []
    #if temp_intersection == null but density_result !=None then set all polygon:cluster_density= 0
    


    if density_result is None:
        #means that there is no cameraID in density data collection.  
        #Make use of current cluster to become unique cluster
        
        #Iterate through 'unique' intersected clusters, then iterate through each polygon. 
        #If there exists an intersect then add to clust_density arr,
        #at the end of each unique cluster, find the average density and push to the current time frame. 
        #Doesnt update mongodb when there is no intersection among the 30 minutes window. 
        # print("No Camera ID")
        for unique_intersection in temp_intersections:
            clust_density_arr= []
            for image in pre_update["metadata"]:
                poly_array  = [Polygon(coords) for coords in image["cluster_shapes"]]
                for j in range(len(poly_array)):
                    inter = poly_array[j].intersection(unique_intersection)

                    if not inter.is_empty:
                        #calculate cluster density
                        cluster_points = image["confirmed_clusters"].count(j)
                        area = poly_array[j].area
                        clust_density_arr.append(cluster_points/area)
            
            average_density = sum(clust_density_arr)/len(clust_density_arr)
            unique_intersection = polygon_to_json(unique_intersection)
            to_push = {"polygon_coords": unique_intersection,
                        time_window: [average_density]}

            current_cluster_to_density_pair_value.append({"polygon":unique_intersection,
                                                        "current_window_density": average_density})
            density_data.update_one({"_id":cameraID}, {"$push":{"cluster_properties":to_push}}, upsert = True)

    else:
        #When intersection dont exists but there's already unique clusters 
        #Get all unique clusters and assign them to 0
        if len(temp_intersections) == 0:
            for unique_cluster in density_result['polygon_coords']:
                current_cluster_to_density_pair_value.append({"polygon":unique_cluster,
                                                        "current_window_density": 0})
            return current_cluster_to_density_pair_value
    
        #intersection obtained compare with each unique cluster
        #This section is also used to update unique cluster if there's any new. 

        #for each new intersection, find the closest unique cluster in terms of jaccard coefficient. 
        #basically to check out if there is any new unique clusters. 

        #Question: How to add new clusters?
        #How to check if the new cluster is not involved in the growth?
        new_polygon_arr = []
        for unique_cluster in density_result['polygon_coords']:
            new_polygon = Polygon(unique_cluster)
            for recent_intersection in temp_intersections:
                new_polygon,check_merge = jaccard_overlap_check(new_polygon, recent_intersection, cameraID)
                if check_merge== True:
                    #Now temp_intersection only contains clusters that does not belong to unique clusters. 
                    temp_intersections.remove(recent_intersection)
            new_polygon_arr.append(new_polygon)
        
        #unique_growth_cluster_pair to update geometry of unique clusters later on 
        unique_growth_cluster_pair = zip(density_result['polygon_coords'], new_polygon_arr)
        
        #Combine all new polygons and unjoined clusters together
        all_unique_poly = new_polygon_arr+temp_intersections
        #Now iterate through metadata to find highest similarity of unique cluster
        all_unique_poly = [polygon_to_json(poly) for poly in all_unique_poly]
        all_unique_poly_dict = {}
        for i in all_unique_poly:
            all_unique_poly_dict[tuple(i)]= []
        
        # for unique_poly in all_unique_poly:
        
        #Get all the highly similar unique clusters for each. 
        #Question: What happens when cluster_shape is empty
        for image in pre_update["metadata"]:
            poly_array  = [Polygon(coords) for coords in image["cluster_shapes"]] 
            if len(poly_array)==0:
                continue
            best_jaccard_polygon_arr= []
            #This for loop allows me to find best fitting polygon in terms of jaccard coefficient.
            #So for each poly array 
            cluster_density_arr = []
            for i in range(len(poly_array)):
                raw_poly = poly_array[i]
                #Calculate density for each raw polygon
                cluster_points = image["confirmed_clusters"].count(i)
                cluster_density_arr.append((cluster_points/raw_poly.area)**3)
                highest_jaccard_value = -1
                best_jaccard_polygon=[[]]
                #This for loop finds best fitting polygon in terms of jaccard coefficient
                for unique_poly in all_unique_poly:
                    unique_poly = Polygon(unique_poly)
                    jaccard_value = jaccard_similarity_polygons(raw_poly, unique_poly)
                    if jaccard_value>highest_jaccard_value:
                        highest_jaccard_value = jaccard_value
                        best_jaccard_polygon = unique_poly
                best_jaccard_polygon_arr.append(best_jaccard_polygon)

            
            #After each image in time window,    
            best_jaccard_polygon_arr = [polygon_to_json(i) for i in best_jaccard_polygon_arr]
            best_raw_unique_density_pair= zip(image["cluster_shapes"],best_jaccard_polygon_arr, cluster_density_arr)
            for i in tuple(best_raw_unique_density_pair):
                all_unique_poly_dict[tuple(i[1])].append(i[2])

        #Now with this dictionary, I want to calculate the average density for each unique clusters.
        #If cluster_density = [], then remove the key
        for i in list(all_unique_poly_dict.keys()):
            if len(all_unique_poly_dict[i]) == 0:
                all_unique_poly_dict.pop(i)
                continue
            all_unique_poly_dict[i]= sum(all_unique_poly_dict[i])/len(all_unique_poly_dict[i])
            
        #So first I make changes to those old polygon
        #Iterate through unique_growth_cluster_pair
        for i in tuple(unique_growth_cluster_pair):
            old_polygon_coords = i[0]
            new_polygon_coords = polygon_to_json(i[1])
            result = density_data.update_one(
                {
                    '_id': cameraID,
                    'cluster_properties': {
                        '$elemMatch': {
                            'polygon_coords': old_polygon_coords
                        }
                    }
                },
                {
                    '$set': {
                        'cluster_properties.$.polygon_coords': new_polygon_coords
                    }
                }
            )
        
        #Now I add new clusters to the mongodb. 
        #Now my problem is that all_unique_poly_dict may return key error. No, confirm will be inside. 
        for l in temp_intersections:
            to_push = {"polygon_coords": polygon_to_json(l)}
            density_data.update_one({"_id":cameraID}, {"$push":{"cluster_properties":to_push}}, upsert = True)
            
        
        #Now that for each cameraID, it contains all the unique polygons. 
        #Now update the mongodb with all_unique_poly_dict cluster values. 
        #Now I need to find cameraID, search the array for the matching polygon, and append the cluster to a timing window. 
        for poly in all_unique_poly_dict.keys():
            #Accepts these as parameters:
            #camera_id, polygon_coords, time, value, density_data
            update_time_window(cameraID, list(poly), time_window, all_unique_poly_dict[poly], density_data)

        mongo_db_dict = []
        
        for i in all_unique_poly_dict.keys():
            mongo_db_dict.append({"polygon":list(i),
                                "current_window_density": all_unique_poly_dict[i]
                                })
        return mongo_db_dict

from PIL import Image
import io

def blob_to_image(blob_data):
    image = Image.open(io.BytesIO(blob_data))
    return image

def image_to_blob(image):
    blob_buffer = BytesIO()
    
    # Save the image in PNG format to the buffer
    image.save(blob_buffer, format="PNG")
    
    # Get the blob data
    blob_data = blob_buffer.getvalue()
    
    return blob_data


def point_in_polygon(car_coords, polygon):
    x = car_coords[0]
    y= car_coords[1]
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside




def coords_restrain_assignment(car_coords, cameraID):
    '''
    Objective:
        takes the car_coords to assign to various polygon constraints such that clustering does
        not overlap across different roads.
        
        Break each clusters into different polygon segments
    
    Params:
        car_coords: car_coordinates retrieved from YOLO
        cameraID: cameraID of image

    Return:
        2D-array where each array consists of coordinates per cluster
    '''

    #This function takes in car coords, reads json file and break each clusters into different polygon segements
    #Read json file here is good as I am transferring my data to mongoDB if not everytime need to incur huge IO when
    #you can just acquire the respective polygons.
    # poly_json = "sample2.json"
    # f = open(f"/tmp/{poly_json}")
    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "sample2.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    polygon_sorting_array = []
    constraint_polygons = data[cameraID]
    for i in range(len(constraint_polygons)):
        in_this_polygon = []
        for j in car_coords:
            if point_in_polygon(j, constraint_polygons[i]):
                in_this_polygon.append(j)
        polygon_sorting_array.append(in_this_polygon)
    # print(polygon_sorting_array)
    return polygon_sorting_array
                
        
def get_camera_location(cameraID):
    '''
    Objective:
        GoogleAPI costs money so camera location names are loaded into a json file.
        json file: ID_location.json
            contains cameraID:location name key value pairs

    Params:
        CameraID

    Return:
        Location of the Camera
    '''

    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "ID_Location.json")
    with open(file_path, 'r') as file:
        data = json.load(file)
    try:
        return data[cameraID]
    except:
        return None


def acquire_TrafficImages(position):
    '''
    Params: 
        Position: Identify which half of the LTA data run inference on

    Objective:
        Acts as a Main function
        1) Connects to various APIs to collect data
        2) Run YOLO inference to get car coordinates
        2) Peform Iterative DBscan on each image and save into noSQL 
        document, Post Clustering Metadata
        3) Check and flush window if exceed time window and perform 
        update_clustering_density_mongodb function
        4) Update Current Info Document
        

    Documents:
        1) Post Clustering Metadata
        A sliding window that stores up to 30 minutes of data before calculating
        average density 
        
        2) Current Info
        Stores different images, consisting of Original, YOLO, and clustered image

        3) Clustering Density
        Stores Cluster shapes and Density for each time window

    '''
    training_file = "best.pt"
    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, f"training_weights/{training_file}")


    model  = YOLO(file_path, task= "segment")


    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "sample2.json")
    #Need to refactor this function. not Single responsible function:
    # image_properties=[]
    client = connect_to_db()
    mydb = client['clusterfyp']
    table = mydb["Post Clustering Metadata"]
    current_info = mydb["Current Info"]
    singapore_timezone = pytz.timezone('Asia/Singapore')
    current_time = datetime.now(singapore_timezone)
    start_time, end_time = find_time_window(current_time)

    google_key = os.getenv("MY_GOOGLE_API_KEY")
    gmaps = googlemaps.Client(key = google_key)


    lta_key= os.getenv("MY_LTA_DATAMALL_KEY")
    url = "http://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    payload = {}
    headers = {
    'AccountKey': lta_key,
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    json_object = json.loads(response.text)

    n = len(json_object["value"])
    split_1 = n // 4
    split_2 = n // 2
    split_3 = 3 * n // 4
    
    # Split the array using slicing
    if position == 1:
        json_object["value"] = json_object["value"][:split_1]
    elif position == 2:
        json_object["value"] = json_object["value"][split_1:split_2]
    elif position == 3:
        json_object["value"] = json_object["value"][split_2:split_3]
    else:
        json_object["value"] = json_object["value"][split_3:]

    

    for i in json_object["value"]:
        url = i["ImageLink"]
        i['ImageString']=download_image_to_base64(url)
        # i['ImageString'] = download_image_to_blob(url)
        
    for i in json_object["value"]:
        cameraID = i["CameraID"]
        latitude = i["Latitude"]
        longitude = i["Longitude"]
        downloadLink = i["ImageLink"]
        original_img_str = i["ImageString"]
        #Using the latitude and longitude of camera, acquire the road name using either googleAPI or geoAPI
        # location  = Reverse_GoogleMap(latitude, longitude, gmaps)
        location = get_camera_location(cameraID)
        if location == None:
            geo_api_key = os.getenv("MY_GEO_API_KEY")
            geoapi = "https://api.geoapify.com/v1/geocode/reverse?lat={}&lon={}&format=json&apiKey={}".format(latitude, longitude, geo_api_key)
            response = requests.get(geoapi)
            geoapidict = response.json()
            try:
                location = geoapidict["results"][0]['street']
            except:
                location = Reverse_locationIQ(latitude,longitude)


        timeofPhoto = downloadLink.split("/")[3] +", "+ downloadLink.split("/")[4]
        timeofPhoto_object = datetime.strptime(timeofPhoto,  "%Y-%m-%d, %H-%M")
        cv_image_str, car_coords, veh_count = run_yolo_inference(original_img_str, model)
        #Since I have the car_coords here, I want to separate them into different constraints first.
        #Some of the car coords may not be included in the polygon.
        constraint_assignment = coords_restrain_assignment(car_coords, cameraID)
        length_of_constraint_assigned_coords = 0
        for i in constraint_assignment:
            length_of_constraint_assigned_coords+=len(i)

        if len(car_coords)!=0:
            confirmed_coordinates= []
            confirmed_clusters = np.array([])
            n_clusters= 0
            outliers = car_coords.copy()
            all_outliers = []
            
            for segment in constraint_assignment:
                eps_value = 0.15
                #Original step size value = 0.05
                step_size = 0.1
                if len(segment) == 0:
                    continue
                # print("segment: ", segment)
                outliers = segment.copy()
                count  =0
                while(count<4):
                    temp = outliers.copy()
                    clustered_coordinates, clusters, outliers = split_outliers(eps_value,outliers)
                    confirmed_coordinates.extend(clustered_coordinates)
                    clusters = np.array([two_sum(i,  n_clusters) for i in clusters])
                    confirmed_clusters = np.concatenate([confirmed_clusters, clusters])
                    n_clusters = max(set(confirmed_clusters))+1 if len(set(confirmed_clusters)) else 0
                    eps_value = eps_value+step_size
                    #if outliers equates to the original input, means the current eps for dbscan is not sufficient
                    #and requires another iteration
                    if len(outliers) == len(temp):
                        count+=1
                    if len(outliers)==0:
                        #means there is not more outliers, then clustering is not required anymore
                        break
                if len(outliers)!=0:
                    all_outliers.append(outliers)
            confirmed_clusters = np.concatenate([confirmed_clusters,np.array((length_of_constraint_assigned_coords-len(confirmed_clusters))*[-1])])
            for i in all_outliers:
                confirmed_coordinates = confirmed_coordinates+i
            confirmed_coordinates = np.array(confirmed_coordinates)
            clust_shape = get_shape_clusters(confirmed_coordinates, confirmed_clusters, cameraID)
            
            #Need to get plotted cluster image as well

            #Need to convert ndarray to json seriablizable. 
            to_save = {"camera_location": [latitude, longitude],
                        "confirmed_coords": confirmed_coordinates.tolist(),
                        "confirmed_clusters":confirmed_clusters.tolist(),
                        "cluster_shapes": clust_shape,
                        "timeofPhoto": timeofPhoto_object,
                        }



            table.update_one({"_id":cameraID}, {"$push":{"metadata":to_save}}, upsert = True)
            change = table.find_one_and_update(
                {"_id": cameraID},
                {
                    '$pull': {
                        'metadata': {
                            'timeofPhoto': {'$lt': start_time}
                        }
                    }
                }
            )
            post_update= table.find_one({"_id": cameraID})

        else:
            #If currently no cluster shape
            to_save = {"camera_location": [latitude, longitude],
                        # "orig_image": original_img_str,
                        "timeofPhoto": timeofPhoto_object,
                        'cluster_shapes':[]
                        }
            
            table.update_one({"_id":cameraID}, {"$push":{"metadata":to_save}}, upsert = True)
            
            change = table.find_one_and_update(
                {"_id": cameraID},
                {
                    '$pull': {
                        'metadata': {
                            'timeofPhoto': {'$lt': start_time}
                        }
                    }
                }
            )
            post_update= table.find_one({"_id": cameraID})
            
            
        if change != post_update:
            #means there exists change to the window, thus need to calculate cluster density. 
            original_img_str = reduce_img_quality_64(original_img_str)
            # print("Updating Clustering Density", change)
            density_data = mydb['Clustering_Density']
            check_cluster_integrity(cameraID, density_data)
            current_cluster_density = update_clustering_density_mongodb(change, cameraID, start_time, client)
            update_veh_count(change,cameraID, start_time, client)
            projection = {"polygon_coords": {
                            "$map": {
                                "input": "$cluster_properties",
                                "as": "cluster",
                                "in": "$$cluster.polygon_coords"
                                }
                            }
                        }
            unique_poly = density_data.find_one({"_id": cameraID}, projection)
            clust_img= plot_individual_graph(original_img_str, unique_poly)
            #new
            centroid_img_str = plot_polygon_centroid(original_img_str, cameraID)
            #new
            veh_plot = veh_count_plotting(cameraID, client)
            plot_str = polygon_density_plotting(cameraID, density_data)
            to_concurrent = {
                            "location":location,
                            "orig_image": centroid_img_str,
                            "yolo_image": cv_image_str,
                            "cluster_img":clust_img,
                            "predicted_count": veh_count,
                            "cluster_density": current_cluster_density,
                            "plot_img": plot_str,
                            "veh_plot": veh_plot, 
                            "last_update":start_time
                            }
        else: 
            density_data = mydb['Clustering_Density']
            check_cluster_integrity(cameraID, density_data)
            projection = {"polygon_coords": {
                            "$map": {
                                "input": "$cluster_properties",
                                "as": "cluster",
                                "in": "$$cluster.polygon_coords"
                                }
                            }
                        }
            unique_poly = density_data.find_one({"_id": cameraID}, projection)
            clust_img= plot_individual_graph(original_img_str, unique_poly)
            #new
            centroid_img_str = plot_polygon_centroid(original_img_str, cameraID)
            #new
            veh_plot = veh_count_plotting(cameraID, client)
            plot_str = polygon_density_plotting(cameraID, density_data)
            to_concurrent = {
                            "location":location, 
                            "orig_image": centroid_img_str,
                            "yolo_image": cv_image_str,
                            "cluster_img":clust_img,
                            "predicted_count": veh_count,
                            "plot_img": plot_str,
                            "veh_plot": veh_plot,
                            "last_update":start_time
                            }

        current_info.update_one({"_id": cameraID}, {"$set": to_concurrent}, upsert = True)


                
def download_image_to_base64(url):
    '''
    Objective: 
        Download Traffic Image URL from LTADataMall S3 Bucket URL into base64 string

    Args:
        URL: S3 bucket URL string from LTA datamall API

    Return:
        base64string

    '''

    # Send a GET request to the URL to fetch the image
    response = requests.get(url).content
    image_buffer = BytesIO(response)


    with Image.open(image_buffer) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        original_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return original_img_str



def run_yolo_inference(orig_img_str, model):
    '''
    Objective:
        Run yolo_inference using trained model to acquire 3 things:
            1) Image with bounding boxes as "img_str"
            2) Coordinates of inferenced car coordinates
            3) Vehicle Counts for each specific type in dictionary form

    Args:
        orig_img_str: image string pulled from LTA datamall
        model: trained YOLOv8 Model

    Return:
        img_str:
            Image with bounding boxes as "img_str" as base64
        car_coordinates:
            Coordinates of inferenced car coordinates
        veh_count:
            Vehicle Counts for each specific type in dictionary form
    '''

    class_names = ["bike","bus", "car", "lorry", "truck"]
    img_data = base64.b64decode(orig_img_str)
    img = Image.open(BytesIO(img_data))

    results = model(source = img, imgsz=640,conf=0.45, show_conf=False, line_width = 1)  # list of Results objects
    veh_count= dict(Counter([class_names[int(cls)] for cls in results[0].boxes.cls]))
    #Now to convert it to count dictionary.
    car_coordinates = []
    for result in results:
        for box in result.boxes:
            x,y,w,h = box.xywhn.tolist()[0]
            car_coordinates.append([x,1-y])
        im_array = result.plot()
        image =  Image.fromarray(im_array[...,::-1])
    
    img_str = reduce_img_quality_64(image)
    return img_str, car_coordinates, veh_count

def polygon_to_json(polygon):
    if isinstance(polygon, Polygon):
        exterior_coords = list(polygon.exterior.coords)
        return exterior_coords
    else:
        raise TypeError("Object is not a Polygon")

def json_to_polygon(json_data):
    if json_data["type"] == "Polygon":
        return Polygon(json_data["coordinates"][0])
    else:
        raise ValueError("JSON does not represent a Polygon")


def get_shape_clusters(coordinates, clusters, cameraID):
    '''
    Params:
        coordinates:
            Coordinates receive car_coordinates per image as np.array
        clusters:
            np.array of cluster identities such as 0,1,2,3,4
        cameraID:
            cameraID

    Objective:
        With the coordinates and clustering assignment array of these clusters,
        create a hull around these clusters respectively:
        If the cluster < 5 points: Use rectangular bounding boxes
        Else:
            Use Normal Polygon Hull

    Return:
        Returns polygon convexhull for each of clusters in an image. 
    '''
    
    shapes_per_image = []
    def alpha_shape(points, alpha):
        #Passes in the coordinates of each cluster within the image. 
        #Return the convex hull of the cluster. 
        """
        Compute the alpha shape (concave hull) of a set of points.

        """
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            #Iterates through the indices of points forming the simplices.
            for i in range(3):
                #I becomes the current coordinates. 
                #J becomes the next coordinate. When I becomes 3, J becomes 1
                i, j = simplex[i], simplex[(i+1)%3]
                if i > j:
                    i, j = j, i
                edges.add((i, j))
        
        boundary_edges = set()
        for i, j in edges:
            if (j, i) not in edges:
                boundary_edges.add((i, j))
        
        boundary_points = set()
        for i, j in boundary_edges:
            boundary_points.add(i)
            boundary_points.add(j)
        return Polygon([points[i] for i in boundary_points]).convex_hull
    
    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "sample2.json")
    f = open(file_path)
    data = json.load(f)
    constraint_polygons = data[cameraID]
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        #iterates through each unique clusters to get the hull of the cluster.
        if cluster != -1:  # Skip noise points
            # Get points for this cluster
            cluster_points = coordinates[clusters == cluster]
            first_coord = cluster_points[0]
            for i in constraint_polygons:
                if point_in_polygon(first_coord, i):
                    relevant_polygon  = Polygon(i)
                    if not relevant_polygon.is_valid:
                        relevant_polygon= relevant_polygon.buffer(0)
                    break
            if len(cluster_points) <=5:
                #Find min and max for x and y coordinates
                min_x, min_y = np.min(cluster_points, axis=0)
                max_x, max_y = np.max(cluster_points, axis=0)
                cluster_shape = box(min_x, min_y, max_x, max_y)
                cluster_shape = cluster_shape.intersection(relevant_polygon)


            else:
            # Compute the concave hull
                cluster_shape = alpha_shape(cluster_points, alpha=0.1)


            #Create intersect here

            #convert the clustershape to json serializable
            #So basically I have to iterate through all the  polygons and see which inntersect which and then calculate the respective density. 
            cluster_shape = polygon_to_json(cluster_shape)
            shapes_per_image.append(cluster_shape)
    return shapes_per_image

def plot_individual_graph(img_string, shapes):
    '''
    Param:
        img_string: 
            Encode64 string of the original image from LTA datamall
        shapes:
            array of various polygon_coordinates
    
    Objective:
        To Plot hull onto image for visualization purpose

    Return:
        base64 cluster_img_str

    '''
    if shapes !=None:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(shapes['polygon_coords'])))
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, 100))


    img_data = base64.b64decode(img_string)
    img = Image.open(BytesIO(img_data))
    new_size = (img.width // 2, img.height // 2)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)  # Use Image.LANCZOS for high-quality downscaling
    img_array = np.array(img_resized)
    fig,ax = plt.subplots()
    ax.imshow(img_array, extent=[0,1,0,1])

    
    # Define colors for outlines (excluding noise if -1 is present)
    
    color_index = 0
    if shapes !=None:
        for j in shapes['polygon_coords']:
            #convert it back into convex hull
            # print(j)
            j = Polygon(j)
            # curr_shape = polygon_to_json(j)
            x,y = j.exterior.xy
            ax.plot(x,y,c=colors[color_index],linewidth = 1)
            color_index+=1
    ax.set_title('Clustering (DBscan)')
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    ax.axis('off')
    cluster_img_str = pyplot_to_image_string(fig)
    plt.close()
    return cluster_img_str


def plot_polygon_centroid(img_string, cameraID):
    '''
    Param:
        img_string: 
            Encode64 string of the original image from LTA datamall
        CameraID:
            string: cameraID of the image 

    Objective:
        To Plot centroid of each constraint polygon with image for vizualization purpose

    Return:
        base64 centroid_img_str

    '''
    img_data = base64.b64decode(img_string)
    img = Image.open(BytesIO(img_data))
    img_array = np.array(img)
    function_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(function_directory, "sample2.json")
    f = open(file_path)
    data = json.load(f)
    fig,ax = plt.subplots()
    ax.imshow(img_array, extent=[0,1,0,1])

    polygons = data[cameraID]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(polygons)))
    count =0 
    for i in polygons:
        try:
            poly = Polygon(i)
        except:
            continue
        if not poly.is_valid:
            poly= poly.buffer(0)
        
        # Get the centroid
        centroid = poly.centroid   
        # Plot the centroid
        ax.plot(centroid.x, centroid.y, color=colors[count], marker='o', markersize=20)
        count+=1
    ax.axis('off')
    centroid_img_str = pyplot_to_image_string(fig)
    plt.close()
    return centroid_img_str


def pyplot_to_image_string(fig, format='png'):
    """
    Convert a pyplot figure to a base64 encoded image string.
    
    :param fig: matplotlib.figure.Figure object
    :param format: str, the desired image format (default is 'png')
    :return: str, base64 encoded image string
    """
    # Save the figure to a bytes buffer
    buf = BytesIO()
    fig.savefig(buf, format=format)
    buf.seek(0)
    
    # Encode the bytes as base64
    img_str = base64.b64encode(buf.getvalue()).decode()
    
    return img_str

def find_intersections(shapes1, shapes2):
    '''
    Find intersection between 2 polygons
    
    '''

    #shape1 accepts the first intersection or the computed intersection
    #shape2 accept the second intersection or the last intersection. 
    shapes_intersected = []
    intersections = []
    for shape1 in shapes1:
        for shape2 in shapes2:
            intersection = shape1.intersection(shape2)
            if not intersection.is_empty:
                shapes_intersected.append([shape1,shape2])
                intersections.append(intersection)

    return intersections

def polygon_density_plotting(cameraID, density_data):
    '''
    Objective:
        For each cameraID, plot a boxplot graph for each of the polygon density

    Params:
        cameraID:
            Used for querying database
        Density data:
            database document object

    Return:
        Pyplot image string in base64
    
    '''
    document= density_data.find_one({"_id": cameraID})
    if document == None:
        return ''
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

def veh_count_plotting(cameraID, client):
    '''
    Objective:
        For each cameraID, plot a line graph for each of the road's vehicle count.

    Params:
        cameraID:
            Used for querying database
        client:
            database document object

    Return:
        Pyplot image string in base64
    
    '''
    mydb = client['clusterfyp']
    density_data = mydb['Counting']
    document= density_data.find_one({"_id": cameraID})
    if document ==None:
        return
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

        ax.plot(positions, median_arr, color=colors[count], marker='o', linestyle='-', linewidth=2, 
                label=f'Polygon {count+1} line')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(all_time_points)))
    ax.set_xticklabels(all_time_points, rotation=45, ha='right')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Median')
    ax.set_title('Median Count of Vehicles across Time')

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




def check_cluster_integrity(cameraID, density_data):
    '''
    Objective: 
        After creating some new polygons and stored in the database, this function aims
        to merge intersectinng individual polygons which are already 
        existing in the database but are not merged.

    Params:
        cameraID:
            Used to query database
        Density_data:
            database object
    
    '''

    # density_data = mydb["Clustering_Density"]
    document= density_data.find_one({"_id": cameraID}, {'cluster_properties': 1, "_id":0})
    if document == None:
        return
    poly_arrays = document["cluster_properties"]
    for i in poly_arrays:
        for j in poly_arrays:
            if i == j:
                continue
            poly, merge = jaccard_overlap_check(Polygon(i['polygon_coords']),Polygon(j['polygon_coords']), cameraID)
            
            if merge == True:
                for jkeys in j.keys():
                    if jkeys =="polygon_coords":
                        continue
                    if jkeys in i.keys():
                        i[jkeys] += j[jkeys]
                    else: 
                        i[jkeys] = j[jkeys]
                    while [] in i[jkeys]:
                        i[jkeys].remove([])
                poly_arrays.remove(j)
                i["polygon_coords"] = [list(x) for x in polygon_to_json(poly)]
        
        
        keys_to_remove = []
        for key in i.keys():
            if i[key] == [[]]:
                keys_to_remove.append(key)
        
        # Remove keys outside the loop
        for key in keys_to_remove:
            del i[key]
        
        if len(i.keys()) ==1:
            poly_arrays.remove(i)


    density_data.update_one(
        {"_id": cameraID},
        {"$set": {"cluster_properties": poly_arrays}}
    )
    

