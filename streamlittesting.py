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
from streamlit_functions import *
from collections import Counter
import time

load_dotenv()

@st.cache_data(ttl=900)
def get_clustering_density(cameraID):
    pass
    
client = connect_to_db()
"st.session_state object", st.session_state
if 'locations_arr' not in st.session_state:
    #Declaring default variables
    st.session_state.locations_arr = []
    st.session_state.num_of_dest = 1
    st.session_state.various_routes =[]
    st.session_state.various_routes_properties =[]
    st.session_state.option_selected_coordinate_arr = []
    st.session_state.prev_option_selected_coordinate_arr = []
    st.session_state.route_cluster_density_fig = go.Figure()
    st.session_state.all_selected = False
if "button_val" not in st.session_state:   
    st.session_state.button_val = False
if "submit_avail" not in st.session_state:
    st.session_state.submit_avail = False

st.title("Welcome LTA Traffic Monitoring")
st.session_state.num_of_dest = st.slider('Number of Destinations (Max:3)', 1,2)


for i in range(st.session_state.num_of_dest +1):
    if "entered_loc_{}".format(i) not in st.session_state:
        st.session_state["entered_loc_{}".format(i)] = None
        st.session_state["option_selected_{}".format(i)] = ""
        st.session_state["option_selected_coordinate_{}".format(i)]=""

        # st.session_state.locations_arr.append({})

for i in range(st.session_state.num_of_dest+1):
    # print("loc_arr:",st.session_state.locations_arr )
    print("current I:",  i)
    if i==0:
        st.text_input("Enter Start Point", on_change = trysuggestions_v2(i), key = "entered_loc_{}".format(i))
    elif i>0:
        st.text_input("Destination {}".format(str(i)), on_change = trysuggestions_v2(i), key = "entered_loc_{}".format(i))

    if st.session_state["entered_loc_{}".format(i)]!=None:
        option = st.selectbox("ST suggestion",list(st.session_state["locations_arr"][i].keys()), label_visibility= "collapsed", key = "option_selected_{}".format(i) )
        if st.session_state["option_selected_{}".format(i)] == "" and list(st.session_state["locations_arr"][i].keys()):
            st.session_state["option_selected_{}".format(i)] == list(st.session_state["locations_arr"][i].keys())[0]
        
        if not list(st.session_state["locations_arr"][i].keys()):
            st.warning("Please enter a valid destination")
            st.session_state["option_selected_{}".format(i)] == ""
            st.session_state.submit_avail = False
            
        else:
            st.session_state.submit_avail = True

    if st.session_state["option_selected_{}".format(i)] not in ["", None]:
        # st.session_state["option_selected_coordinate_{}".format(i)] = st.session_state["locations_arr"][i][st.session_state["option_selected_{}".format(i)]]
        if i<len(st.session_state.option_selected_coordinate_arr):
            st.session_state.option_selected_coordinate_arr[i] = st.session_state["locations_arr"][i][st.session_state["option_selected_{}".format(i)]]

        else:
            st.session_state.option_selected_coordinate_arr.append(st.session_state["locations_arr"][i][st.session_state["option_selected_{}".format(i)]])


if st.session_state.submit_avail == True:
    if st.button('Submit'):
        st.session_state.button_val = True
    print(st.session_state.option_selected_coordinate_arr)

st.session_state.map_object = folium.Map(location=(1.290270,103.851959), zoom_start =11)

if check_duplicate_coords() and st.session_state.button_val == True:
    print(check_duplicate_coords)
    st.warning("Error: 2 same locations. Please edit the locations again")
    st.session_state.button_val = False

if st.session_state.button_val == True:
    st.session_state.option_selected_coordinate_arr = st.session_state.option_selected_coordinate_arr[:st.session_state["num_of_dest"]+1]
    print("Running Routing LocationIQ")
    Routing_locationIQ_v2()
    st.session_state.button_val= False
    

    for i in range(len(st.session_state.various_routes)):
        current_route= st.session_state.various_routes[i]["geometry"]
        route_property_dict = {}
        route_property_dict["stacked_coord"] = pd.DataFrame({"path":[current_route], "color":[randomcolorvalue()]})
        new_path_1, new_path_2=  haversine_formula(route_property_dict["stacked_coord"]["path"][0])

        route_property_dict['stacked_coord'].loc[len(route_property_dict['stacked_coord'].index)] = [new_path_1,randomcolorvalue()]
        route_property_dict['stacked_coord'].loc[len(route_property_dict['stacked_coord'].index)] = [new_path_2,randomcolorvalue()]
        print(route_property_dict)

        req_cameras_json = asyncio.run(optimized_points_between_parallel_lines(new_path_1, new_path_2))

        camera_names = []
        for j in req_cameras_json.keys():
            _,location,_,_,_,_,_= get_camera_data(j)
            camera_names.append(location)
        route_property_dict["via"] = most_frequent_element(camera_names)
        route_property_dict["req_camera_df"]=pd.DataFrame({'CameraID':req_cameras_json.keys(), 'loc':req_cameras_json.values()})
        
        if i< len(st.session_state.various_routes_properties):
            st.session_state.various_routes_properties[i] = route_property_dict
        else:
            st.session_state.various_routes_properties.append(route_property_dict)

    # speed_path_pair= []
    
path_Layer = folium.FeatureGroup(name="Pathing Layer", show=True).add_to(st.session_state.map_object)
traffic_camera_Layer = folium.FeatureGroup(name="Camera Layer", show=True).add_to(st.session_state.map_object)
cluser_density_layer =  folium.FeatureGroup(name="Cluster Density Layer", show=True).add_to(st.session_state.map_object)
speed_layer = folium.FeatureGroup(name="Check Speed Layer", show=True).add_to(st.session_state.map_object)


if len(st.session_state.various_routes):
    columns = st.columns(len(st.session_state.various_routes))
    for i in range(len(columns)):
        # st.session_state["timing_{}".format(i)] = time(0,0)
        with columns[i]:
            #These buttons are used change the variables added to the map. 
            #Buttons are to indicate the duration and the route road passed.
            duration = st.session_state.various_routes[i]["duration"]
            minutes = int(duration // 60)
            via = st.session_state.various_routes_properties[i]["via"]
            # print(via)
            Route_selection_button = st.button(label = "Show Route {} \n via: {} \n Time(mins):{} ".format(i+1, via, minutes), key= "Route {}".format(i), use_container_width=True)
        
            if Route_selection_button:
                st.session_state['selected_route'] = i

    
    if 'selected_route' in st.session_state:

        # Async 1:# Plot to pathing to map
        start_time = time.time()
        for j in range(len(st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"].index)):
            #This for loop plot all the required lines
            route_string = route_string = "{}".format(st.session_state["option_selected_0"].split(",")[0])
            for k in range(1, len(st.session_state["option_selected_coordinate_arr"])):
                route_string += " TO {}".format(st.session_state["option_selected_{}".format(k)].split(",")[0])
            map_pathing  = folium.PolyLine(
                locations = st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['path'][j],
                tooltip= route_string,
                color = st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['color'][j],
                weight = 5,
                smoooth_factor = 30,
                )
            map_pathing.add_to(path_Layer)
            break
        print("--- %s plot pathing seconds ---" % (time.time() - start_time))

        #Async plot Camera to  map
        start_time = time.time()
        for l in range(len(st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"].index)):
            #Now i need to put the json into a table format: can try to convert json to dataframe. 

            # document = current_info.find_one({"_id": str(st.session_state.req_cameras_df['CameraID'][i])}, {"yolo_image":1, '_id':0})
            # blob = document.get('yolo_image') if document else None
            img_str,location,vehicle_count, clust_img, plot_img, orig_img, veh_plot= get_camera_data(
                str(st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['CameraID'][l]))
            vehicle_count_df = pd.DataFrame([vehicle_count])
            
            # Save to an HTML file without the index
            vehicle_count_html=vehicle_count_df.to_html(index=False, classes="table-striped table-hover table-condensed")
            iframe  = branca.element.IFrame(html(location, img_str, vehicle_count_html, orig_img, veh_plot), width=600, height=450)
            icon = folium.CustomIcon(
                camera_icon_url,
                icon_size=(38, 50),
                icon_anchor=(22, 50),
                popup_anchor=(-3, -30),
            )

            
            camera = folium.Marker(
                location = st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['loc'][l],
                tooltip= st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['CameraID'][l],
                lazy=True,
                icon = icon,
                )
            camera.add_child(folium.Popup(iframe, max_width=500))
            # camera.add_child(folium.Popup(vehicle_count_html))
            camera.add_to(traffic_camera_Layer)

            # graph_str = polygon_density_plotting(str(st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['CameraID'][l]))
            clust_iframe  = branca.element.IFrame(clust_html(location, clust_img, plot_img), width=600, height=450)

            clust_icon = folium.CustomIcon(
                graph_icon_url,
                icon_size=(38, 50),
                icon_anchor=(22, 100),
                popup_anchor=(-3, -30),
            )

            poly_graph = folium.Marker(
                location = st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['loc'][l],
                tooltip= st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['CameraID'][l],
                lazy=True,
                icon = clust_icon,
                )
            poly_graph.add_child(folium.Popup(clust_iframe, max_width=900))
            # camera.add_child(folium.Popup(vehicle_count_html))
            poly_graph.add_to(cluser_density_layer)



        print("--- %s plot camera seconds ---" % (time.time() - start_time))

        #Async Plot speed path to map
        start_time = time.time()
        current_route = st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['path'][0]
        path_1 = st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['path'][1]
        path_2 = st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['path'][2]
        speed_path_pair = asyncio.run(speed_labelling(current_route, path_1, path_2))
        # for m in st.session_state.various_routes_properties[st.session_state['selected_route']]["speed_path"]:
        # print(tuple(speed_path_pair))
        for m in speed_path_pair:
            print(m[0])
            if len(m[0]) == 0:
                continue 
            # print(m)
            if m[1] <= 30:
                color = '#FF0000'
            elif m[1]<=60:
                color = '#FFFF00'
            else:
                color="#008000"
            speed_pathing  = folium.PolyLine(
                locations = m[0] ,
                tooltip= f"Estimated Speed :{m[1]}",
                color = color,
                weight = 5,
                smoooth_factor = 30,
                )
            speed_pathing.add_to(speed_layer)
            speed_icon = folium.CustomIcon(
                car_speed_icon_url,
                icon_size=(38, 50),
                icon_anchor=(22, 50),
                popup_anchor=(-3, -30),
            )
            folium.Marker(
            location=m[0][0],
            popup=folium.Popup("currentSpeed: {}".format(m[1]), parse_html=True, max_width=100),
            icon=speed_icon
            ).add_to(speed_layer)
        folium.Marker(
            location=st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['path'][0][0],
            popup=folium.Popup("START", parse_html=True, max_width=100)
            ).add_to(speed_layer)
        
        folium.Marker(
            location=st.session_state.various_routes_properties[st.session_state['selected_route']]["stacked_coord"]['path'][0][-1],
            popup=folium.Popup("END", parse_html=True, max_width=100),
            ).add_to(speed_layer)
        
        print("--- %s plot speed_path_pair seconds ---" % (time.time() - start_time))
        # asyncio.run(run_all_uis())

        start_time = time.time()
        with st.container(border= True):
            if "timing" not in st.session_state:
                # current_time,  _ = find_time_window(datetime.now())
                # st.session_state["timing"] = datetime.time(current_time.hour, current_time.minute)
                
                # st.session_state["timing"] = current_time.time()
                st.session_state["timing"] = datetime.time(6,30)

            
            timing  = st.slider("Select a time window for cluster", 
                                value= st.session_state["timing"], 
                                step = timedelta(minutes = 15),
                                help= "Move me to select the time window to display the cluster density.\n Match the polygon colours to respective ones shown in Clustering Density Icon",
                                key = "timing"
            )

            st.write("Current time : {}".format(st.session_state["timing"]))
            cameraIDs = list(st.session_state.various_routes_properties[st.session_state['selected_route']]["req_camera_df"]['CameraID'])
            plot = asyncio.run(route_poly_plotting_df(cameraIDs))
            with st.popover("Density throughout Journey", use_container_width=True):
                st.plotly_chart(st.session_state.route_cluster_density_fig, key = "bruh")
        print("--- %s plot graph seconds ---" % (time.time() - start_time))
folium.LayerControl().add_to(st.session_state.map_object)
with st.container(border = True):
    st_data = st_folium(st.session_state.map_object, key = "fig1",width =690, height= 550, returned_objects=[])

st.write("")

    






