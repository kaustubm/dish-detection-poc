import streamlit as st
import boto3
import requests
import json
import base64
from io import BytesIO
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pandas as pd
import folium
from streamlit_folium import folium_static
import time
from geopy.geocoders import Nominatim

# Set page config
st.set_page_config(page_title="Ultra-High Resolution Satellite Dish Detector", layout="wide")

def get_location_name(lat, lng):
    """Get location name from coordinates using geocoding"""
    try:
        geolocator = Nominatim(user_agent="dish-detector")
        location = geolocator.reverse(f"{lat}, {lng}", exactly_one=True)
        if location and location.address:
            return location.address
        return f"Unknown Location ({lat}, {lng})"
    except Exception as e:
        st.warning(f"Could not retrieve location name: {str(e)}")
        return f"Unknown Location ({lat}, {lng})"

def get_enhanced_satellite_image(lat, lng, api_key, zoom=21):
    """Get the clearest possible satellite image using multiple enhancement techniques"""
    # Get the highest resolution image possible from Google
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom={zoom}&size=640x640&scale=4&maptype=satellite&key={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image: {response.status_code}")
    
    image = Image.open(BytesIO(response.content))
    
    # Convert to RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Make a copy of the original image
    original_image = image.copy()
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(image)
    
    try:
        # Advanced image processing pipeline
        # 1. Apply histogram equalization for better contrast
        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # 2. Apply adaptive sharpening
        blurred = cv2.GaussianBlur(img_eq, (0, 0), 3)
        img_sharp = cv2.addWeighted(img_eq, 1.5, blurred, -0.5, 0)
        
        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img_sharp, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))  # Convert to list to make it mutable
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 4. Enhance edges for better dish boundary detection
        gray = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img_with_edges = cv2.addWeighted(img_clahe, 0.9, edges_rgb, 0.1, 0)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(img_with_edges)
        enhanced_no_edges = Image.fromarray(img_clahe)
        
        return original_image, enhanced_image, enhanced_no_edges
    
    except Exception as e:
        st.warning(f"Image enhancement failed: {str(e)}. Using original image.")
        return original_image, original_image, original_image

def analyze_with_bedrock(image, aws_region):
    """Use AWS Bedrock's Claude model to analyze for satellite dishes"""
    # Convert PIL Image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Create prompt for Claude
    prompt = """
    Analyze this ultra-high resolution satellite image very carefully and identify if there are any visible satellite dishes or antenna arrays.
    
    Satellite dishes typically appear as:
    - Small circular or oval objects (white, light gray, or metallic in color)
    - Often mounted on rooftops, balconies, or attached to walls
    - Usually aimed in a specific direction (not flat against surfaces)
    - Sometimes grouped together in arrays
    - Typically 0.5 to 2 meters in diameter
    - Often have a distinctive shadow due to their raised position
    - May appear as bright spots due to high reflectivity
    
    Please be extremely thorough in your examination. Look at every rooftop and structure visible in the image. Pay special attention to:
    - Rooftops of residential buildings and houses
    - Commercial building rooftops
    - Balconies and terraces
    - Side-mounted installations on walls
    
    Provide the following information:
    1. Are there any satellite dishes visible? Yes/No
    2. How many satellite dishes can you see? (Give an exact count)
    3. How confident are you in your assessment (low/medium/high)?
    4. Where are the dishes located in the image? (e.g., "on the rooftop of the large building in center")
    5. Is there anything unusual about the configuration of dishes (if any)? For example, an unusually high number in one location could indicate illegal redistribution.
    
    Format your response as a JSON object like this:
    {
        "dishes_detected": true/false,
        "dish_count": number,
        "confidence": "low"/"medium"/"high",
        "locations": "detailed description of where dishes are located",
        "unusual_configuration": true/false,
        "notes": "detailed description of what you see including any patterns"
    }
    
    Only respond with the JSON object, no additional text.
    """
    
    try:
        # Initialize AWS Bedrock client
        # This will automatically use the EC2 instance role
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
        
        # Create request body for Anthropic Claude model
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        # Call Bedrock with Claude 3 Haiku
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text']
        
        # Extract JSON from response
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            st.error(f"Failed to parse JSON from Claude response")
            st.code(response_text)
            return {
                "dishes_detected": False,
                "dish_count": 0,
                "confidence": "low",
                "locations": "",
                "unusual_configuration": False,
                "notes": "Error parsing response"
            }
            
    except Exception as e:
        st.error(f"Error calling AWS Bedrock: {str(e)}")
        return {
            "dishes_detected": False,
            "dish_count": 0,
            "confidence": "low",
            "locations": "",
            "unusual_configuration": False,
            "notes": f"Error: {str(e)}"
        }

def main():
    st.title("Satellite Dish Detector")
    st.write("Get the clearest possible satellite imagery for accurate dish detection")
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        google_api_key = st.text_input("Google Maps API Key", type="password")
        aws_region = st.selectbox("AWS Region", 
                               options=["ap-southeast-1", "us-east-1", "us-west-2", "eu-west-1"],
                               index=0)
        
        st.subheader("Image Settings")
        
        # Zoom level selector (higher zoom = closer view)
        zoom_level = st.slider("Satellite Image Zoom Level", min_value=19, max_value=22, value=21, 
                             help="Higher values zoom in closer (21-22 recommended for dish detection)")
        
        # Image enhancement options
        enhancement_option = st.radio(
            "Image Enhancement Mode",
            ["Original Image", "Enhanced", "Enhanced with Edge Detection"],
            index=2,
            help="Choose how to enhance the satellite imagery"
        )
        
        # Toggle side-by-side comparison
        side_by_side = st.checkbox("Show Original vs. Enhanced", value=True,
                                help="Display both original and enhanced images side by side")
        
        st.markdown("---")
        st.write("Powered by Claude 3 on AWS Bedrock")
    
    # Define tabs
    tab1, tab2 = st.tabs(["Scan Locations", "Results"])
    
    with tab1:
        st.header("Enter Coordinates")
        
        # Input method selection
        input_method = st.radio("Input Method", ["Single Location", "Multiple Locations"], horizontal=True)
        
        if input_method == "Single Location":
            # Single location input
            col1, col2 = st.columns(2)
            
            with col1:
                latitude = st.number_input("Latitude", value=3.1370, format="%.6f")
            
            with col2:
                longitude = st.number_input("Longitude", value=101.6839, format="%.6f")
            
            # Show preview map
            st.subheader("Location Preview")
            preview_map = folium.Map(location=[latitude, longitude], zoom_start=17)
            folium.Marker([latitude, longitude], tooltip="Scan Location").add_to(preview_map)
            folium_static(preview_map, width=800, height=400)
            
            if st.button("Scan This Location", type="primary"):
                if not google_api_key:
                    st.error("Google Maps API Key is required.")
                else:
                    with st.spinner("Fetching and enhancing ultra-high resolution satellite imagery..."):
                        # Get location name
                        location_name = get_location_name(latitude, longitude)
                        
                        # Fetch image
                        try:
                            # Get enhanced image
                            original_image, enhanced_edge_image, enhanced_no_edge_image = get_enhanced_satellite_image(
                                latitude, longitude, google_api_key, zoom=zoom_level
                            )
                            
                            # Select image for display and analysis based on user preference
                            if enhancement_option == "Original Image":
                                display_image = original_image
                                analysis_image = original_image
                                image_caption = "Original Satellite Image"
                            elif enhancement_option == "Enhanced":
                                display_image = enhanced_no_edge_image
                                analysis_image = enhanced_no_edge_image
                                image_caption = "Enhanced Satellite Image"
                            else:  # Enhanced with Edge Detection
                                display_image = enhanced_edge_image
                                analysis_image = enhanced_edge_image
                                image_caption = "Enhanced Satellite Image with Edge Detection"
                            
                            # Display images
                            if side_by_side and enhancement_option != "Original Image":
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(original_image, caption=f"Original Image (zoom: {zoom_level})", use_container_width=True)
                                with col2:
                                    st.image(display_image, caption=image_caption, use_column_width=True)
                            else:
                                # Just show the selected image
                                st.image(display_image, caption=f"{image_caption} of {location_name}", use_column_width=True)
                            
                            # Analyze with Bedrock
                            with st.spinner("Analyzing image with AI..."):
                                analysis = analyze_with_bedrock(analysis_image, aws_region)
                            
                            # Create result entry
                            result = {
                                "latitude": latitude,
                                "longitude": longitude,
                                "location_name": location_name,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "dishes_detected": analysis.get("dishes_detected", False),
                                "dish_count": analysis.get("dish_count", 0),
                                "confidence": analysis.get("confidence", "low"),
                                "locations": analysis.get("locations", ""),
                                "unusual_configuration": analysis.get("unusual_configuration", False),
                                "notes": analysis.get("notes", "")
                            }
                            
                            # Add to results
                            st.session_state.results.append(result)
                            
                            # Display result with good formatting
                            st.subheader("Detection Results")
                            if analysis.get("dishes_detected", False):
                                st.success(f"✅ Satellite dishes detected! ({analysis.get('dish_count', 0)} dishes)")
                                
                                # Create result card
                                with st.container(border=True):
                                    st.subheader(f"Analysis for {location_name}")
                                    st.markdown(f"**Number of Dishes:** {analysis.get('dish_count', 0)}")
                                    st.markdown(f"**Confidence:** {analysis.get('confidence', 'Low')}")
                                    st.markdown(f"**Dish Locations:** {analysis.get('locations', 'Not specified')}")
                                    
                                    if analysis.get("unusual_configuration", False):
                                        st.warning("⚠️ **Unusual configuration detected!** This may indicate illegal redistribution.")
                                    
                                    st.markdown("**Detailed Notes:**")
                                    st.info(analysis.get("notes", "No additional notes."))
                            else:
                                st.info("❌ No satellite dishes detected in this location.")
                                if analysis.get("notes"):
                                    st.text(analysis.get("notes"))
                            
                            # Display full analysis JSON for reference
                            with st.expander("View Raw Analysis Data"):
                                st.json(analysis)
                        
                        except Exception as e:
                            st.error(f"Error processing location: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
        
        else:  # Multiple Locations
            st.subheader("Multiple Locations")
            
            # Option to upload CSV
            st.write("Upload a CSV file with columns 'latitude' and 'longitude'")
            uploaded_file = st.file_uploader("Choose a file", type="csv")
            
            # Or enter manually
            st.write("Or enter coordinates manually (one per line in format: latitude,longitude)")
            coord_text = st.text_area("Coordinates", height=150, 
                                    placeholder="3.1370,101.6839\n3.1270,101.6939")
            
            if uploaded_file is not None:
                # Process uploaded CSV
                df = pd.read_csv(uploaded_file)
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    coordinates = df[['latitude', 'longitude']].values.tolist()
                    
                    # Show preview map
                    st.subheader("Locations Preview")
                    center_lat = coordinates[0][0]
                    center_lng = coordinates[0][1]
                    preview_map = folium.Map(location=[center_lat, center_lng], zoom_start=13)
                    
                    for i, (lat, lng) in enumerate(coordinates):
                        folium.Marker([lat, lng], tooltip=f"Location {i+1}").add_to(preview_map)
                    
                    folium_static(preview_map, width=800, height=400)
                    
                    if st.button("Scan All Locations", type="primary"):
                        process_multiple_locations(
                            coordinates, google_api_key, aws_region, 
                            zoom_level, enhancement_option
                        )
                else:
                    st.error("CSV must contain 'latitude' and 'longitude' columns.")
            
            elif coord_text.strip():
                # Process manually entered coordinates
                try:
                    coordinates = []
                    for line in coord_text.split('\n'):
                        if line.strip():
                            lat, lng = map(float, line.strip().split(','))
                            coordinates.append([lat, lng])
                    
                    if coordinates:
                        # Show preview map
                        st.subheader("Locations Preview")
                        center_lat = coordinates[0][0]
                        center_lng = coordinates[0][1]
                        preview_map = folium.Map(location=[center_lat, center_lng], zoom_start=13)
                        
                        for i, (lat, lng) in enumerate(coordinates):
                            folium.Marker([lat, lng], tooltip=f"Location {i+1}").add_to(preview_map)
                        
                        folium_static(preview_map, width=800, height=400)
                        
                        if st.button("Scan All Locations", type="primary"):
                            process_multiple_locations(
                                coordinates, google_api_key, aws_region, 
                                zoom_level, enhancement_option
                            )
                except Exception as e:
                    st.error(f"Error parsing coordinates: {str(e)}")
    
    with tab2:
        st.header("Satellite Dish Detection Results")
        
        if not st.session_state.results:
            st.info("No results yet. Scan locations to see results here.")
        else:
            # Create a results DataFrame
            results_df = pd.DataFrame(st.session_state.results)
            
            # Display a map with results
            st.subheader("Results Map")
            
            # Create map centered at first result
            center_lat = results_df['latitude'].iloc[0]
            center_lng = results_df['longitude'].iloc[0]
            results_map = folium.Map(location=[center_lat, center_lng], zoom_start=13)
            
            # Add markers for each result
            for _, row in results_df.iterrows():
                # Choose color based on detection
                if row['dishes_detected']:
                    color = 'red' if row['unusual_configuration'] else 'orange'
                    icon = 'glyphicon-alert' if row['unusual_configuration'] else 'glyphicon-signal'
                else:
                    color = 'green'
                    icon = 'glyphicon-ok'
                
                # Create popup content
                popup_content = f"""
                <b>{row['location_name']}</b><br>
                Dishes: {'Yes' if row['dishes_detected'] else 'No'}<br>
                Count: {row['dish_count']}<br>
                Confidence: {row['confidence']}<br>
                Location: {row.get('locations', '')}<br>
                Unusual: {'Yes' if row['unusual_configuration'] else 'No'}<br>
                """
                
                # Add marker
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"{row['dish_count']} dishes" if row['dishes_detected'] else "No dishes",
                    icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
                ).add_to(results_map)
            
            # Display the map
            folium_static(results_map, width=800, height=500)
            
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_locations = len(results_df)
                st.metric("Total Locations", total_locations)
            
            with col2:
                locations_with_dishes = sum(results_df['dishes_detected'])
                st.metric("Locations with Dishes", locations_with_dishes)
            
            with col3:
                unusual_configs = sum(results_df['unusual_configuration'])
                st.metric("Unusual Configurations", unusual_configs)
            
            # Display detailed results table
            st.subheader("Detailed Results")
            display_columns = ['location_name', 'latitude', 'longitude', 'dishes_detected', 
                             'dish_count', 'confidence', 'locations',
                             'unusual_configuration', 'notes', 'timestamp']
            # Make sure all columns exist
            display_columns = [col for col in display_columns if col in results_df.columns]
            st.dataframe(results_df[display_columns], use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results as CSV",
                csv,
                "satellite_dish_detection_results.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Clear results button
            if st.button("Clear All Results"):
                st.session_state.results = []
                st.rerun()

def process_multiple_locations(coordinates, google_api_key, aws_region, 
                              zoom_level, enhancement_option="Enhanced with Edge Detection"):
    """Process multiple locations and update results"""
    if not google_api_key:
        st.error("Google Maps API Key is required.")
        return
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each location
    for i, (lat, lng) in enumerate(coordinates):
        # Update progress
        progress = (i + 1) / len(coordinates)
        progress_bar.progress(progress)
        status_text.text(f"Processing location {i+1}/{len(coordinates)}...")
        
        try:
            # Get location name
            location_name = get_location_name(lat, lng)
            
            # Get enhanced image
            original_image, enhanced_edge_image, enhanced_no_edge_image = get_enhanced_satellite_image(
                lat, lng, google_api_key, zoom=zoom_level
            )
            
            # Select image for analysis based on user preference
            if enhancement_option == "Original Image":
                analysis_image = original_image
            elif enhancement_option == "Enhanced":
                analysis_image = enhanced_no_edge_image
            else:  # Enhanced with Edge Detection
                analysis_image = enhanced_edge_image
            
            # Analyze with Bedrock
            analysis = analyze_with_bedrock(analysis_image, aws_region)
            
            # Create result entry
            result = {
                "latitude": lat,
                "longitude": lng,
                "location_name": location_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dishes_detected": analysis.get("dishes_detected", False),
                "dish_count": analysis.get("dish_count", 0),
                "confidence": analysis.get("confidence", "low"),
                "locations": analysis.get("locations", ""),
                "unusual_configuration": analysis.get("unusual_configuration", False),
                "notes": analysis.get("notes", "")
            }
            
            # Add to results
            st.session_state.results.append(result)
            
            # Show brief status
            if analysis.get("dishes_detected", False):
                status_text.success(f"✅ {location_name}: {analysis.get('dish_count', 0)} dishes detected")
            else:
                status_text.info(f"❌ {location_name}: No dishes detected")
                
            # Small delay to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            status_text.error(f"Error processing location ({lat}, {lng}): {str(e)}")
            time.sleep(2)  # Longer delay after error
    
    # Complete
    progress_bar.progress(1.0)
    status_text.success(f"Completed scanning {len(coordinates)} locations!")

if __name__ == "__main__":
    main()