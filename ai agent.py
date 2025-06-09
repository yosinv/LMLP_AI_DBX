# Databricks notebook source
# Create a text input widget
dbutils.widgets.text("input_text", "", "What city?")


# COMMAND ----------

# %pip install --upgrade databricks-langchain langchain-community langchain databricks-sql-connector

# COMMAND ----------

# %pip install databricks-sqlalchemy

# COMMAND ----------

# %python
# %pip install transformers
# %pip install torch

# COMMAND ----------

# dbutils.library.restartPython() 

# COMMAND ----------

from langchain.sql_database import SQLDatabase
db = SQLDatabase.from_databricks(catalog="aidsf", schema="bright_initiative", engine_args={"pool_pre_ping": True})

# COMMAND ----------

from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(
       endpoint="databricks-meta-llama-3-3-70b-instruct",
       temperature=0.1,
       max_tokens=250,
   )

# COMMAND ----------

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# COMMAND ----------

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def detect_amenities(image_path, confidence_threshold=0.15):
    # Load and process the image
    image = Image.open(image_path)
     
    

    amenities_list = [
    "Wifi",
    "Kitchen",
    "TV",
    "Hot water",
    "Smoke alarm",
    "Hangers",
    "Dishes and silverware",
    "Hair dryer",
    "Cooking basics",
    "Essentials",
    "Free parking on premises",
    "Microwave",
    "Refrigerator",
    "Bed linens",
    "Iron",
    "Fire extinguisher",
    "Shampoo",
    "Self check-in",
    "Air conditioning",
    "Dining table",
    "Washer",
    "Dedicated workspace",
    "Freezer",
    "Heating",
    "Carbon monoxide alarm",
    "First aid kit",
    "Oven",
    "Toaster",
    "Extra pillows and blankets",
    "Private entrance",
    "Wine glasses",
    "Dishwasher",
    "Cleaning products",
    "Hot water kettle",
    "Coffee maker",
    "Outdoor dining area",
    "Body soap",
    "Outdoor furniture",
    "Stove",
    "Shower gel",
    "BBQ grill",
    "Coffee",
    "Room-darkening shades",
    "Baking sheet",
    "Pets allowed",
    "Bathtub",
    "Exterior security cameras on property",
    "Free street parking",
    "Long-term stays allowed",
    "Private patio or balcony",
    "Conditioner",
    "Blender",
    "Drying rack for clothing",
    "Dryer",
    "Ceiling fan",
    "Lockbox",
    "Clothing storage",
    "Barbecue utensils",
    "Patio or balcony",
    "Books and reading material",
    "Central heating",
    "Single level home",
    "Pool",
    "Keypad",
    "Wi-Fi",
    "Board games",
    "Elevator",
    "Host greets you",
    "Laundromat nearby",
    "Free washer – In unit",
    "Free dryer – In unit",
    "Portable fans",
    "Long term stays allowed",
    "Backyard",
    "Luggage drop-off allowed",
    "Indoor fireplace",
    "Fire pit",
    "Beach access – Beachfront",
    "Ethernet connection",
    "Central air conditioning",
    "Bidet",
    "Smart lock",
    "Mini fridge",
    "Waterfront",
    "Clothing storage: closet",
    "Cleaning available during stay",
    "Outdoor shower",
    "Sun loungers",
    "Crib",
    "High chair",
    "Pack ’n play/Travel crib",
    "Smoking allowed",
    "TV with standard cable",
    "Private backyard – Fully fenced",
    "Garden view",
    "Coffee maker: drip coffee maker",
    "Mountain view",
    "Children’s dinnerware",
    "Luggage dropoff allowed",
    "Safe",
    "Cocina",
    "Exercise equipment",
    "Children’s books and toys",
    "Hot tub",
    "Lock on bedroom door",
    "Lake access",
    "Dishes and cutlery",
    "Clothing storage: closet and dresser",
    "Electric stove",
    "Building staff",
    "Beach essentials",
    "Gym",
    "Bed linen",
    "Paid parking on premises",
    "Agua caliente",
    "Küche",
    "Föhn",
    "Microondas",
    "Refrigerador",
    "Platos y cubiertos",
    "Mosquito net",
    "WLAN",
    "Hammock",
    "Coffee maker: Keurig coffee machine",
    "Stainless steel oven",
    "City skyline view",
    "Fridge",
    "Outdoor playground",
    "Warmwasser",
    "Geschirr und Besteck",
    "Pool table",
    "Clothing storage: wardrobe",
    "Grundausstattung",
    "Plancha",
    "Shared beach access",
    "Private backyard – Not fully fenced",
    "Kleiderbügel",
    "Kühlschrank",
    "Rice cooker",
    "Grundausstattung zum Kochen",
    "Coffee maker: Nespresso",
    "Кухня",
    "Detector de humo",
    "Bettwäsche",
    "Gas stove",
    "Secadora de pelo",
    "Beach view",
    "Trash compactor",
    "Breakfast",
    "Washing machine",
    "Sea view",
    "Sound system",
    "Bügeleisen",
    "Mikrowelle",
    "Shared pool",
    "Kostenloser Parkplatz auf dem Grundstück",
    "Radiant heating",
    "Courtyard view",
    "AC – split-type ductless system",
    "Mesa de comedor",
    "Estacionamiento gratuito en las instalaciones",
    "Paid parking off premises",
    "Pool view",
    "Rauchmelder",
    "Babysitter recommendations",
    "Resort access",
    "Window guards",
    "Ocean view",
    "Dapur",
    "Congelador",
    "Shared patio or balcony",
    "Ping pong table",
    "Extintor de incendios",
    "Productos de limpieza",
    "Café",
    "Indoor fireplace: wood-burning",
    "Pocket wifi",
    "Almohadas y mantas adicionales",
    "Calefacción",
    "Fireplace guards",
    "Lavadora",
    "Single oven",
    "Copas de vino",
    "Bikes",
    "Esstisch",
    "Private hot tub",
    "Private backyard",
    "Window AC unit",
    "Baby safety gates",
    "Aire acondicionado",
    "Llegada autónoma",
    "Arbeitsplatz",
    "EV charger",
    "Feuerlöscher",
    "Coffee maker: espresso machine",
    "Ropa de cama",
    "مطبخ",
    "Outlet covers",
    "Portable heater",
    "Horno",
    "Free washer – In building",
    "Eigenständiger Check-in",
    "Free dryer – In building",
    "Lift",
    "Kettle",
    "Tostadora",
    "واي فاي",
    "Langzeitaufenthalte sind möglich",
    "Waschmaschine",
    "Wasserkocher",
    "Reinigungsprodukte",
    "Extrakissen und -decken",
    "Gefrierschrank",
    "Valley view",
    "Heizung",
    "WiFi",
    "Weingläser",
    "Lake view",
    "Bath",
    "Privater Eingang",
    "Холодильник",
    "Persianas o cortinas opacas",
    "Cafetera",
    "Cooker",
    "Air panas",
    "Shared gym in building",
    "Servicios imprescindibles",
    "الأطباق والأواني الفضية",
    "ماء ساخن",
    "Erste-Hilfe-Set",
    "Ofen",
    "Heating – split-type ductless system",
    "Klimaanlage",
    "AC - split type ductless system",
    "Lavavajillas",
    "HDTV",
    "Фен",
    "Bay view",
    "Shared hot tub",
    "Ganchos para la ropa",
    "Grill",
    "الضروريات",
    "Shared backyard – Fully fenced",
    "Room-darkening blinds",
    "Utensilios básicos para cocinar",
    "Stainless steel single oven",
    "Essbereich im Freien",
    "Duschgel",
    "شماعات",
    "Sábanas",
    "Abdunklung",
    "ثلاجة",
    "Kayak",
    "Телевизор",
    "ضروريات الطهي",
    "Kaffeemaschine",
    "Noise decibel monitors on property",
    "Piano",
    "Rice maker",
    "Free on-street parking",
    "Induction stove",
    "Geschirrspülmaschine",
    "Pengering rambut",
    "Gartenmöbel",
    "مفارش الأسرة",
    "Estacionamiento gratuito en la calle",
    "Paid washer – In building",
    "Wäscheständer für Kleidung",
    "مجفف الشعر",
    "Detector de monóxido de carbono",
    "Pack ’n play/Travel crib – available upon request",
    "Se permite dejar el equipaje",
    "Cuisine",
    "Cozinha",
    "Kostenlose Parkplätze an der Straße",
    "Elementos básicos",
    "Shared backyard – Not fully fenced",
    "Seife oder Duschgel",
    "Kohlenmonoxidmelder",
    "Keuken",
    "Hervidor de agua",
    "å»šæˆ¿",
    "Посуда и столовые приборы",
    "Горячая вода",
    "Wi-fi",
    "Paid street parking off premises",
    "Baby bath",
    "مكواة",
    "Entrada independiente",
    "Ganchos",
    "Arcade games",
    "تلفزيون",
    "مواقف سيارات مجانية في الأبنية",
    "Paid dryer – In building",
    "Entrada privada",
    "Kitchenette",
    "Jabón corporal",
    "Artículos básicos de cocina",
    "Herd",
    "Coffee maker: drip coffee maker, Keurig coffee machine",
    "الميكروويف",
    "Indoor fireplace: gas",
    "Предметы первой необходимости",
    "Champú",
    "Coffee maker: pour-over coffee",
    "Zona de trabajo",
    "Children's playroom",
    "Haustiere erlaubt",
    "Patio o balcón privado",
    "Free resort access",
    "Gel de ducha",
    "Kaffee",
    "جهاز إنذار للكشف عن الدخان",
    "Private pool",
    "Плечики",
    "Privater Innenhof oder Balkon",
    "Всё необходимое для приготовления еды",
    "Тостер",
    "Vaisselle et couverts",
    "Teplá voda",
    "Eau chaude"
    # ... (continues, add the rest as needed)
]

    # Process inputs for CLIP
    inputs = processor(
        text=[f"a photo of {amenity}" for amenity in amenities_list], 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    # Get similarity scores
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Filter amenities above confidence threshold
    detected_amenities = []
    for i, (amenity, confidence) in enumerate(zip(amenities_list, probs[0])):
        if confidence.item() > confidence_threshold:
            detected_amenities.append({
                'amenity': amenity,
                'confidence': round(confidence.item(), 3)
            })
    
    # Sort by confidence score
    detected_amenities.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detected_amenities

# Usage example
image_path = "/Workspace/Shared/ezgif.com-webp-to-png-converter.png"  # Replace with your local file path

amenities = detect_amenities(image_path, confidence_threshold=0.15)

# Extract just the amenity names for further processing
amenity_names = [item['amenity'] for item in amenities]
print("Detected amenities:", amenity_names)


# COMMAND ----------


# Retrieve the value of the text input widget
input_text = dbutils.widgets.get("input_text")

# Display the input text
print(f"User input: {input_text}")

# COMMAND ----------

text = f"Give me best place to stay in {input_text} by price which includes most of these amenities: {amenity_names}. Show the coordinates and URL only."
print(text)


# COMMAND ----------

output = agent.run(text)

# COMMAND ----------

output

# COMMAND ----------

output1a = agent.run(f"Search for the URL of the Airbnb property with in the SQL table: airbnb_properties_information - you will find a column 'URL' - make sure to match the location of the Airbnb with the User input {input_text} and use the {output} for context")

# COMMAND ----------

output1a

# COMMAND ----------

output2 = agent.run("Give me 5 best rated businesses in San Francisco close to {output}, show me only coordinates of places.")

# COMMAND ----------

output2

# COMMAND ----------

import ast
coordinates = ast.literal_eval(output2)
ref = (37.7711325,-122.4284177)#ast.literal_eval(output)


# COMMAND ----------

reference

# COMMAND ----------

import math

def calculate_distance(coord1, coord2):
    # Haversine formula to calculate the distance between two coordinates
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def order_coordinates(reference, coordinates):
    ordered_path = [reference]
    remaining_coords = coordinates.copy()

    while remaining_coords:
        # Find the nearest coordinate to the current reference
        nearest_coord = min(remaining_coords, key=lambda coord: calculate_distance(reference, coord))
        ordered_path.append(nearest_coord)
        remaining_coords.remove(nearest_coord)
        reference = nearest_coord  # Update reference to the newly added coordinate

    return ordered_path

reference = ref

ordered = order_coordinates(reference, coordinates)

# Print the ordered path
for coord in ordered:
    print(coord)


# COMMAND ----------

import urllib.parse

def create_google_maps_route(coordinates):
    base_url = "https://www.google.com/maps/dir/"
    coord_strings = [f"{lat},{lon}" for lat, lon in coordinates]
    route_url = base_url + "/".join(coord_strings)
    return route_url

# Generate the Google Maps route URL
route_url = create_google_maps_route(ordered)
print("Google Maps Route URL:", route_url)


# COMMAND ----------

%pip install plotly

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

# Function to display Google Maps
def show_google_maps(coordinates):
    map_url = f"https://www.google.com/maps?q={coordinates[0]},{coordinates[1]}"
    return map_url

# Load and display the image
image_path = "/Workspace/Shared/WIN_20250609_12_33_13_Pro.jpg"  # Replace with your local file path
image = Image.open(image_path)

# Display the image using Plotly
fig = px.imshow(image)
fig.update_layout(title="Uploaded Image")
fig.show()

# Example prompt and Airbnb selection
prompt = "Give me best place to stay in San Francisco by value based on these amenities: Wifi, Kitchen, TV. Show me the coordinates only."
airbnb_selected = "Airbnb in San Francisco with Wifi, Kitchen, TV"
coordinates = (37.7749, -122.4194)  # Example coordinates for San Francisco

# Display the prompt and selected Airbnb
fig_prompt = go.Figure()
fig_prompt.add_trace(go.Scatter(x=[0], y=[0], text=[prompt], mode="text"))
fig_prompt.update_layout(title="Agent Prompt", xaxis=dict(visible=False), yaxis=dict(visible=False))
fig_prompt.show()

fig_airbnb = go.Figure()
fig_airbnb.add_trace(go.Scatter(x=[0], y=[0], text=[airbnb_selected], mode="text"))
fig_airbnb.update_layout(title="Selected Airbnb", xaxis=dict(visible=False), yaxis=dict(visible=False))
fig_airbnb.show()

# Display Google Maps link
map_url = show_google_maps(coordinates)
fig_map = go.Figure()
fig_map.add_trace(go.Scatter(x=[0], y=[0], text=[f"<a href='{map_url}'>View on Google Maps</a>"], mode="text"))
fig_map.update_layout(title="Google Maps", xaxis=dict(visible=False), yaxis=dict(visible=False))
fig_map.show()

# COMMAND ----------

