import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_databricks import ChatDatabricks
from databricks.sdk import WorkspaceClient
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import mlflow
import asyncio
import json
import os
from dotenv import load_dotenv

# Setup
mlflow.langchain.autolog()
load_dotenv()

# Configure workspace tokens
w = WorkspaceClient()
os.environ["DATABRICKS_HOST"] = w.config.host
os.environ["DATABRICKS_TOKEN"] = w.tokens.create(comment="for model serving", lifetime_seconds=1200).token_value

# Initialize LLM
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")

# Setup MCP client
NIMBLE_API_KEY = '70cfe596186e4383862076cbb11eeaafc16b2d5d2f9d40c2a83af5a3fa8857ab'
client = MultiServerMCPClient({
    "nimble": {
        "url": "https://mcp.nimbleway.com/sse",
        "transport": "sse",
        "headers": {
            "Authorization": f"Bearer {NIMBLE_API_KEY}"
        }
    }
})

def find_accessible_airbnb_properties(location: str) -> pd.DataFrame:
    """
    Queries the Bright Initiative Airbnb dataset for properties in a specific location
    that have reviews mentioning "wheelchair".
    """
    query = f"""
        SELECT
        listing_name,
        location_details,
        location,
        details,
        description_by_sections,
        reviews
        FROM `aidsf`.bright_initiative.airbnb_properties_information_csv
        WHERE 1=1
        AND location ILIKE '%{location}%'
        AND host_number_of_reviews > 100
        AND EXISTS(
            FROM_JSON(reviews, 'array<struct<review:string>>'),
            review -> review.review ILIKE '%wheelchair%'
        )
        LIMIT 50
    """
    return spark.sql(query).toPandas()

def find_accessible_hotels_properties(location: str) -> pd.DataFrame:
    """
    Queries the Bright Initiative hotel dataset for properties in a specific location
    that have reviews mentioning "wheelchair".
    """
    query = f"""
        SELECT
        title,
        location,
        city,
        property_information,
        reviews_scores,
        number_of_reviews,
        top_reviews
        FROM `aidsf`.bright_initiative.booking_hotel_listings_csv
        WHERE 1=1
        AND location ILIKE '%{location}%'
        AND number_of_reviews > 100
        AND EXISTS(
            FROM_JSON(top_reviews, 'array<struct<review:string>>'),
            review -> review.review ILIKE '%wheelchair%'
        )
        LIMIT 50
    """
    return spark.sql(query).toPandas()

async def add_google_maps_links(properties_data: dict) -> dict:
    """
    Uses MCP agent to add Google Maps links to property data.
    """
    # Get tools from MCP client
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    
    # Process Airbnb properties
    for property_item in properties_data.get('airbnb_properties', []):
        property_name = property_item.get('listing_name', '')
        location = property_item.get('location', '') or property_item.get('location_details', '')
        
        if property_name and location:
            # Create search query for Google Maps
            search_query = f"{property_name} {location}"
            
            try:
                # Use agent to find Google Maps link
                response = await agent.ainvoke({
                    "messages": [{
                        "role": "user", 
                        "content": f"Find the Google Maps link for: {search_query}. Return only the maps.google.com URL."
                    }]
                })
                
                # Handle different response structures
                content = ""
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, dict):
                    if 'messages' in response and response['messages']:
                        last_message = response['messages'][-1]
                        if hasattr(last_message, 'content'):
                            content = last_message.content
                        elif isinstance(last_message, dict) and 'content' in last_message:
                            content = last_message['content']
                elif hasattr(response, 'messages') and response.messages:
                    last_message = response.messages[-1]
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                
                # Extract the Google Maps URL from response
                maps_url = extract_maps_url(content)
                if maps_url:
                    property_item['google_maps_url'] = maps_url
                    
            except Exception as e:
                print(f"Error getting maps link for {property_name}: {e}")
                property_item['google_maps_url'] = f"https://maps.google.com/maps?q={property_name.replace(' ', '+')}+{location.replace(' ', '+')}"
    
    # Process Hotel properties
    for property_item in properties_data.get('hotel_properties', []):
        property_name = property_item.get('title', '')
        location = property_item.get('location', '') or property_item.get('city', '')
        
        if property_name and location:
            # Create search query for Google Maps
            search_query = f"{property_name} {location}"
            
            try:
                # Use agent to find Google Maps link
                response = await agent.ainvoke({
                    "messages": [{
                        "role": "user", 
                        "content": f"Find the Google Maps link for: {search_query}. Return only the maps.google.com URL."
                    }]
                })
                
                # Extract the Google Maps URL from response
                maps_url = extract_maps_url(response.get('messages', [{}])[-1].get('content', ''))
                if maps_url:
                    property_item['google_maps_url'] = maps_url
                    
            except Exception as e:
                print(f"Error getting maps link for {property_name}: {e}")
                property_item['google_maps_url'] = f"https://maps.google.com/maps?q={property_name.replace(' ', '+')}+{location.replace(' ', '+')}"
    
    return properties_data

def extract_maps_url(text: str) -> str:
    """
    Extracts Google Maps URL from text response.
    """
    import re
    
    # Look for Google Maps URLs
    maps_patterns = [
        r'https://maps\.google\.com[^\s]+',
        r'https://www\.google\.com/maps[^\s]+',
        r'maps\.google\.com[^\s]+',
    ]
    
    for pattern in maps_patterns:
        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            # Ensure it starts with https://
            if not url.startswith('http'):
                url = 'https://' + url
            return url
    
    return None

async def find_all_accessible_properties_with_maps(location: str) -> str:
    """
    Combines both Airbnb and hotel data for a given location and adds Google Maps links.
    Returns formatted JSON string with both property types and maps links.
    """
    # Get both datasets
    airbnb_df = find_accessible_airbnb_properties(location)
    hotels_df = find_accessible_hotels_properties(location)
    
    # Add property type identifier
    airbnb_df['property_type'] = 'Airbnb'
    hotels_df['property_type'] = 'Hotel'
    
    # Create combined data
    combined_data = {
        'airbnb_properties': airbnb_df.to_dict('records'),
        'hotel_properties': hotels_df.to_dict('records'),
        'total_airbnb_count': len(airbnb_df),
        'total_hotel_count': len(hotels_df)
    }
    
    # Add Google Maps links using MCP agent
    enhanced_data = await add_google_maps_links(combined_data)
    
    return json.dumps(enhanced_data, indent=2)

# Enhanced prompt template that includes Google Maps links
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful assistant for accessible travel. Your goal is to summarize potential Airbnb and hotel listings for a user.

    The following listings *mention* wheelchairs but may not actually be accessible. Closely review the descriptions and reviews,
    and then summarize the accessibility features (or lack thereof) for both property types.

    Please organize your response by:
    1. Airbnb Properties - highlighting the most promising accessible options with their Google Maps links
    2. Hotel Properties - highlighting the most promising accessible options with their Google Maps links
    3. Overall recommendations based on accessibility features mentioned

    For each property, include the Google Maps link if available to help users easily find the location.

    Here is the JSON data containing both Airbnb and hotel properties with Google Maps links:
    {context}
    """
)

async def get_accessible_properties_with_maps(location: str) -> str:
    """
    Main function to get accessible properties with Google Maps links and LLM analysis.
    """
    # Get properties with maps links
    properties_json = await find_all_accessible_properties_with_maps(location)
    
    # Create the chain for LLM processing
    chain = (
        prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Process with LLM
    result = chain.invoke({"context": properties_json})
    return result

# Example usage
async def main():
    location = "New York"
    print(f"Finding accessible properties in {location} with Google Maps links...")
    
    result = await get_accessible_properties_with_maps(location)
    print(result)

# Run the enhanced system
async def run_example():
    location = "New York"
    print(f"Finding accessible properties in {location} with Google Maps links...")
    
    result = await get_accessible_properties_with_maps(location)
    print(result)
    return result

await run_example()            
