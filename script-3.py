import pandas as pd
import spacy
import folium
from folium.plugins import HeatMap
from collections import Counter

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('issr-task-2.csv')

# Function to extract locations using spaCy NER
def extract_locations(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    return locations if locations else None

# Apply location extraction to the existing DataFrame
df['Locations'] = df['Content'].apply(extract_locations)

# Display results
print(df[['Content', 'Sentiment', 'Risk_level', 'Locations']])

# Count occurrences of each location
location_counts = df['Locations'].explode().value_counts()

print("\nTop 5 locations mentioned:")
print(location_counts.head())