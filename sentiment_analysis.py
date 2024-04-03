import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# Load CSV file
csv_file_path = 'preprocessed.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)
# Assuming the text is in a column named 'text_column'
text_data = ' '.join(data['Tweet'])  # Concatenate all text from the column
# Load the PNG image to use as a mask
image_path = 'tweete.jpg'  # Replace with your PNG image path
mask = np.array(Image.open(image_path))

# Generate word cloud
wordcloud = WordCloud(mask=mask, background_color='white').generate(text_data)

# Display the word cloud with the image
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Display the image as a background
image = Image.open(image_path)
plt.imshow(image, alpha=0.6)
plt.axis('off')

plt.show()
import pandas as pd
from textblob import TextBlob

# Load CSV file
csv_file_path = 'Axisbank_tweets.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Assuming the text is in a column named 'text_column'
text_data = data['Tweet']  # Replace 'text_column' with your actual column name

# Perform sentiment analysis for each text in the column
sentiments = []

for text in text_data:
    blob = TextBlob(str(text))  # Convert to string and create a TextBlob object
    sentiment = blob.sentiment.polarity
    sentiments.append(sentiment)

# Add sentiment scores to the DataFrame
data['sentiment_score'] = sentiments

# Define the path for the new CSV file
new_csv_file_path = 'processeddata.csv'  # Replace with your desired file path

# Store the updated data in a new CSV file
data.to_csv(new_csv_file_path, index=False)

print(f"Data with sentiment scores saved to '{new_csv_file_path}'")
import pandas as pd
import re

# Load CSV file
csv_file_path = 'Axisbank_tweets.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)

# Assuming the text is in a column named 'text_column'
text_column_name = 'Tweet'  # Replace 'text_column' with your actual column name

# Function to remove specified characters and URLs from text
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove '@' and '#' signs
    text = text.replace('@', '').replace('#', '')
    
    return text

# Preprocess text data in the specified column
data[text_column_name] = data[text_column_name].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
new_csv_file_path = 'preprocessed.csv'  # Replace with your desired file path
data.to_csv(new_csv_file_path, index=False)

print(f"Preprocessed data saved to '{new_csv_file_path}'")
from textblob import TextBlob

# Assuming the text is in a column named 'text'
text_column_name = 'Tweet'  # Replace 'text' with your actual column name

# Function to classify sentiment into 'positive', 'neutral', or 'negative'
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment analysis function to each tweet in the DataFrame
data['sentiment'] = data[text_column_name].apply(get_sentiment)

# Count the number of neutral, positive, and negative tweets related to Axis Bank
neutral_count = data[data['sentiment'] == 'neutral'].shape[0]
positive_count = data[data['sentiment'] == 'positive'].shape[0]
negative_count = data[data['sentiment'] == 'negative'].shape[0]

# Display the counts
print("Number of Neutral Tweets:", neutral_count)
print("Number of Positive Tweets:", positive_count)
print("Number of Negative Tweets:", negative_count)
import matplotlib.pyplot as plt
sentiments = ['Neutral', 'Positive', 'Negative']
counts = [neutral_count, positive_count, negative_count]

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(sentiments, counts, color=['grey', 'green', 'red'])

# Add labels and title
plt.xlabel('Sentiments')
plt.ylabel('Count')
plt.title('Sentiment Analysis for Axis Bank Tweets')

# Show the plot
plt.show()