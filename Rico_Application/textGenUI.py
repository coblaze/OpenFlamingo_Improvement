
## Applying the model to UI Dataset and Descriptions

from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch
import os
import json

# Create model and transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
    )

# Load model checkpoint

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

# List all files in the directory containing your dataset
rico_data = os.listdir('/content/rico-dataset/rico_dataset_v0.1_semantic_annotations/semantic_annotations')

# Sort the files to ensure they are in the correct order
rico_data.sort()

# Select the first 6 items
rico_data = rico_data[:6]

# Separate the images and json files into two lists
ui_images = [file for file in rico_data if file.endswith('.png')]
ui_descriptions = [file for file in rico_data if file.endswith('.json')]

# For each json file, read the file and extract the description
#text_descriptions = []
#for file in ui_descriptions:
#    with open('path_to_your_dataset/' + file) as json_file:
#        data = json.load(json_file)
#        text_descriptions.append(data.get('description', ''))  # Use an empty string as default if 'description' is not found

ui_descriptions = []
for file in ui_descriptions_files:
    with open('/content/rico-dataset/rico_dataset_v0.1_semantic_annotations/semantic_annotations/' + file) as json_file:
        data = json.load(json_file)
        ui_descriptions.append(data.get('description', ''))  # Use an empty string as default if 'description' is not found

# Preprocess your text


tokenizer.padding_side = "left"

for i in range(len(ui_images)):
    vision_x = image_processor(Image.open('/content/rico-dataset/rico_dataset_v0.1_semantic_annotations/semantic_annotations/' + ui_images[i])).unsqueeze(0).unsqueeze(1).unsqueeze(0)
    lang_x = tokenizer(
        ["<image>" + desc + "<|endofchunk|>" for desc in ui_descriptions],
        return_tensors="pt",
)

# Generate text


generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text for image ", i, ": ", tokenizer.decode(generated_text[0]))

# Initialize a TfidfVectorizer
vectorizer = TfidfVectorizer()

# For each image-description pair
for i in range(len(ui_images)):
    # ... (existing code to generate text) ...

    # Get the original and generated descriptions
    original_description = ui_descriptions[i]
    generated_description = tokenizer.decode(generated_text[0])

    # Calculate the TF-IDF vectors of the original and generated descriptions
    tfidf_matrix = vectorizer.fit_transform([original_description, generated_description])

    # Calculate the cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    print("Similarity for image ", i, ": ", similarity[0][0])


# Print a sample of original and generated descriptions
for i in range(min(5, len(ui_images))):  # Print up to 5 samples
    print("Original description for image ", i, ": ", ui_descriptions[i])
    print("Generated description for image ", i, ": ", tokenizer.decode(generated_text[0]))


# Installing word Cloud
pip install wordcloud matplotlib


# Combine all generated text into one string
all_generated_text = ' '.join([tokenizer.decode(generated_text[0]) for generated_text in generated_texts])

# Create a WordCloud object
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = None,
                min_font_size = 10).generate(all_generated_text)

# Plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Combine your original descriptions and generated text into one list
all_text = ui_descriptions + generated_text

# Convert your text into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(all_text)

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

plt.figure(figsize=(10, 10))
sns.heatmap(cosine_sim_matrix, annot=True, cmap='coolwarm')
plt.show()