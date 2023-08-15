import spacy
from langdetect import detect
import matplotlib.pyplot as plt
import numpy as np


def formality_score(text, nlp_model):
    doc = nlp_model(text)
    total_verbs = sum(1 for token in doc if token.pos_ == 'VERB')
    total_words = len(doc)
    uppercase_words = sum(1 for token in doc if token.text.isupper())
    
    if total_words == 0:
        return 0  # Pour éviter une division par zéro si le texte est vide

    verb_frequency = total_verbs / total_words
    uppercase_frequency = uppercase_words / total_words
    formality_score = verb_frequency - uppercase_frequency
    return formality_score
def punctuation_frequency(text, nlp_model):
    doc = nlp_model(text)
    total_tokens = len(doc)
    punctuation_tokens = sum(1 for token in doc if token.is_punct)
    
    if total_tokens == 0:
        return 0  # Pour éviter une division par zéro si le texte est vide

    punctuation_frequency = punctuation_tokens / total_tokens
    return punctuation_frequency
def average_word_length(text, nlp_model):
    doc = nlp_model(text)
    total_length = 0
    total_words = 0
    max_word_length = float('-inf')  # Initialiser à une valeur négative infinie

    for token in doc:
        if token.is_alpha:
            word_length = len(token.text)
            total_length += word_length
            total_words += 1
            if word_length > max_word_length:
                max_word_length = word_length

    if total_words == 0:
        return 0  # Pour éviter une division par zéro si le texte est vide

    average_length = total_length / total_words
    normalized_average_length = average_length / max_word_length
    return normalized_average_length

def lexical_diversity(text, nlp_model):
    doc = nlp_model(text)
    total_words = len(doc)
    unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
    lexical_diversity_ratio = unique_words / total_words
    return lexical_diversity_ratio
def rare_word_score(text, nlp_model):
    doc = nlp_model(text)
    total_words = len(doc)
    word_counts = {}
    
    if total_words == 0:
        return 0  # To avoid division by zero if the text is empty
    
    for token in doc:
        if token.is_alpha:
            word_counts[token.text.lower()] = word_counts.get(token.text.lower(), 0) + 1
    
    rare_word_score = 0
    for word, word_count in word_counts.items():
        if word_count <= 1:  # Consider words with count less than or equal to 5 as rare
            rare_word_score += (1 - (word_count / total_words))
    
    rare_word_score /= len(word_counts)  # Calculate the average rare word score
    return rare_word_score

def complexity_score(text, nlp_model):
    doc = nlp_model(text)
    total_words = len(doc)
    total_complex_words = sum(1 for token in doc if token.is_alpha and len(token.text) >= 6)  # Define a threshold for complexity
    
    if total_words == 0:
        return 0  # To avoid division by zero if the text is empty
    
    complexity_score = total_complex_words / total_words
    return complexity_score



def error_management_score(text, nlp_model):
    doc = nlp_model(text)
    total_words = len(doc)
    error_count = 0  # Initialize the error count

    # Define part-of-speech tags that are considered errors (e.g., nouns in this case)
    error_pos_tags = ["NOUN"]

    for token in doc:
        if is_error(token, error_pos_tags):  # Use your error detection logic
            error_count += 1
    
    if total_words == 0:
        return 0  # To avoid division by zero if the text is empty
    
    error_ratio = error_count / total_words
    error_score = 0.8 - error_ratio  # Higher error ratio leads to a lower score

    return error_score
def is_error(token, error_pos_tags):
    return token.pos_ in error_pos_tags


# Charger le modèle de langue anglais
nlp_en = spacy.load('en_core_web_sm')

# Charger le modèle de langue français
nlp_fr = spacy.load('fr_core_news_sm')
# Charger le modèle de langue arabe
nlp_ar = spacy.load('xx_ent_wiki_sm')

# Fonction pour ouvrir un fichier et lire son contenu
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Define the file path to the text you want to analyze
file_path = r'C:\Users\LENOVO\Desktop\stage\final\english.txt'

# Read the content of the text file
text = read_text_file(file_path)

# Detect the language of the text
detected_lang = detect(text)

# Choose the appropriate language model based on the detected language
if detected_lang == 'en':
    print("Detected Language: English")
    nlp_model = nlp_en
elif detected_lang == 'fr':
    print("Detected Language: French")
    nlp_model = nlp_fr
else:
    print("Detected Language: Arabe")
    nlp_model = nlp_ar

# Calculate the various linguistic features using the provided functions
formality = formality_score(text, nlp_model)
punctuation_freq = punctuation_frequency(text, nlp_model)
normalized_avg_word_length = average_word_length(text, nlp_model)
diversity_ratio = lexical_diversity(text, nlp_model)
score = rare_word_score(text, nlp_model)
complexity = complexity_score(text, nlp_model)

error_score = error_management_score(text, nlp_model)

# Normalize the complexity score between 0 and 1
normalized_complexity = (complexity - 0) / (1 - 0)  # Assuming complexity ranges from 0 to 1


# Print the calculated scores
print("Formality Score:", formality)
print("Punctuation Frequency:", punctuation_freq)
print("Normalized Average Word Length:", normalized_avg_word_length)
print("Lexical Diversity Ratio:", diversity_ratio)
print("Rare Word Score:", score)
print("Normalized Complexity Score:", normalized_complexity)

print("Error Management Score:", error_score)

# Create a radar chart to visualize the results
categories = ['Formality', 'Punctuation Frequency', 'Normalized Avg. Word Length', 'Lexical Diversity', 'Rare Word Score', 'Normalized Complexity Score','error_score']
values = [formality, punctuation_freq, normalized_avg_word_length, diversity_ratio, score, normalized_complexity,error_score]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(-0.5 * np.pi)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
ax.set_xticks(theta)
ax.set_xticklabels(categories)

ax.set_ylim(0, 1)

line = ax.plot(theta, values, color='b', marker='o', linestyle='-')
ax.fill(theta, values, color='b', alpha=0.25)

plt.title("Text Features Analysis", y=1.08)

plt.show()
