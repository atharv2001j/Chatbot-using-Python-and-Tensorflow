# Import necessary libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


import nltk
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Load intents from a JSON file
intents = json.loads(open('E:\projects\Chatbot\chatbot\intents.json').read())

# Create empty lists for data processing
# for storing the duplicates words
words = []
# to store the different intents
classes = []
# To store the responses to the intent
documents = []

# Characters to ignore during processing
ignoreLetters = ['?', '!', '.', ',']

# Process intents and patterns
for intent in intents['intents']:
    # Here we create a bag of words (BoW)
    for pattern in intent['patterns']:
        # Tokenize words in the pattern
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Append a tuple of tokenized words and intent tag to documents
        documents.append((wordList, intent['tag']))
        # Add the intent tag to classes if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignored characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# Remove duplicates and sort the words
words = sorted(set(words))

# Sort and get unique classes
classes = sorted(set(classes))

# Save words and classes as pickled files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
# Create an empty list with zeros equal to the number of classes
outputEmpty = [0] * len(classes)

# Process documents to create training data
for document in documents:
    bag = []
    wordPatterns = document[0]
    # Lemmatize and normalize words in the pattern
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        # Create a bag of words indicating the presence of words in the pattern
        bag.append(1) if word in wordPatterns else bag.append(0)

    # Create output row with 1 at the index of the intent tag
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    # Combine bag of words and output row and append to training data
    training.append(bag + outputRow)

# Shuffle the training data
random.shuffle(training)
# Convert training data to NumPy array
training = np.array(training)

# Split data into input (X) and output (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build a neural network model using TensorFlow/Keras
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation='softmax'))


# Configure the model with SGD optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5', hist)
print('Done')
