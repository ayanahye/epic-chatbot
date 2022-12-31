import random
import json
import numpy
import nltk
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

nltk.download('punkt') # tokenizes text, spliting it into words
from nltk.stem.lancaster import LancasterStemmer # helps get the root word
stemmer = LancasterStemmer()

app = Flask(__name__)

# configuration for the flask application, users messages are stored in the database using the SQLALCHEMY library
# specifying the database location
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'

db = SQLAlchemy(app) # database = the SQLALchemy Flask App

# initialize flask-migrate
migrate = Migrate(app, db)

# create the message model for the database
# message model has an id, sender, and message
class Message(db.Model):
    __tablename__ = 'messages'  # specify the table name
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(80))
    message = db.Column(db.String(120))

# Create the message table
with app.app_context():
    db.create_all()

# process data into the intents file
def process_intents(intents_file):
    words = []
    labels = []
    docs = []

    # load the data from the json file
    with open(intents_file) as file:
        data = json.load(file)
    
    # loop through the intents (the tags)
    for intent in data["intents"]:
        # loop through the patterns in the intent (recognize the input)
        for pattern in intent["patterns"]:
            # Make sure the pattern is a string before tokenizing it
            if isinstance(pattern, str):
                # tokenize the pattern and add it to the list of words
                # splitting a large sample of text into words
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds) # add individual words not a list
                # add a tuple of the tokenized pattern and the tag to the docs list
                docs.append((wrds, intent["tag"]))

    # stem the words and remove the question marks
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"] 
    # sort the words in a list, removing duplicates (set() is unique elements)
    words = sorted(list(set(words)))
    # get the labels from the docs list and sort them (the tag)
    labels = sorted(list(set([doc[1] for doc in docs])))

    return words, labels, docs

# function to create a bag of words representation of the docs
def bag_of_words(words, docs):
    # initialize the lists for X and Y 
    X = []
    y = []

    # loop through the docs (contains words with the tag)
    for doc in docs:
        # empty bag of words
        bag = []
        # taking the word in the doc and stemming it to its root
        wrds = [stemmer.stem(w) for w in doc[0]]
        # loop through all the list of rooted words
        for w in words:
            # if word is in the doc words
            if w in wrds:
                # add 1 to the bag
                bag.append(1)
            else:
                # add 0, since its not in the docs words
                bag.append(0)

        # adding the bag list of 1s and 0s to the X list
        X.append(bag)
        # adding all the tags to the Y list
        y.append(doc[1])

    return X, y

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the intents data
words, labels, docs = process_intents("intents.json")

# Create a bag of words representation of the documents
# X is the list of lists of 1s and 0s, Y is the tags for each document corresponding to the bag of words (1 or 0)
X, y = bag_of_words(words, docs)

# Split the data into training and test sets
# train_test_split will split the data into two sets: a training and a testing set. 
# 20% of the data is used for testing and 80% is used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier() # defining the object
# train the data using the fit method.
model.fit(X_train, y_train)

# Test the model, score is the avg accuracy of the test samples
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")

# Function to classify a message
def classify(X, y, message, model, words, labels):
    # faster processing array with larger sets of data
    X = numpy.array(X)
    # take the message and break it into words
    message = nltk.word_tokenize(message)
    # get the root words
    message = [stemmer.stem(w.lower()) for w in message if w not in "?"]
    # take the message and put it into bag-of-words function 
    new_X, new_y = bag_of_words(words, [(message, "")])
    new_X = new_X[0]  # get the first (and only) element in the list
    new_y = new_y[0]  # get the first (and only) element in the list
    # convert new_X to a numpy array to get faster proccessing
    new_X = numpy.array(new_X)
    # must reshape the array since the machine learning model expects a 2d array with a row (1) & collumns=how many elements there are (-1 for numpy to infer this)
    new_X = new_X.reshape(1, -1)  # reshape the array to a 2D array with a single row
    pred = model.predict(new_X)[0]  # predict the label for the single sample, and get the only element of the array
    # try to convert the predicted label to an integer, here for troubleshooting 
    try:
        pred = int(pred)
    except ValueError:
        # Handle the error here
        pass
    # find the index of the predicted tag in the labels list
    pred_index = labels.index(pred) # pred is one of the labels, indec will return its number in the labels list
    return pred_index, labels[pred_index] # returns the predicted label and the index of that label (the label is the tag)

# Flask routes
@app.route('/')
def index():
    # Get all messages from the database
    messages = Message.query.all()
    return render_template('index.html', messages=messages)

@app.route('/send_message', methods=['POST'])
def send_message():
    # Get the message from the form
    message = request.form['message']

    num_messages = Message.query.count()

    if num_messages > 20:
        db.session.query(Message).delete()
        db.session.commit()

    # Classify the message
    # pred should be the index and label should be the predicted label
    pred, label = classify(X, y, message, model, words, labels)

    # Get the response for the classified message
    with open("intents.json") as file:
        intents = json.load(file)
        for intent in intents['intents']:
            if intent['tag'] == label: # if tag = predicted tag
                response = random.choice(intent['responses']) # get a random response from that tags responses

    # Create a new message object
    new_message = Message(sender="me", message=message)

    # Add the message to the database
    db.session.add(new_message)
    db.session.commit()

    # Create a new message object for the response
    response_message = Message(sender="bot", message=response)

    # Add the response to the database
    db.session.add(response_message)
    db.session.commit()

    return redirect(url_for('index')) 

@app.route('/delete_messages', methods=['POST'])
def delete_messages():
    # Delete all messages from the database
    num_rows_deleted = db.session.query(Message).delete()
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run()

