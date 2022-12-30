"""import json
import pickle
import numpy
import nltk
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
db = SQLAlchemy(app)

migrate = Migrate(app, db)

class Message(db.Model):
    __tablename__ = 'messages'  # specify the table name
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(80))
    message = db.Column(db.String(120))

# Create the message table
with app.app_context():
    db.create_all()

def process_intents(intents_file):
    words = []
    labels = []
    docs = []

    with open(intents_file) as file:
        data = json.load(file)
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                # Make sure the pattern is a string before tokenizing it
                if isinstance(pattern, str):
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs.append((wrds, intent["tag"]))

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"] 
    words = sorted(list(set(words)))

    labels = sorted(list(set([doc[1] for doc in docs])))

    return words, labels, docs

def bag_of_words(words, docs):
    X = []
    y = []

    for doc in docs:
        bag = []
        wrds = [stemmer.stem(w) for w in doc[0]]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        X.append(bag)
        y.append(doc[1])

    return X, y

# Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the intents data
words, labels, docs = process_intents("intents.json")

# Create a bag of words representation of the documents
X, y = bag_of_words(words, docs)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")

# Function to classify a message
def classify(X, y, message, model, words, labels):
    X = numpy.array(X)
    message = nltk.word_tokenize(message)
    message = [stemmer.stem(w.lower()) for w in message if w not in "?"]
    new_X, new_y = bag_of_words(words, [(message, "")])
    new_X = new_X[0]  # get the first (and only) element in the list
    new_y = new_y[0]  # get the first (and only) element in the list
    new_X = new_X.reshape(1, -1)  # reshape the array to a 2D array with a single row
    pred = int(model.predict(new_X)[0])  # predict the label for the single sample
    return pred, labels[pred]

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        pred, tag = classify(X, y, message, model, words, labels)
        response = labels[pred]
        
        # Save the message to the database
        new_message = Message(sender="Me", message=message)
        db.session.add(new_message)
        db.session.commit()

        # Redirect to the index page and display the response
        return redirect(url_for('index', response=response))
    else:
        # Fetch all messages from the database and pass them to the template
        messages = Message.query.all()
        return render_template('index.html', messages=messages)

# Function to send a message
@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    sender = "me"
    new_message = Message(sender=sender, message=message)
    db.session.add(new_message)  # add the new message to the database
    db.session.commit()  # commit the change to the database
    return redirect('/')

@app.route('/get_messages')
def get_messages():
    messages = Message.query.all()  # get all the messages from the database
    return render_template('messages.html', messages=messages)

# Add a new route for deleting all messages
@app.route('/delete_messages', methods=['POST'])
def delete_messages():
    # Delete all messages from the database
    num_deleted = db.session.query(Message).delete()
    db.session.commit()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
"""

import random
import json
import pickle
import numpy
import nltk
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
db = SQLAlchemy(app)

migrate = Migrate(app, db)

class Message(db.Model):
    __tablename__ = 'messages'  # specify the table name
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(80))
    message = db.Column(db.String(120))

# Create the message table
with app.app_context():
    db.create_all()

def process_intents(intents_file):
    words = []
    labels = []
    docs = []

    with open(intents_file) as file:
        data = json.load(file)
        
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Make sure the pattern is a string before tokenizing it
            if isinstance(pattern, str):
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs.append((wrds, intent["tag"]))

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"] 
    words = sorted(list(set(words)))

    labels = sorted(list(set([doc[1] for doc in docs])))

    return words, labels, docs

def bag_of_words(words, docs):
    X = []
    y = []

    for doc in docs:
        bag = []
        wrds = [stemmer.stem(w) for w in doc[0]]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        X.append(bag)
        y.append(doc[1])

    return X, y

# Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the intents data
words, labels, docs = process_intents("intents.json")

# Create a bag of words representation of the documents
X, y = bag_of_words(words, docs)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")

# Function to classify a message
def classify(X, y, message, model, words, labels):
    X = numpy.array(X)
    message = nltk.word_tokenize(message)
    message = [stemmer.stem(w.lower()) for w in message if w not in "?"]
    new_X, new_y = bag_of_words(words, [(message, "")])
    new_X = new_X[0]  # get the first (and only) element in the list
    new_y = new_y[0]  # get the first (and only) element in the list
    new_X = numpy.array(new_X)
    new_X = new_X.reshape(1, -1)  # reshape the array to a 2D array with a single row
    pred = model.predict(new_X)[0]  # predict the label for the single sample
    
    try:
        pred = int(pred)
    except ValueError:
        # Handle the error here
        pass
    pred_index = labels.index(pred)
    return pred_index, labels[pred_index]

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

    # Classify the message
    pred, label = classify(X, y, message, model, words, labels)

    # Get the response for the classified message
    with open("intents.json") as file:
        intents = json.load(file)
        for intent in intents['intents']:
            if intent['tag'] == label:
                response = random.choice(intent['responses'])

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

