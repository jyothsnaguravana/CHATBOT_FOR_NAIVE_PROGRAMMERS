import random
import json
import pickle
import numpy as np
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.layers import Embedding
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag =[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training , dtype=object)


train_x = list(training[:, 0])
train_y = list(training[:, 1])
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
#LSTM model starts here 
embedding_vector_features=45
voc_size = 50
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features))
model.add(LSTM(458,input_shape=(training.shape),activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(458,activation='relu'))
model.add(Dropout(0.2))
#LSTM model ends here

optimizer = Adam(learning_rate=0.001) # changed learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=32, verbose=1) # increased epochs and batch size
model.save('chatbotmodel.h5')