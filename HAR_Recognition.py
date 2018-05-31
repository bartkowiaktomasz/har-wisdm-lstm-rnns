import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from scipy import stats
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pickle

##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

RANDOM_SEED = 42

# Data preprocessing
SEGMENT_TIME_SIZE = 200
TIME_STEP = 200

# Model
N_CLASSES = 6
N_LSTM_LAYERS = 2
N_HIDDEN_NEURONS = 32
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters
L2_LOSS = 0.0015
LEARNING_RATE = 0.0025
EPOCHS_COUNT = 20
BATCH_SIZE = 64

##################################################
### FUNCTIONS
##################################################

# Returns a tenforflow LSTM NN
def createLSTM(input):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_NEURONS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_NEURONS, N_CLASSES]))
    }

    b = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_NEURONS], mean=1.0)),
        'output': tf.Variable(tf.Variable(tf.random_normal([N_CLASSES])))
    }

    X = tf.transpose(input, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + b['hidden'])
    hidden = tf.split(hidden, SEGMENT_TIME_SIZE, 0)

    # Stack two LSTM cells on top of each other
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_NEURONS, forget_bias=1.0)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_NEURONS, forget_bias=1.0)
    lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2])

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output'] + b['output'])

##################################################
### MAIN
##################################################
if __name__ == '__main__':

    # LOAD DATA
    data = pd.read_csv('data/WISDM_ar_v1.1_raw.txt', header=None, names=COLUMN_NAMES)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    data = data.dropna()

    # SHOW ACTIVITY GRAPH
    # data['activity'].value_counts().plot(kind='bar', title='Activity type')

    # DATA PREPROCESSING
    data_convoluted = []
    labels = []

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
        labels.append(label)

    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    print("Convoluted data shape: ", data_convoluted.shape)
    print("Labels shape:", labels.shape)


    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3, random_state=RANDOM_SEED)
    print("X train size: ", len(X_train))
    print("X test size: ", len(X_test))
    print("y train size: ", len(y_train))
    print("y test size: ", len(y_test))


    ##### BUILD A MODEL
    # Placeholders
    X = tf.placeholder(tf.float32, [None, SEGMENT_TIME_SIZE, N_FEATURES], name="input")
    y = tf.placeholder(tf.float32, [None, N_CLASSES])

    y_pred = createLSTM(X)
    pred_softmax = tf.nn.softmax(y_pred)

    # LOSS
    l2 = L2_LOSS * sum(tf.nn.l2_loss(i) for i in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)) + l2

    # OPTIMIZER
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # TRAINING
    saver = tf.train.Saver()

    history = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_count = len(X_train)

    for i in range(1, EPOCHS_COUNT + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
            sess.run(optimizer, feed_dict={X: X_train[start:end], y: y_train[start:end]})

        _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_train, y: y_train})
        _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, y: y_test})

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)

        if(i % 5 != 0):
            continue

        print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, y: y_test})
    print(f'final results: accuracy: {acc_final} loss: {loss_final}')

    # PLOT GRAPHS
    plt.figure(figsize=(12, 8))

    plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
    plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

    plt.plot(np.array(history['test_loss']), "r-", label="Test loss")
    plt.plot(np.array(history['test_acc']), "g-", label="Test accuracy")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training Epoch')
    plt.ylim(0)

    #plt.show()


    # CONFUSION MATRIX
    LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

    max_test = np.argmax(y_test, axis=1)
    max_predictions = np.argmax(predictions, axis=1)
    confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show();
