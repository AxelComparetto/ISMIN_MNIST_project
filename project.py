## Library importation and data load ##
#libraires
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn import mixture
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras import Model
from prettytable import PrettyTable
import random

#data
print(os.getcwd()) #may need to run the entire script once to set the current working directory
X = np.load("MNIST_X_28x28.npy") #images
y = np.load("MNIST_y.npy") #labels

#lists that will store results
res_model = []
res_param = []
res_train_acc = []
res_valid_acc = []
res_test_acc = []
random.seed(42)

print("load done: libraries and data ")

##  What are the shape of the data? Display samples from the dataset ##

print("The dataset shape is", X.shape)
fav_sample = 42
plt.figure(figsize=(8,4.5))
for i in range(0,8): #displays 8 samples
    plt.subplot(2,4,i+1)
    plt.imshow(X[fav_sample*(i+1)])
    img_title = 'Mnist sample ' + str(fav_sample*(i+1))
    plt.title(img_title, fontsize=12)
plt.show()
plt.clf()
    
##  Is the dataset well balanced? ##

proportion = []
for i in range(0,10):
    proportion.append( np.count_nonzero(y == i)/len(y) ) #the proportion of data labelled i in MNIST
plt.plot(range(0,10), proportion, color='blue', label="Frequency")
plt.plot(range(0,10), 10*[0.1], color='red', linestyle='--', label="Theoretical value: 0.1")
plt.title("Frequency of each digit in the whole MNIST dataset")
plt.xlabel("Digits")
plt.xticks(range(0, 10, 1))
plt.ylabel("Frequency of the digit")
plt.legend()
plt.show()
plt.clf()

mean = np.mean(proportion)
sd = np.std(proportion)
print("Mean:", mean, "Standard deviation:", sd)

##  Split the dataset in one train set and one test set ##

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #20% test set, 80% training set

##  Perform a Principal Component Analysis (PCA) with sklearn ##

nsamples, dimx, dimy = X.shape
d2_X = X.reshape((nsamples, dimx*dimy)) #flatten the pictures from 28x28 matrix to 784 array
pca = PCA(n_components=3)
pca.fit(d2_X)

## Try different n_components. ##

pca = PCA(n_components=dimx*dimy) #all the components
components = [5, 20, 50, 100]
pca.fit(d2_X)
for i in range(0, len(components)):
    comp = components[i]
    plt.bar(range(1, comp+1), height = pca.explained_variance_ratio_[0:comp])
    plt.title('PCA with ' + str(comp) + ' components', fontsize=12)
    plt.xlabel("Index of the component", fontsize=10)
    plt.ylabel("Variance ratio explained")
    plt.show()
plt.clf()
for i in components:
    print("Variation explained by the first " + str(i) + " components: " + str(sum(pca.explained_variance_ratio_[0:i])))

## Try to display some MNIST pictures with different n_components. ##

components = [5, 20, 50, 100, 200]
for index in range(0, len(components)):
    i = components[index]
    pca = PCA(n_components=i)
    reduced_d2_X = pca.fit_transform(d2_X) #fits the PCA
    approx_d2_X = pca.inverse_transform(reduced_d2_X) #reconstructs the data
    approx_X = approx_d2_X.reshape(nsamples, dimx, dimy)
    plt.subplot(2, 3, index+1)
    img_title = 'Mnist sample ' + str(fav_sample) + ' with various number of PCA components'
    plt.suptitle(img_title)
    plt.imshow(approx_X[fav_sample])
    plt.title(str(i) + ' components')
plt.subplot(2, 3, len(components)+1)
plt.imshow(X[fav_sample])
plt.title('Original sample')
plt.show()
plt.clf()

## Use the values of PCA.explained_variance_ratio_ to fit n_components ##

pca = PCA(n_components=dimx*dimy)
pca.fit(d2_X)
cumulative_var = dimx*dimy*[None]
cumulative_var[0] = pca.explained_variance_ratio_[0]
for i in range(1, dimx*dimy): #computes the cumulated sum in pca.explained_variance_ratio_
    cumulative_var[i] = cumulative_var[i-1] + pca.explained_variance_ratio_[i]
plt.plot(cumulative_var)
plt.title('Cumulative variation explained against principal components')
plt.xlabel("Principal components")
plt.ylabel("Cumulative variance ratio explained")
plt.axhline(0.9, 0, dimx*dimy, color='y', linestyle='--', label='90% of variation')
plt.axhline(0.95, 0, dimx*dimy, color='orange', linestyle='--', label='95% of variation')
plt.axhline(0.99, 0, dimx*dimy, color='r', linestyle='--', label='99% of variation')
plt.legend()
plt.show()
plt.clf()

pca = PCA(0.95)
pca.fit(d2_X)
index95 = pca.n_components_
print("Number of components required to keep 95% of the information:", index95)

## With sklearn, perform K-MEANS. Play with the parameter K as well as the initialization (KMEANS++, random, or fixed array). ##

#reshaping test and train set
nsamples_test, dimx, dimy = X_test.shape
d2_X_test = X_test.reshape((nsamples_test, dimx*dimy))
nsamples_train, dimx, dimy = X_train.shape
d2_X_train = X_train.reshape((nsamples_train, dimx*dimy))

# just display the code for Kmeans and explain parameters

## For the correct K, evaluate how good is this partition (with the knowledge of y) ##

ncluster = 10
kmeans = KMeans(n_clusters=ncluster, init='k-means++').fit(d2_X_train)
y_pred = kmeans.predict(d2_X_test)
print("Evaluation of the clustering (K-Means, all information):")
print("Homoegneity within clusters:", homogeneity_score(y_test, y_pred) )
print("Completeness score:", completeness_score(y_test, y_pred) )
print("V-measure: ", v_measure_score(y_test, y_pred) )

res_model.append("K-means")
res_param.append("10 clusters \nAll data")
res_train_acc.append("-")
res_valid_acc.append("-")
res_test_acc.append(v_measure_score(y_test, y_pred))

## Apply K-MEANS with K=10 and PCA::n_components = 2. Display the partition and comment ##

pca = PCA(n_components=2)
reduced_d2_X_train = pca.fit_transform(d2_X_train)
reduced_d2_X_test = pca.transform(d2_X_test)
kmeans = KMeans(n_clusters=10, init='k-means++').fit(reduced_d2_X_train)
y_pred = kmeans.predict(reduced_d2_X_test)
print("Evaluation of the clustering (K-Means, 2 principal components):")
print("Homoegneity within clusters:", homogeneity_score(y_test, y_pred) )
print("Completeness score:", completeness_score(y_test, y_pred) )
print("V-measure: ", v_measure_score(y_test, y_pred) )

plt.suptitle('KMeans in 2 dimensions compared with ground truth')
plt.subplot(1,2,1)
plt.title('KMeans with 2 principle components')
for i in range(0,10):
    lbl = 'Class ' + str(i)
    plt.scatter(reduced_d2_X_test[(y_pred==i),0], reduced_d2_X_test[(y_pred==i),1], s=1, label=lbl)
plt.legend(prop={'size': 8})
plt.subplot(1,2,2)
plt.title('Ground truth')
for i in range(0,10):
    lbl = 'Class ' + str(i)
    plt.scatter(reduced_d2_X_test[(y_test==i),0], reduced_d2_X_test[(y_test==i),1], s=1, label=lbl)
plt.legend(prop={'size': 8})
plt.show()

res_model.append("K-means")
res_param.append("10 clusters \n2 principle components")
res_train_acc.append("-")
res_valid_acc.append("-")
res_test_acc.append(v_measure_score(y_test, y_pred))

## Do the same job with the EM-clustering using the good K parameter (10 for MNIST). Comment your results. ##

emgm = mixture.GaussianMixture(n_components=10,covariance_type='full').fit(reduced_d2_X_train)
y_pred = emgm.predict(reduced_d2_X_test)
print("Evaluation of the clustering (EMGM, 2 principal components):")
print("Homoegneity within clusters:", homogeneity_score(y_test, y_pred) )
print("Completeness score:", completeness_score(y_test, y_pred) )
print("V-measure: ", v_measure_score(y_test, y_pred) )

plt.figure(figsize=(10,5))
plt.suptitle('Expectation-Maximization with Gaussian Mixture in 2 dimensions compared with ground truth')
plt.subplot(1,2,1)
plt.title('EMGM with 2 principle components')
for i in range(0,10):
    lbl = 'Class ' + str(i)
    plt.scatter(reduced_d2_X_test[(y_pred==i),0], reduced_d2_X_test[(y_pred==i),1], s=1, label=lbl)
plt.legend(prop={'size': 8})
plt.subplot(1,2,2)
plt.title('Ground truth')
for i in range(0,10):
    lbl = 'Class ' + str(i)
    plt.scatter(reduced_d2_X_test[(y_test==i),0], reduced_d2_X_test[(y_test==i),1], s=1, label=lbl)
plt.legend(prop={'size': 8})
plt.show()

res_model.append("EM Gaussian Mix")
res_param.append("10 clusters \n2 principle components")
res_train_acc.append("-")
res_valid_acc.append("-")
res_test_acc.append(v_measure_score(y_test, y_pred))

## Decision tree ##

tree1 = DecisionTreeClassifier()
tree1.fit(d2_X_train, y_train)
print("Model tree classifier: unrestricted model")
print("Depth: ", tree1.get_depth(), ", Number of leaves: ", tree1.get_n_leaves())
print("Accuracy on the training set: ", tree1.score(d2_X_train, y_train))
print("Accuracy on the test set: ", tree1.score(d2_X_test, y_test))

res_model.append("Decision tree")
res_param.append("Unrestricted max depth \nAll information")
res_train_acc.append(tree1.score(d2_X_train, y_train))
res_valid_acc.append("-")
res_test_acc.append(tree1.score(d2_X_test, y_test))

depths = range(5, 21) #evaluation for each depth of tree
nleaves = len(depths)*[None]
accuracies_train = len(depths)*[None]
accuracies_test = len(depths)*[None]
for depth in depths:
    tree2 = DecisionTreeClassifier(max_depth=depth)
    model = tree2.fit(d2_X_train, y_train)
    nleaves[depth-depths[0]] = tree2.get_n_leaves()
    accuracies_train[depth-depths[0]] = tree2.score(d2_X_train, y_train)
    accuracies_test[depth-depths[0]] = tree2.score(d2_X_test, y_test)
    
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Number of leaves plotted against max depth')
plt.plot(depths, np.log2(nleaves), label='restricted model', color='b')
plt.axhline(np.log2(tree1.get_n_leaves()), 0, depths[-1], color='darkblue', linestyle='--', label='log2 of n_leaves for the unrestricted model')
plt.xlabel("Depth")
plt.ylabel("Log base 2 of the number of leaves")
plt.legend()
plt.subplot(1,2,2)
plt.title('Accuracy plotted against max depth')
plt.plot(depths, accuracies_train, label='Accuracy on train set, restricted model', color='r')
plt.axhline(tree1.score(d2_X_train, y_train), 0, depths[-1], color='darkred', linestyle='--', label='Accuracy on train set, unrestricted model')
plt.plot(depths, accuracies_test, label='Accuracy on test set, restricted model', color='b')
plt.axhline(tree1.score(d2_X_test, y_test), 0, depths[-1], color='darkblue', linestyle='--', label='Accuracy on test set, unrestricted model')
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.suptitle("Comparison of several depths of tree")
plt.show()

res_model.append("Decision tree")
res_param.append("Max depth: 15 \nAll information")
res_train_acc.append(accuracies_train[10])
res_valid_acc.append("-")
res_test_acc.append(accuracies_test[10])

## Support Vector Machine ##

nbfit = 20 #number of svm to be fitted, must have nbfit%4 = 0
kernels = nbfit//4*['rbf'] + nbfit//4*['linear'] + nbfit//4*['poly'] + nbfit//4*['sigmoid']#the first batch of svm will be gaussian kernel, the second batch will be linear kernel, then polynomial and sigmoid
Cs = nbfit//4*[0.1, 0.5, 1, 2] #we test the same Cs with the two types of kernel
accuracies = nbfit*[None] #table of accuracies
pca = PCA(n_components=index95) #PCA to reduce dimensionality
reduced_d2_X_train = pca.fit_transform(d2_X_train)
reduced_d2_X_test = pca.transform(d2_X_test)
nsamples = reduced_d2_X_train.shape[0] #number of samples in the total train set
for k in range(0, nbfit):
    cur_svm = svm.SVC(kernel=kernels[k], C=Cs[k], gamma='auto') #selects the parameters of the current SVM
    indexes_train = list( range(k*nsamples//nbfit, (k+1)*nsamples//nbfit) ) #indexes used to train the current SVM
    indexes_valid = list( range(0, k*nsamples//nbfit) ) + list( range((k+1)*nsamples//nbfit, nsamples) ) #indexes to valid the current SVM, all the train indexes except those used to train the current SVM
    reduced_d2_X_train_svm = reduced_d2_X_train[indexes_train]
    reduced_d2_X_valid_svm = reduced_d2_X_train[indexes_valid]
    y_train_svm = y_train[indexes_train]
    y_valid_svm = y_train[indexes_valid]
    cur_svm.fit(reduced_d2_X_train_svm, y_train_svm) #training of the current svm
    print("SVM fitted: ", k+1, "/", nbfit)
    accuracies[k] = cur_svm.score(reduced_d2_X_valid_svm, y_valid_svm) #accuracy of the current SVM on its validation set 
    
index_bestSVM = np.argmax(accuracies)
print("Best SVM: kernel ", kernels[index_bestSVM], ", C=", Cs[index_bestSVM], ", Accuracy on validation set: ", accuracies[index_bestSVM] )
bestSVM = svm.SVC(kernel=kernels[index_bestSVM], C=Cs[index_bestSVM], gamma='auto')
indexes_train = list( range(index_bestSVM*nsamples//nbfit, (index_bestSVM+1)*nsamples//nbfit) )
reduced_d2_X_train_svm = reduced_d2_X_train[indexes_train]
y_train_svm = y_train[indexes_train]
bestSVM.fit(reduced_d2_X_train_svm, y_train_svm)
print("Accuracy on test set: ", bestSVM.score(reduced_d2_X_test, y_test))

res_model.append("SVM")
res_param_string = "K-cross validation: 20 folds \n154 principle components \nkernel" + str(kernels[index_bestSVM]) + ", C=" + str(Cs[index_bestSVM])
res_param.append(res_param_string)
res_train_acc.append("-")
res_valid_acc.append(accuracies[index_bestSVM])
res_test_acc.append(bestSVM.score(reduced_d2_X_test, y_test))

## Logistic Regression ##

lr2 = LogisticRegression(penalty='l2', dual=False, C=1, solver='lbfgs', multi_class='auto', max_iter=500)
lr2.fit(d2_X_train, y_train)
lr1 = LogisticRegression(penalty='l1', dual=False, C=1, solver='liblinear', multi_class='auto', max_iter=500)
lr1.fit(d2_X_train, y_train)

print("Logistic regression (penalty L1):")
print("Accuracy on train set: ", lr1.score(d2_X_train, y_train),"Accuracy on test set: ", lr1.score(d2_X_test, y_test))
print("Logistic regression (penalty L2):")
print("Accuracy on train set: ", lr2.score(d2_X_train, y_train),"Accuracy on test set: ", lr2.score(d2_X_test, y_test))

res_model.append("Logistic regression")
res_param.append("Penalty L1 \nAll information")
res_train_acc.append(lr1.score(d2_X_train, y_train))
res_valid_acc.append("-")
res_test_acc.append(lr1.score(d2_X_test, y_test))
res_model.append("Logistic regression")
res_param.append("Penalty L2 \nAll information")
res_train_acc.append(lr2.score(d2_X_train, y_train))
res_valid_acc.append("-")
res_test_acc.append(lr2.score(d2_X_test, y_test))

## Naive Bayes Classifier ##

naivebayes = GaussianNB()
naivebayes.fit(d2_X_train, y_train)
print("Naive Bayes Classifier: ")
print("Accuracy on train set: ", naivebayes.score(d2_X_train, y_train))
print("Accuracy on test set: ", naivebayes.score(d2_X_test, y_test))

res_model.append("Gaussian Naive Bayes")
res_param.append("-")
res_train_acc.append(naivebayes.score(d2_X_train, y_train))
res_valid_acc.append("-")
res_test_acc.append(naivebayes.score(d2_X_test, y_test))

## PCA on decision tree ##

info = [0.66, 0.8, 0.9, 0.95]
depths = len(info)*[None]
nleaves = len(info)*[None]
accuracies_train = len(info)*[None]
accuracies_test = len(info)*[None]

for index in range(0, len(info)):
    elem = info[index]
    pca = PCA(elem)
    reduced_d2_X_train = pca.fit_transform(d2_X_train)
    reduced_d2_X_test = pca.transform(d2_X_test)
    tree = DecisionTreeClassifier()
    model = tree.fit(reduced_d2_X_train, y_train)
    depths[index] = tree.get_depth()
    nleaves[index] = tree.get_n_leaves()
    accuracies_train[index] = tree.score(reduced_d2_X_train, y_train)
    accuracies_test[index] = tree.score(reduced_d2_X_test, y_test)
    
t = PrettyTable()
t.add_column("% information", info)
t.add_column("Tree depth", depths)
t.add_column("number of leaves", nleaves)
t.add_column("Accuracy on train set", accuracies_train)
t.add_column("Accuracy on test set", accuracies_test)
print(t)

res_model.append("Decision tree")
res_param.append("Unrestricted max depth \n66% of information")
res_train_acc.append(accuracies_train[0])
res_valid_acc.append("-")
res_test_acc.append(accuracies_test[0])

## Reshaping dataset before neural networks ##

#transform values 0-255 -> 0.0-1.0
d2_X_train = d2_X_train.astype("float32") / 255
d2_X_test = d2_X_test.astype("float32") / 255
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

## MLP: first model, underfitting ##

epochs = 10

inputs = keras.Input(shape=(d2_X_train.shape[1], ))
x = Dense(32, activation='relu') (inputs)
outputs = Dense(10, activation='softmax') (x)
mlp = Model(inputs, outputs, name="MLP0")

mlp.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
mlp_hist = mlp.fit(d2_X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
mlp_score = mlp.evaluate(d2_X_test, y_test, verbose=0)

mlp.summary()
print("Activation function: ReLu (Softmax for output), number of epochs:", epochs)
plt.figure(figsize=(9, 4.5))
plt.subplot(1,2,1)
plt.suptitle("History of the trainning for MLP0")
plt.plot(mlp_hist.epoch, mlp_hist.history['accuracy'], 'b--', label='Train')
plt.plot(mlp_hist.epoch, mlp_hist.history['val_accuracy'], 'b', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy on train and validation set')
plt.legend()
plt.subplot(1,2,2)
plt.plot(mlp_hist.epoch, mlp_hist.history['loss'], 'b--', label='Train')
plt.plot(mlp_hist.epoch, mlp_hist.history['val_loss'], 'b', label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss on train and validation set')
plt.legend()
plt.show()
print(mlp.name,": Test loss:", mlp_score[0], "Test accuracy:", mlp_score[1])

res_model.append("MLP")
res_param_string = "1 hidden layer," + str(mlp.count_params()) + " parameters \n" + str(mlp_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(mlp_hist.history['accuracy'][-1])
res_valid_acc.append(mlp_hist.history['val_accuracy'][-1])
res_test_acc.append(mlp_score[1])

## MLP: overfitting ##

epochs = [0, 50, 10, 50] #epoch[0] is not to be used

inputs = keras.Input(shape=(d2_X_train.shape[1], ))
x = Dense(32, activation='relu') (inputs)
outputs = Dense(10, activation='softmax') (x)
mlp1 = Model(inputs, outputs, name="MLP1")

inputs = keras.Input(shape=(d2_X_train.shape[1], ))
x = Dense(128, activation='relu') (inputs)
x = Dense(128, activation='relu') (x)
x = Dense(128, activation='relu') (x)
x = Dense(128, activation='relu') (x)
outputs = Dense(10, activation='softmax') (x)
mlp2 = Model(inputs, outputs, name="MLP2")

inputs = keras.Input(shape=(784,))
x = Dense(128, activation='relu') (inputs)
x = Dense(128, activation='relu') (x)
x = Dense(128, activation='relu') (x)
x = Dense(128, activation='relu') (x)
outputs = Dense(10, activation='softmax') (x)
mlp3 = Model(inputs, outputs, name="MPL3")

mlp1.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
mlp1_hist = mlp1.fit(d2_X_train, y_train, epochs=epochs[1], validation_split=0.2, verbose=0)
mlp1_score = mlp1.evaluate(d2_X_test, y_test, verbose=0)
mlp2.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
mlp2_hist = mlp2.fit(d2_X_train, y_train, epochs=epochs[2], validation_split=0.2, verbose=0)
mlp2_score = mlp2.evaluate(d2_X_test, y_test, verbose=0)
mlp3.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
mlp3_hist = mlp3.fit(d2_X_train, y_train, epochs=epochs[3], validation_split=0.2, verbose=0)
mlp3_score = mlp3.evaluate(d2_X_test, y_test, verbose=0)

plt.figure(figsize=(9, 4.5))
plt.subplot(1,2,1)
plt.suptitle("History of the training for MLP1, MLP2, MLP3")
plt.plot(mlp1_hist.epoch, mlp1_hist.history['accuracy'], 'b--', label='MLP1, train')
plt.plot(mlp1_hist.epoch, mlp1_hist.history['val_accuracy'], 'b', label='MLP1 validation')
plt.plot(mlp2_hist.epoch, mlp2_hist.history['accuracy'], 'r--', label='MLP2, train')
plt.plot(mlp2_hist.epoch, mlp2_hist.history['val_accuracy'], 'r', label='MLP2 validation')
plt.plot(mlp3_hist.epoch, mlp3_hist.history['accuracy'], 'g--', label='MLP3, train')
plt.plot(mlp3_hist.epoch, mlp3_hist.history['val_accuracy'], 'g', label='MLP3 validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy on train and validation set')
plt.legend()
plt.subplot(1,2,2)
plt.plot(mlp1_hist.epoch, mlp1_hist.history['loss'], 'b--', label='MLP1, train')
plt.plot(mlp1_hist.epoch, mlp1_hist.history['val_loss'], 'b', label='MLP1 validation')
plt.plot(mlp2_hist.epoch, mlp2_hist.history['loss'], 'r--', label='MLP2, train')
plt.plot(mlp2_hist.epoch, mlp2_hist.history['val_loss'], 'r', label='MLP2 validation')
plt.plot(mlp3_hist.epoch, mlp3_hist.history['loss'], 'g--', label='MLP3, train')
plt.plot(mlp3_hist.epoch, mlp3_hist.history['val_loss'], 'g', label='MLP3 validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss on train and validation set')
plt.legend()
plt.show()
print(mlp1.name,": Test loss:", mlp1_score[0], "Test accuracy:", mlp1_score[1])
print(mlp2.name,": Test loss:", mlp2_score[0], "Test accuracy:", mlp2_score[1])
print(mlp3.name,": Test loss:", mlp3_score[0], "Test accuracy:", mlp3_score[1])

res_model.append("MLP")
res_param_string = "1 hidden layer," + str(mlp1.count_params()) + " parameters \n" + str(mlp1_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(mlp1_hist.history['accuracy'][-1])
res_valid_acc.append(mlp1_hist.history['val_accuracy'][-1])
res_test_acc.append(mlp1_score[1])
res_model.append("MLP")
res_param_string = "4 hidden layers," + str(mlp2.count_params()) + " parameters \n" + str(mlp2_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(mlp2_hist.history['accuracy'][-1])
res_valid_acc.append(mlp2_hist.history['val_accuracy'][-1])
res_test_acc.append(mlp2_score[1])
res_model.append("MLP")
res_param_string = "4 hidden layers," + str(mlp3.count_params()) + " parameters \n" + str(mlp3_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(mlp3_hist.history['accuracy'][-1])
res_valid_acc.append(mlp3_hist.history['val_accuracy'][-1])
res_test_acc.append(mlp3_score[1])

## MLP: best model ##

epochs = 20

inputs = keras.Input(shape=(d2_X_train.shape[1], ))
x = Dense(128, activation='relu') (inputs)
x = Dense(64, activation='relu') (x)
x = Dense(32, activation='relu') (x)
outputs = Dense(10, activation='softmax') (x)
mlp4 = Model(inputs, outputs, name="MLP4")

inputs = keras.Input(shape=(d2_X_train.shape[1], ))
x = Dense(128, activation='relu') (inputs)
x = Dropout(0.15) (x)
x = Dense(64, activation='relu') (x)
x = Dropout(0.15) (x)
x = Dense(32, activation='relu') (x)
outputs = Dense(10, activation='softmax') (x)
mlp5 = Model(inputs, outputs, name="MLP5")

mlp4.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
mlp4_hist = mlp4.fit(d2_X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
mlp4_score = mlp4.evaluate(d2_X_test, y_test, verbose=0)
mlp5.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
mlp5_hist = mlp5.fit(d2_X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
mlp5_score = mlp5.evaluate(d2_X_test, y_test, verbose=0)

mlp4.summary()

plt.figure(figsize=(9, 4.5))
plt.subplot(1,2,1)
plt.suptitle("History of the trainning for MLP4 and MLP5")
plt.plot(mlp4_hist.epoch, mlp4_hist.history['accuracy'], 'b--', label='MLP4 Train')
plt.plot(mlp4_hist.epoch, mlp4_hist.history['val_accuracy'], 'b', label='MLP4 Validation')
plt.plot(mlp5_hist.epoch, mlp5_hist.history['accuracy'], 'r--', label='MLP5 Train')
plt.plot(mlp5_hist.epoch, mlp5_hist.history['val_accuracy'], 'r', label='MLP5 Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy on train and validation set')
plt.legend()
plt.subplot(1,2,2)
plt.plot(mlp4_hist.epoch, mlp4_hist.history['loss'], 'b--', label='MLP4 Train')
plt.plot(mlp4_hist.epoch, mlp4_hist.history['val_loss'], 'b', label='MLP4 Validation')
plt.plot(mlp5_hist.epoch, mlp5_hist.history['loss'], 'r--', label='MLP5 Train')
plt.plot(mlp5_hist.epoch, mlp5_hist.history['val_loss'], 'r', label='MLP5 Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss on train and validation set')
plt.legend()
plt.show()
print(mlp4.name,": Test loss:", mlp4_score[0], "Test accuracy:", mlp4_score[1])
print(mlp5.name,": Test loss:", mlp5_score[0], "Test accuracy:", mlp5_score[1])

res_model.append("MLP")
res_param_string = "3 hidden layers," + str(mlp4.count_params()) + " parameters \n" + str(mlp4_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(mlp4_hist.history['accuracy'][-1])
res_valid_acc.append(mlp4_hist.history['val_accuracy'][-1])
res_test_acc.append(mlp4_score[1])
res_model.append("MLP")
res_param_string = "3 hidden layers," + str(mlp5.count_params()) + " parameters \n" + str(mlp5_hist.epoch[-1]+1) + " epochs, Dropout 15%"
res_param.append(res_param_string)
res_train_acc.append(mlp5_hist.history['accuracy'][-1])
res_valid_acc.append(mlp5_hist.history['val_accuracy'][-1])
res_test_acc.append(mlp5_score[1])

## CNN: first model ##

epochs = 15

inputs = keras.Input(shape=(28, 28, 1))
x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation='relu', input_shape=(32,28,28,1)) (inputs)
x = Flatten() (x)
x = Dense(64, activation='relu') (x)
outputs = Dense(10, activation='softmax') (x)
cnn = Model(inputs, outputs, name="CNN0")

cnn.summary()

cnn.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
cnn_hist = cnn.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
cnn_score = cnn.evaluate(X_test, y_test, verbose=0)

plt.figure(figsize=(9, 4.5))
plt.subplot(1,2,1)
plt.suptitle("History of the trainning for CNN0")
plt.plot(cnn_hist.epoch, cnn_hist.history['accuracy'], 'b--', label='train')
plt.plot(cnn_hist.epoch, cnn_hist.history['val_accuracy'], 'b', label='validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy on train and validation set')
plt.legend()
plt.subplot(1,2,2)
plt.plot(cnn_hist.epoch, cnn_hist.history['loss'], 'b--', label='train')
plt.plot(cnn_hist.epoch, cnn_hist.history['val_loss'], 'b', label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss on train and validation set')
plt.legend()
plt.show()
print("CNN: Test loss:", cnn_score[0], "Test accuracy:", cnn_score[1])

res_model.append("CNN")
res_param_string = "1 convolutionnal layer," + str(cnn.count_params()) + " parameters \n" + str(cnn_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(cnn_hist.history['accuracy'][-1])
res_valid_acc.append(cnn_hist.history['val_accuracy'][-1])
res_test_acc.append(cnn_score[1])

## CNN: best model ##

epochs = 15

inputs = keras.Input(shape=(28, 28, 1))
x = Conv2D(filters=32, kernel_size=3, strides=1, padding="valid", activation='relu', input_shape=(32,28,28,1)) (inputs)
x = MaxPooling2D(2) (x)
x = Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation='relu', input_shape=(32,28,28,1)) (x)
x = MaxPooling2D(2) (x)
x = Conv2D(filters=32, kernel_size=3, strides=1, padding="valid", activation='relu', input_shape=(32,28,28,1)) (x)
x = Flatten() (x) #11*11*64
x = Dense(64, activation='relu') (x)
x = Dropout(0.1) (x)
outputs = Dense(10, activation='softmax') (x)
cnn = Model(inputs, outputs, name="CNN1")

cnn.summary()

cnn.compile( loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"], )
cnn_hist = cnn.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
cnn_score = cnn.evaluate(X_test, y_test, verbose=0)

plt.figure(figsize=(9, 4.5))
plt.subplot(1,2,1)
plt.suptitle("History of the trainning for CNN1")
plt.plot(cnn_hist.epoch, cnn_hist.history['accuracy'], 'b--', label='train')
plt.plot(cnn_hist.epoch, cnn_hist.history['val_accuracy'], 'b', label='validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy on train and validation set')
plt.legend()
plt.subplot(1,2,2)
plt.plot(cnn_hist.epoch, cnn_hist.history['loss'], 'b--', label='train')
plt.plot(cnn_hist.epoch, cnn_hist.history['val_loss'], 'b', label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss on train and validation set')
plt.legend()
plt.show()
print("CNN: Test loss:", cnn_score[0], "Test accuracy:", cnn_score[1])

res_model.append("CNN")
res_param_string = "3 convolutionnal layers," + str(cnn.count_params()) + " parameters \n" + str(cnn_hist.epoch[-1]+1) + " epochs"
res_param.append(res_param_string)
res_train_acc.append(cnn_hist.history['accuracy'][-1])
res_valid_acc.append(cnn_hist.history['val_accuracy'][-1])
res_test_acc.append(cnn_score[1])

## Conclusion ##

res = PrettyTable()
res.add_column("Model", res_model)
res.add_column("Parameters", res_param)
res.add_column("Accuracy (train)", res_train_acc)
res.add_column("Accuracy (validation)", res_valid_acc)
res.add_column("Accuracy (test)", res_test_acc)
print(res)