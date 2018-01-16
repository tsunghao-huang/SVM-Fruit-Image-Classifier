import random
from sklearn import svm
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from skimage import io, filters
from skimage.transform import resize
from skimage.feature import hog, ORB, CENSURE, corner_peaks, corner_harris, BRIEF
from sklearn.externals import joblib


#enter fruit names here: Choose from apple, banana, pineapple, kiwi
#example:
classes = ['banana', 'pineapple']

def main():
    DataSet = []
    LabelSet = []
    lengthV = []
    trainPaths = ['./fruit/'+c+ '_train/' for c in classes ]
    testPaths =  ['./fruit/'+c+' test/'   for c in classes ]
    
    resList = []
    boolList = []
    pos = 0
    ind = 0
    #if you wish to automatically perform both feature selection optimzation and svm optimization at the same time
    #comment out next line and comment in section above
    #Warning: Very long runtime for algorithm because of grid search
    useList = [True, True, True, True]
    #print(useList)
    
    for c in range(len(classes)):
        #get label for features to be added
        className = classes[c]
        #get file path for folder with images
        path = trainPaths[c]
        #initialize feature detectors/extractors
        #Censure extractor
        detector = CENSURE()
        #ORB extractor
        detector2 = ORB(n_keypoints=50)
        #get all file names from the folder
        files = os.listdir(path)
        nfiles = len(files)
        #repeat for each file
        for i in range(nfiles):
            #initialize feature vector as empty list
            featureVector = []
            infile = files[i]
            #read image as grayscale numpy.ndarray
            img = io.imread(path+infile, as_grey=True)
            #get histogram for grayscale value intensity
            hist = np.histogram(img, bins=256)
            #resize image
            img = resize(img, (400,400))
            #extract features but do not yet add them to feature vector
            detector2.detect_and_extract(img)
            #extract HOG features, add them to featurevector
            a = fd = hog(img, orientations=9, pixels_per_cell=(32, 32),
                    cells_per_block=(1,1), visualise=False)
            #add histogramm to featurevector
            for h in hist:
                fd = np.append(fd, h)
            #if corresponding boolean in uselist is true add features to featureVector --> Feature selection happens here
            if(useList[0]):                            
                detector.detect(img)
                fd = np.append(fd, [np.array(detector.keypoints).flatten()])
            if(useList[1]):
                fd = np.append(fd, detector2.keypoints)
            if(useList[2]):
                fd = np.append(fd, edgeExtract(img, 100))
            if(useList[3]):
                corners =  corner_peaks(corner_harris(img),min_distance=1)
                fd = np.append(fd, corners)
            #get length of featurevector for later operations
            lengthV.append(len(fd))
            #add featureVector list to dataset that is fed into svm
            DataSet.append(fd)
            #get label name
            ind = classes.index(className)
            #add label to label dataset that is fed into svm
            LabelSet.append(ind)
    #get length of biggest sized featurevector
    max = np.amax(lengthV)
    lengthV = []
    DataSet2 = []
    #pad dataset with zeroes so that all featurevectors have the same length --> important for svm
    for d in DataSet:
        d = np.pad(d, (0, max - len(d)), 'constant')
        DataSet2.append(d)
        lengthV.append(len(d))
    DataSet = DataSet2
    #perform a grid search with maximum number of possible threads (usually 4)
    if __name__=='__main__':
        gridSearch(DataSet, LabelSet)
    #train and examine svm with default values for comparison later
    clf = svm.SVC(kernel='rbf', C=10.0, gamma=1.0000000000000001e-09)
    clf.fit(DataSet, LabelSet)
    joblib.dump(clf, classes[0]+' '+ classes[1]+'.pk1')
    scores = cross_val_score(clf, DataSet, LabelSet, cv=10)
    #print results of default svm
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#extract edge histogramm, bins number = 100    
def edgeExtract(img, bins):
    retVal = []
    #apply vertical and horizontal sobel filters to get two histogramms, once of vertical and once of horizontal edges
    #vertical
    fs = filters.sobel_v(img)
    #horizontal
    angs = filters.sobel_h(img)
    #compute histograms
    lhist = np.histogram(fs,bins,normed=True,range=(0,1))
    ahist = np.histogram(angs, bins,normed=True,range=(-180,180))
    #fuse histograms into one list
    retVal.extend(lhist[0].tolist())
    retVal.extend(ahist[0].tolist())
    return retVal
#Perform grid search, 
# if optimum feature selection list is to be found verbalize = false
def gridSearch(DataSet, LabelSet, verbalize = True):
    #define logspace/interval from which c and gamma valuest are computed and saved to a dictionary to be passed as a parameter
    #c from 1e-2 to 1e10
    C_range = np.logspace(-2, 10, 13)
    #gamma from 1e-9 to 1e3
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    #grid input parameter
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    #perform grid search with multiple threads for added performance
    if(verbalize):
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=-1)
    #perform grid search with one thread and return value --> multiple threads do not work with return values
    else:
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=1)
    #find optimal values
    grid.fit(DataSet, LabelSet)
    #print results of test for optimal values
    if(verbalize):
        print("The best parameters are %s with a score of %0.2f" 
            % (grid.best_params_, grid.best_score_))
    else:
        #return best scores for different feature selections
        return grid.best_score_
#same as main function, however a sample of 100 images is drawn for each image
def selectFeatures(useList):
    DataSet = []
    LabelSet = []
    lengthV = []
    trainPaths = ['./fruit/'+c+ '_train/' for c in classes ]
    testPaths =  ['./fruit/'+c+' test/'   for c in classes ]
    for c in range(len(classes)):
        className = classes[c]
        path = trainPaths[c]
        detector = CENSURE()
        detector2 = ORB(n_keypoints=50)
        detector3 = BRIEF(patch_size=49)
        files = os.listdir(path)
        #sample
        files = random.sample(files, 100)
        nfiles = len(files)
        for i in range(nfiles):
            featureVector = []
            infile = files[i]
            img = io.imread(path+infile, as_grey=True)
            hist = np.histogram(img, bins=256)
            img = resize(img, (400,400))
            detector2.detect_and_extract(img)
            detector.detect(img)
            a = fd = hog(img, orientations=9, pixels_per_cell=(32, 32),
                    cells_per_block=(1,1), visualise=False)
            for h in hist:
                fd = np.append(fd, h)
            if(useList[0]):
                fd = np.append(fd, [np.array(detector.keypoints).flatten()])
            if(useList[1]):
                fd = np.append(fd, detector2.keypoints)
            if(useList[2]):
                fd = np.append(fd, edgeExtract(img, 100))
            l1 = len(fd)
            corners =  corner_peaks(corner_harris(img),min_distance=1)
            if(useList[3]):
                fd = np.append(fd, corners)
            lengthV.append(len(fd))  
            DataSet.append(fd)
            ind = classes.index(className)
            LabelSet.append(ind)
    max = np.amax(lengthV)
    lengthV = []
    DataSet2 = []
    for d in DataSet:
        d = np.pad(d, (0, max - len(d)), 'constant')
        DataSet2.append(d)
        lengthV.append(len(d))
    DataSet = DataSet2
    res = 0
    #perform gridsearch with one thread
    if __name__=='__main__':
        res = gridSearch(DataSet, LabelSet, False)
        return res


main()

