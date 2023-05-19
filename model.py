import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV as calib_clf
from sklearn.calibration import calibration_curve 


#Unused.  Here for reference.
def generateWinModel(winModelDB):


    #winModelDB = prepDFforModel(winModelDB)



    Y = winModelDB['Result'].values
    Y = Y.astype('int')

    fighterList_df = winModelDB['Fighter']

    fighterList_df = fighterList_df.drop_duplicates()
    fighterList_df  = fighterList_df.sort_values()

    X = winModelDB.drop(labels=['Result', 'Fighter'], axis=1)

    print(X)

    testSize = 0.2
    trainSize = 0.4
    valSize = 0.4



    #default:
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    #x, x_test, y, y_test = train_test_split(xtrain,labels,test_size=0.2,train_size=0.8)
    #x_train, x_cv, y_train, y_cv = train_test_split(x,y,test_size = 0.25,train_size =0.75)

#random state here
    x, X_test, y, Y_test = train_test_split(X, Y,test_size=testSize,train_size=trainSize + valSize, random_state=1234)
    #using X and Y (from earlier), to create 2 sets of data. 
    #X_test, Y_test are test group.
    #x and y (lowercase) are "train" (which we will split into two separate groups.)

#random state here
    X_train, X_val, Y_train, Y_val = train_test_split(x,y,test_size = 0.5, random_state=12345)
    #x and y make up trainSize and valSize

    #Splitting "x" and "y" from the above split into two.
    #The first group will be our *true* train group, X_train, Y_train.
    #The second group will be our validation group, x_cv, and y_cv ("test" group)
    #



    #X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state = 20)

#random state here
    model = RandomForestClassifier(n_estimators = 10, random_state=123456)


    model.fit(X_train, Y_train)

    prediction_test = model.predict(X_test)

    print("No calibration accuracy = ", metrics.accuracy_score (Y_test, prediction_test))

    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)

    print(feature_imp)
    print('hi')

    #model.predict_proba(X)
    np.max(model.predict_proba(X), axis=1)


    #Calibration curve takes the TRUE TEST response variable (Y) for the dataset, as well as the PREDICTED (Y)
    #using our model, for the TEST Y. (So use X_TEST to predict Y_TEST)

    #additonally, the predicted Y values (below, Y_pred) must be in an array [:,1]


    #*calibrated* but uses the same data as the model... so not right.
    Y_pred = model.predict_proba(X_test)[:,1]
    fop, mpv = calibration_curve(Y_test, Y_pred, n_bins=10)



    #a, b = calibration_curve(y_cv, Y_pred, n_bins=10)



    #NOW, creating a *calibrator* based off our model.
    #need to figure out what to use for cv.
    calibrator1 = calib_clf(model, cv='prefit')
    calibrator2 = calib_clf(model, method = "sigmoid", cv ='prefit')
    calibrator3 = calib_clf(model, method = "isotonic", cv ='prefit')



    #fit calibrator to hold out / validation data.

    calibrator1.fit(X_val, Y_val)
    calibrator2.fit(X_val, Y_val)
    calibrator3.fit(X_val, Y_val)

    yhat1 = calibrator1.predict(X_test)
    yhat2 = calibrator2.predict(X_test)
    yhat3 = calibrator3.predict(X_test)

    #print(yhat)

    #FIGURE OUT WHICH CV TO USE.
    #ALMOST HTER BRO

    fop_calibrated1, mpv_calibrated1 = calibration_curve(Y_test, yhat1, n_bins=10)
    fop_calibrated2, mpv_calibrated2 = calibration_curve(Y_test, yhat2, n_bins=10)
    fop_calibrated3, mpv_calibrated3 = calibration_curve(Y_test, yhat3, n_bins=10)

    #PLOTTING CALIBRATION CURVE.
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(mpv, fop, marker='.')
    plt.plot(mpv_calibrated1, fop_calibrated1, marker='.')
    plt.plot(mpv_calibrated2, fop_calibrated2, marker='.')
    plt.plot(mpv_calibrated3, fop_calibrated3, marker='.')
    plt.show()
    print('model')
    print(model.predict_proba(X_test))

    print('c1')
    print(calibrator1.predict_proba(X_test))


    print('c2')
    print(calibrator2.predict_proba(X_test))


    print('c3')
    print(calibrator3.predict_proba(X_test))

    print("C1 accuracy = ", metrics.accuracy_score (Y_test, yhat1))
    print("C2 accuracy = ", metrics.accuracy_score (Y_test, yhat2))
    print("C3 accuracy = ", metrics.accuracy_score (Y_test, yhat3))

    return model, fighterList_df



#Unused.  Here for reference.
def generateKOmodel(KOmodelDBdf):

    Y = KOmodelDBdf['KO GIVEN'].values
    Y = Y.astype('int')

    X = KOmodelDBdf.drop(labels=['Result', 'Fighter', 'KO GIVEN'], axis=1)

    print(X)

    testSize = 0.3
    trainSize = 0.7

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=testSize,train_size=trainSize, random_state=1234)
   
    model = RandomForestClassifier(n_estimators = 10, random_state=123456)


    model.fit(X_train, Y_train)

    prediction_test = model.predict(X_test)

    print("KO MODEL ACCURACY = ", metrics.accuracy_score (Y_test, prediction_test))

    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)

    print(feature_imp)

    np.max(model.predict_proba(X), axis=1)
    Y_pred = model.predict_proba(X_test)[:,1]

    print('model')
    print(model.predict_proba(X_test))

    return model


#Unused.  Here for reference.
def generateSUBmodel(SUBmodelDBdf):

    Y = SUBmodelDBdf['SUB GIVEN'].values
    Y = Y.astype('int')

    X = SUBmodelDBdf.drop(labels=['Result', 'Fighter', 'KO GIVEN'], axis=1)

    print(X)

    testSize = 0.3
    trainSize = 0.7

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=testSize,train_size=trainSize, random_state=1234)
   
    model = RandomForestClassifier(n_estimators = 10, random_state=123456)


    model.fit(X_train, Y_train)

    prediction_test = model.predict(X_test)

    print("KO MODEL ACCURACY = ", metrics.accuracy_score (Y_test, prediction_test))

    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)

    print(feature_imp)

    np.max(model.predict_proba(X), axis=1)
    Y_pred = model.predict_proba(X_test)[:,1]

    print('model')
    print(model.predict_proba(X_test))

    return model



#modelDB = fight database AFTER scrape, advanced stats calc, and pruning. (after createModelDBdf()).
#YcolumnName = which column we want to use for our predicted variable (differs for KO, SUB, WIN)

def generateModel(modelDB, YcolumnName, testsize, trainsize, seed1, seed2):
#Model uses random forests.


    Y = modelDB[YcolumnName].values
    Y = Y.astype('int')

    X = modelDB.drop(labels=['Result', 'Fighter', 'KO GIVEN', 'SUB GIVEN'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=testsize,train_size=trainsize, random_state=seed1)


    model = RandomForestClassifier(n_estimators = 100, random_state=seed2)

    model.fit(X_train, Y_train)

    prediction_test = model.predict(X_test)

    print(YcolumnName + " prediction accuracy: ", metrics.accuracy_score (Y_test, prediction_test))
    

    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_, index = feature_list).sort_values(ascending = False)

    print(feature_imp)

    np.max(model.predict_proba(X), axis=1)
    print(model.predict_proba(X_test))



    return model
