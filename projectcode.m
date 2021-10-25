activity_labels = {'Climbingdown','Climbingup','Jumping','Lying','Running','Sitting','Standing','Walking'};
uiopen('C:\Users\turhal\Desktop\data\y_train.xlsx',1);
trainActivity = categorical(ytrain,1:8,activity_labels);
trainActivity = mergecats(trainActivity,{'Climbingdown','Climbingup'},'ClimbingStairs');
trainActivity = reordercats(trainActivity ,{'ClimbingStairs','Jumping','Lying','Running','Sitting','Standing','Walking'});
uiopen('C:\Users\turhal\Desktop\data\waist_acc_x_128\totalwaistaccx\waist_total_acc_x_.csv',1)
uiopen('C:\Users\turhal\Desktop\data\waist_acc_y_128\totalwaistaccy\waist_total_acc_y.csv',1)
uiopen('C:\Users\turhal\Desktop\data\waist_acc_z_128\totalwaistaccz\waist_total_acc_z.csv',1)
uiopen('C:\Users\turhal\Desktop\data\waist_gyr_x_128\totalwaistgyrx\waist_total_gyr_x_.csv',1)
uiopen('C:\Users\turhal\Desktop\data\waist_gyr_y_128\totalgyry\waist_total_gyr_y_.csv',1)
uiopen('C:\Users\turhal\Desktop\data\waist_gyr_z_128\totalgyrz\waist_total_gyr_z_.csv',1)
plotRawSensorData(waisttotalaccx, waisttotalaccy, waisttotalaccz,trainActivity,1000);
rawSensorDataTrain = table(waisttotalaccx, waisttotalaccy, waisttotalaccz, waisttotalgyrx, waisttotalgyry, waisttotalgyrz);
humanActivityData = varfun(@Wmean,rawSensorDataTrain);
humanActivityData.activity = trainActivity;
classificationLearner
T_mean = varfun(@Wmean, rawSensorDataTrain);
T_stdv = varfun(@Wstd,rawSensorDataTrain);
T_pca  = varfun(@Wpca1,rawSensorDataTrain);
T_median = varfun(@Wmedian,rawSensorDataTrain);
T_mode = varfun(@Wmode,rawSensorDataTrain);
T_var = varfun(@Wvar,rawSensorDataTrain);
T_iqr = varfun(@Wiqr,rawSensorDataTrain);
T_kurt = varfun(@Wkurt,rawSensorDataTrain);
T_meandev = varfun(@Wmeandev,rawSensorDataTrain);
T_fft = varfun(@Wfft,rawSensorDataTrain);
humanActivityData = [T_mean, T_stdv, T_median, T_mode, T_var, T_iqr, T_kurt, T_meandev];
humanActivityData.activity = trainActivity;
classificationLearner
