import pandas as pd

import behavior.data_parameter as pr
from sklearn.model_selection import train_test_split


path = './initial_data/Cow_man.csv'
ActualData = pd.read_csv(path, usecols=['time','gFx', 'gFy', 'gFz', 'Class'], sep=',', low_memory=False)
time = ActualData['time']
Ax = ActualData['gFx']
Ay = ActualData['gFy']
Az = ActualData['gFz']
Class = ActualData['Class']

size_window = 600

ax_mean = pr.meanpar(Ax, size_window)  # #3600-1 min
ay_mean = pr.meanpar(Ay, size_window)
az_mean = pr.meanpar(Az, size_window)
standard_deviation_x = pr.sd(Ax, size_window)
standard_deviation_y = pr.sd(Ay, size_window)
standard_deviation_z = pr.sd(Az, size_window)
overall_dynamic_body = pr.meanpar(pr.odba(Ax, Ay, Az), size_window)
vectorial_dynamic_body = pr.meanpar(pr.vedba(Ax, Ay, Az), size_window)
quan_01_x = pr.quantiles(Ax, size_window, 0.1)
quan_01_y = pr.quantiles(Ay, size_window, 0.1)
quan_01_z = pr.quantiles(Az, size_window, 0.1)
quan_025_x = pr.quantiles(Ax, size_window, 0.25)
quan_025_y = pr.quantiles(Ay, size_window, 0.25)
quan_025_z = pr.quantiles(Az, size_window, 0.25)
quan_05_x = pr.quantiles(Ax, size_window, 0.5)
quan_05_y = pr.quantiles(Ay, size_window, 0.5)
quan_05_z = pr.quantiles(Az, size_window, 0.5)
quan_075_x = pr.quantiles(Ax, size_window, 0.75)
quan_075_y = pr.quantiles(Ay, size_window, 0.75)
quan_075_z = pr.quantiles(Az, size_window, 0.75)
quan_09_x = pr.quantiles(Ax, size_window, 0.9)
quan_09_y = pr.quantiles(Ay, size_window, 0.9)
quan_09_z = pr.quantiles(Az, size_window, 0.9)
skewness_x = pr.skewness(Ax, size_window)
skewness_y = pr.skewness(Ay, size_window)
skewness_z = pr.skewness(Az, size_window)
kurtosis_x = pr.kurtosis(Ax, size_window)
kurtosis_y = pr.kurtosis(Ay, size_window)
kurtosis_z = pr.kurtosis(Az, size_window)
sum_quantiles_min_01_x = pr.sumQuantilesMinMax(Ax, size_window, 0.1, 1)
sum_quantiles_min_01_y = pr.sumQuantilesMinMax(Ay, size_window, 0.1, 1)
sum_quantiles_min_01_z = pr.sumQuantilesMinMax(Az, size_window, 0.1, 1)
sum_quantiles_min_squares_01_x = pr.sumQuantilesMinMax(Ax, size_window, 0.1, 2)
sum_quantiles_min_squares_01_y = pr.sumQuantilesMinMax(Ay, size_window, 0.1, 2)
sum_quantiles_min_squares_01_z = pr.sumQuantilesMinMax(Az, size_window, 0.1, 2)
sum_quantiles_max_01_x = pr.sumQuantilesMinMax(Ax, size_window, 0.1, 3)
sum_quantiles_max_01_y = pr.sumQuantilesMinMax(Ay, size_window, 0.1, 3)
sum_quantiles_max_01_z = pr.sumQuantilesMinMax(Az, size_window, 0.1, 3)
sum_quantiles_max_squares_01_x = pr.sumQuantilesMinMax(Ax, size_window, 0.1, 4)
sum_quantiles_max_squares_01_y = pr.sumQuantilesMinMax(Ay, size_window, 0.1, 4)
sum_quantiles_max_squares_01_z = pr.sumQuantilesMinMax(Az, size_window, 0.1, 4)
sum_quantiles_min_05_x = pr.sumQuantilesMinMax(Ax, size_window, 0.5, 1)
sum_quantiles_min_05_y = pr.sumQuantilesMinMax(Ay, size_window, 0.5, 1)
sum_quantiles_min_05_z = pr.sumQuantilesMinMax(Az, size_window, 0.5, 1)
sum_quantiles_min_squares_05_x = pr.sumQuantilesMinMax(Ax, size_window, 0.5, 2)
sum_quantiles_min_squares_05_y = pr.sumQuantilesMinMax(Ay, size_window, 0.5, 2)
sum_quantiles_min_squares_05_z = pr.sumQuantilesMinMax(Az, size_window, 0.5, 2)
sum_quantiles_max_05_x = pr.sumQuantilesMinMax(Ax, size_window, 0.5, 3)
sum_quantiles_max_05_y = pr.sumQuantilesMinMax(Ay, size_window, 0.5, 3)
sum_quantiles_max_05_z = pr.sumQuantilesMinMax(Az, size_window, 0.5, 3)
sum_quantiles_max_squares_05_x = pr.sumQuantilesMinMax(Ax, size_window, 0.5, 4)
sum_quantiles_max_squares_05_y = pr.sumQuantilesMinMax(Ay, size_window, 0.5, 4)
sum_quantiles_max_squares_05_z = pr.sumQuantilesMinMax(Az, size_window, 0.5, 4)
sum_quantiles_min_09_x = pr.sumQuantilesMinMax(Ax, size_window, 0.9, 1)
sum_quantiles_min_09_y = pr.sumQuantilesMinMax(Ay, size_window, 0.9, 1)
sum_quantiles_min_09_z = pr.sumQuantilesMinMax(Az, size_window, 0.9, 1)
sum_quantiles_min_squares_09_x = pr.sumQuantilesMinMax(Ax, size_window, 0.9, 2)
sum_quantiles_min_squares_09_y = pr.sumQuantilesMinMax(Ay, size_window, 0.9, 2)
sum_quantiles_min_squares_09_z = pr.sumQuantilesMinMax(Az, size_window, 0.9, 2)
sum_quantiles_max_09_x = pr.sumQuantilesMinMax(Ax, size_window, 0.9, 3)
sum_quantiles_max_09_y = pr.sumQuantilesMinMax(Ay, size_window, 0.9, 3)
sum_quantiles_max_09_z = pr.sumQuantilesMinMax(Az, size_window, 0.9, 3)
sum_quantiles_max_squares_09_x = pr.sumQuantilesMinMax(Ax, size_window, 0.9, 4)
sum_quantiles_max_squares_09_y = pr.sumQuantilesMinMax(Ay, size_window, 0.9, 4)
sum_quantiles_max_squares_09_z = pr.sumQuantilesMinMax(Az, size_window, 0.9, 4)
pairwise_differences_005_01_x = pr.pairwiseDifferences(Ax, size_window, 0.05, 0.1)
pairwise_differences_005_01_y = pr.pairwiseDifferences(Ay, size_window, 0.05, 0.1)
pairwise_differences_005_01_z = pr.pairwiseDifferences(Az, size_window, 0.05, 0.1)
pairwise_differences_01_025_x = pr.pairwiseDifferences(Ax, size_window, 0.1, 0.25)
pairwise_differences_01_025_y = pr.pairwiseDifferences(Ay, size_window, 0.1, 0.25)
pairwise_differences_01_025_z = pr.pairwiseDifferences(Az, size_window, 0.1, 0.25)
pairwise_differences_025_05_x = pr.pairwiseDifferences(Ax, size_window, 0.25, 0.5)
pairwise_differences_025_05_y = pr.pairwiseDifferences(Ay, size_window, 0.25, 0.5)
pairwise_differences_025_05_z = pr.pairwiseDifferences(Az, size_window, 0.25, 0.5)
pairwise_differences_05_075_x = pr.pairwiseDifferences(Ax, size_window, 0.5, 0.75)
pairwise_differences_05_075_y = pr.pairwiseDifferences(Ay, size_window, 0.5, 0.75)
pairwise_differences_05_075_z = pr.pairwiseDifferences(Az, size_window, 0.5, 0.75)
pairwise_differences_075_09_x = pr.pairwiseDifferences(Ax, size_window, 0.75, 0.9)
pairwise_differences_075_09_y = pr.pairwiseDifferences(Ay, size_window, 0.75, 0.9)
pairwise_differences_075_09_z = pr.pairwiseDifferences(Az, size_window, 0.75, 0.9)
average_absolute_difference = pr.AAD(Ax, Ay, Az, size_window)
average_intensity = pr.meanpar(pr.averageIntensity(Ax, Ay, Az), size_window)
signal_magnitude_area = pr.signalMagnitudeArea(Ax, Ay, Az, size_window)
movement_variation = pr.movementVariation(Ax, Ay, Az, size_window)
entropy = pr.entropy(Ax, Ay, Az, size_window)
energy = pr.energy(Ax, Ay, Az, size_window)
activity_x = pr.activity(Ax, size_window)
activity_y = pr.activity(Ay, size_window)
activity_z = pr.activity(Az, size_window)
mobility_x = pr.mobility(Ax, size_window)
mobility_y = pr.mobility(Ay, size_window)
mobility_z = pr.mobility(Az, size_window)
complexity_x = pr.complexity(Ax, size_window)
complexity_y = pr.complexity(Ay, size_window)
complexity_z = pr.complexity(Az, size_window)
durbin_watson_x = pr.dwa(Ax, size_window)
durbin_watson_y = pr.dwa(Ay, size_window)
durbin_watson_z = pr.dwa(Az, size_window)
correlation_magnitude_horizontal = pr.meanpar(pr.correlation(Ax, Ay, Az, 1), size_window)
correlation_magnitude_vertical = pr.meanpar(pr.correlation(Ax, Ay, Az, 2), size_window)
correlation_horizontal_vertical = pr.meanpar(pr.correlation(Ax, Ay, Az, 3), size_window)

time1 = pr.meantime(time, size_window)
activity = pr.meanclass(Class, size_window)

activity_name = []

for i in range(len(activity)):
    if activity[i] == 0:
        activity_name.append('STEPING')
    elif activity[i] == 1:
        activity_name.append('STANDING')
    elif activity[i] == 2:
        activity_name.append('LAYING')
    elif activity[i] == 3:
        activity_name.append('FEEDING')


d={'ax_mean': ax_mean,'ay_mean': ay_mean,'az_mean': az_mean,
   'standard_deviation_x': standard_deviation_x, 'standard_deviation_y': standard_deviation_y, 'standard_deviation_z': standard_deviation_z,
   'overall_dynamic_body': overall_dynamic_body, 'vectorial_dynamic_body': vectorial_dynamic_body,
   'quan_01_x': quan_01_x, 'quan_01_y': quan_01_y, 'quan_01_z': quan_01_z,
   'quan_025_x': quan_025_x, 'quan_025_y': quan_025_y, 'quan_025_z': quan_025_z,
   'quan_05_x': quan_05_x, 'quan_05_y': quan_05_y, 'quan_05_z': quan_05_z,
   'quan_075_x': quan_075_x, 'quan_075_y': quan_075_y, 'quan_075_z': quan_075_z,
   'quan_09_x': quan_09_x, 'quan_09_y': quan_09_y, 'quan_09_z': quan_09_z,
   'skewness_x': skewness_x, 'skewness_y': skewness_y, 'skewness_z': skewness_z,
   'kurtosis_x': kurtosis_x, 'kurtosis_y': kurtosis_y, 'kurtosis_z': kurtosis_z,
   'sum_quantiles_min_01_x': sum_quantiles_min_01_x, 'sum_quantiles_min_01_y': sum_quantiles_min_01_y, 'sum_quantiles_min_01_z': sum_quantiles_min_01_z,
   'sum_quantiles_min_squares_01_x': sum_quantiles_min_squares_01_x,
   'sum_quantiles_min_squares_01_y': sum_quantiles_min_squares_01_y,
   'sum_quantiles_min_squares_01_z': sum_quantiles_min_squares_01_z,
   'sum_quantiles_max_01_x': sum_quantiles_max_01_x, 'sum_quantiles_max_01_y': sum_quantiles_max_01_y, 'sum_quantiles_max_01_z': sum_quantiles_max_01_z,
   'sum_quantiles_max_squares_01_x': sum_quantiles_max_squares_01_x,
   'sum_quantiles_max_squares_01_y': sum_quantiles_max_squares_01_y,
   'sum_quantiles_max_squares_01_z': sum_quantiles_max_squares_01_z,
   'sum_quantiles_min_05_x': sum_quantiles_min_05_x, 'sum_quantiles_min_05_y': sum_quantiles_min_05_y, 'sum_quantiles_min_05_z': sum_quantiles_min_05_z,
   'sum_quantiles_min_squares_05_x': sum_quantiles_min_squares_05_x,
   'sum_quantiles_min_squares_05_y': sum_quantiles_min_squares_05_y,
   'sum_quantiles_min_squares_05_z': sum_quantiles_min_squares_05_z,
   'sum_quantiles_max_05_x': sum_quantiles_max_05_x, 'sum_quantiles_max_05_y': sum_quantiles_max_05_y, 'sum_quantiles_max_05_z': sum_quantiles_max_05_z,
   'sum_quantiles_max_squares_05_x': sum_quantiles_max_squares_05_x,
   'sum_quantiles_max_squares_05_y': sum_quantiles_max_squares_05_y,
   'sum_quantiles_max_squares_05_z': sum_quantiles_max_squares_05_z,
   'sum_quantiles_min_09_x': sum_quantiles_min_09_x, 'sum_quantiles_min_09_y': sum_quantiles_min_09_y, 'sum_quantiles_min_09_z': sum_quantiles_min_09_z,
   'sum_quantiles_min_squares_09_x': sum_quantiles_min_squares_09_x,
   'sum_quantiles_min_squares_09_y': sum_quantiles_min_squares_09_y,
   'sum_quantiles_min_squares_09_z': sum_quantiles_min_squares_09_z,
   'sum_quantiles_max_09_x': sum_quantiles_max_09_x, 'sum_quantiles_max_09_y': sum_quantiles_max_09_y, 'sum_quantiles_max_09_z': sum_quantiles_max_09_z,
   'sum_quantiles_max_squares_09_x': sum_quantiles_max_squares_09_x,
   'sum_quantiles_max_squares_09_y': sum_quantiles_max_squares_09_y,
   'sum_quantiles_max_squares_09_z': sum_quantiles_max_squares_09_z,
   'pairwise_differences_005_01_x': pairwise_differences_005_01_x,
   'pairwise_differences_005_01_y': pairwise_differences_005_01_y,
   'pairwise_differences_005_01_z': pairwise_differences_005_01_z,
   'pairwise_differences_01_025_x': pairwise_differences_005_01_x,
   'pairwise_differences_01_025_y': pairwise_differences_01_025_y,
   'pairwise_differences_01_025_z': pairwise_differences_01_025_z,
   'pairwise_differences_025_05_x': pairwise_differences_025_05_x,
   'pairwise_differences_025_05_y': pairwise_differences_025_05_y,
   'pairwise_differences_025_05_z': pairwise_differences_025_05_z,
   'average_absolute_difference': average_absolute_difference, 'average_intensity': average_intensity,
   'signal_magnitude_area': signal_magnitude_area, 'movement_variation': movement_variation,
   'entropy': entropy, 'energy': energy,
   'activity_x': activity_x, 'activity_y': activity_y, 'activity_z': activity_z,
   'mobility_x': mobility_x, 'mobility_y': mobility_y, 'mobility_z': mobility_z,
   'complexity_x': complexity_x, 'complexity_y': complexity_y, 'complexity_z': complexity_z,
   'durbin_watson_x': durbin_watson_x, 'durbin_watson_y': durbin_watson_y, 'durbin_watson_z': durbin_watson_z,
   'correlation_magnitude_horizontal': correlation_magnitude_horizontal,
   'correlation_magnitude_vertical': correlation_magnitude_vertical,
   'correlation_horizontal_vertical': correlation_horizontal_vertical, 'Activity': activity, 'ActivityName': activity_name}

full = pd.DataFrame(data=d)
train, test = train_test_split(full, test_size=0.50)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

