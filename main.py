import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, d2_absolute_error_score, median_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

cardio = pd.read_csv('cardioActivities.csv')

dropcols = ["Route Name", "Friend's Tagged", 'GPX File', 'Activity Id']
cardio = cardio.drop(columns=dropcols, axis=1)

totalcol = cardio.columns
# print(totalcol)

value = cardio["Type"].value_counts()
print(value)

cardio["Type"].value_counts().plot(kind="bar")
plt.show()

cycle = cardio[cardio["Type"] == "Cycling"].copy()
run = cardio[cardio["Type"] == "Running"].copy()
walk = cardio[cardio["Type"] == "Walking"].copy()
other = cardio[cardio["Type"] == "Other"].copy()

print(cycle.head())
print(run.head())
print(walk.head())
print(other)

# avg Heart rate in all types.

avgHRincycle = cardio[cardio["Type"] == "Cycling"]["Average Heart Rate (bpm)"].mean()
print(avgHRincycle)

avgHRinrun = cardio[cardio["Type"] == "Running"]["Average Heart Rate (bpm)"].mean()
print(avgHRinrun)

avgHRinwalk = cardio[cardio["Type"] == "Walking"]["Average Heart Rate (bpm)"].mean()
print(avgHRinwalk)

avgHRinother = cardio[cardio["Type"] == "other"]["Average Heart Rate (bpm)"].mean()
print(avgHRinother)

# avg cal burn in all types

avgCBincycle = cardio[cardio["Type"] == "Cycling"]["Calories Burned"].mean()
print(avgCBincycle)

avgCBinrun = cardio[cardio["Type"] == "Running"]["Calories Burned"].mean()
print(avgCBinrun)

avgCBinwalk = cardio[cardio["Type"] == "Walking"]["Calories Burned"].mean()
print(avgCBinwalk)

# Avg of 2018

cardio['Date'] = pd.to_datetime(cardio['Date'])
cardio.sort_values(by='Date', inplace=True)

date = cardio[cardio['Date'] >= '2018']
# print(date.head())
numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
date.set_index('Date', inplace=True)

annualAvg = date.resample('A')[numbercols].mean()
weeklyAvg = date.resample('W')[numbercols].mean()

print("How my average run looks in 2018: \n")
print(annualAvg)
print(weeklyAvg)

# line plot of Calories Burned with Average Speed .

sns.lineplot(data=date, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Average Speed ')
plt.show()

# scatter plot of Calories Burned with Average Speed.

sns.scatterplot(data=date, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Average Speed ')
plt.show()

# lm-plot of Calories Burned with Average Speed.

sns.lmplot(data=date, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Average Speed ')
plt.show()

# hist-plot of Calories Burned with Average Speed.

sns.histplot(data=date, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Average Speed ')
plt.show()

# seaborn plot of Distance (km) with Average Speed .

sns.lineplot(data=date, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('line plot of Distance (km) with Average Speed ')
plt.show()

# scatter plot of Distance (km) with Average Speed.

sns.scatterplot(data=date, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed ')
plt.show()

# lm-plot of Distance (km) with Average Speed.

sns.lmplot(data=date, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('lm-plot of Distance (km) with Average Speed ')
plt.show()

# hist plot of Distance (km) with Average Speed.

sns.histplot(data=date, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('hist plot of Distance (km) with Average Speed ')
plt.show()

# seaborn plot of Average Heart Rate (bpm) with Average Speed .

sns.lineplot(data=date, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Speed ')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Speed.

sns.scatterplot(data=date, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed ')
plt.show()

# lm-plot of Average Heart Rate (bpm) with Average Speed.

sns.lmplot(data=date, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Average Speed ')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Speed.

sns.histplot(data=date, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Speed ')
plt.show()

# seaborn plot of Average Heart Rate (bpm) withAverage Pace.

sns.lineplot(data=date, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Pace')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Pace.

sns.scatterplot(data=date, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Average Pace ')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Pace.

sns.histplot(data=date, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Pace.')
plt.show()

# line plot of Average Heart Rate (bpm) with Climb (m).

sns.lineplot(data=date, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Climb (m) ')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Climb (m).

sns.scatterplot(data=date, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Climb (m) ')
plt.show()

# hist plot of Average Heart Rate (bpm) with Climb (m).

sns.histplot(data=date, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Climb (m).')
plt.show()

# lm-plot of Average Heart Rate (bpm) with climb .

sns.lmplot(data=date, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Climb (m)')
# plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Climb (m) ')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Climb (m)')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Climb (m).')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Climb (m) ')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date, x='Climb (m)', y='Calories Burned')
plt.title('line plot of Calories Burned with Climb (m) ')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date, x='Climb (m)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Climb (m) ')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date, x='Climb (m)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Climb (m).')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date, x='Climb (m)', y='Calories Burned')
plt.title('lm-plot of Calories Burned with Climb (m)')
plt.show()

# joint plot of Calories Burned with Climb (m).

sns.jointplot(data=date, x='Climb (m)', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with climb .

sns.boxplot(data=date, x='Climb (m)', y='Calories Burned')
plt.title('box-plot of Calories Burned with Climb (m) ')
plt.show()

# line plot of Calories Burned with Type.

sns.lineplot(data=date, x='Type', y='Calories Burned')
plt.title('line plot of Calories Burned with Type')
plt.show()

# scatter plot of Calories Burned with Type.

sns.scatterplot(data=date, x='Type', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Type')
plt.show()

# hist plot of Calories Burned with Type.

sns.histplot(data=date, x='Type', y='Calories Burned')
plt.title('hist plot of Calories Burned with Type.')
plt.show()

# joint plot of Calories Burned with Type.

sns.jointplot(data=date, x='Type', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with Type .

sns.boxplot(data=date, x='Type', y='Calories Burned')
plt.title('box-plot of Calories Burned with Type ')
plt.show()

# line plot of Calories Burned with Distance (km).

sns.lineplot(data=date, x='Distance (km)', y='Calories Burned')
plt.title('line plot of Calories Burned with Distance (km)')
plt.show()

# scatter plot of Calories Burned with Distance (km).

sns.scatterplot(data=date, x='Distance (km)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Distance (km)')
plt.show()

# hist plot of Calories Burned with Distance (km).

sns.histplot(data=date, x='Distance (km)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Distance (km).')
plt.show()

# joint plot of Calories Burned with Distance (km).

sns.jointplot(data=date, x='Distance (km)', y='Calories Burned', color='blue')
plt.show()

# Avg of 2017

date17 = cardio[cardio['Date'] >= '2017']
# print(date.head())

numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
date17.set_index('Date', inplace=True)

annualAvg = date17.resample('A')[numbercols].mean()
weeklyAvg = date17.resample('W')[numbercols].mean()

print("How my average run looks in 2017: \n")
print(annualAvg)
print(weeklyAvg)

# line plot of Calories Burned with Average Speed .

sns.lineplot(data=date17, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Average Speed 0f 2017')
plt.show()

# scatter plot of Calories Burned with Average Speed.

sns.scatterplot(data=date17, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Average Speed 0f 2017')
plt.show()

# lm-plot of Calories Burned with Average Speed.

sns.lmplot(data=date17, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Average Speed 0f 2017')
plt.show()

# hist-plot of Calories Burned with Average Speed.

sns.histplot(data=date17, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Average Speed 0f 2017')
plt.show()

# seaborn plot of Distance (km) with Average Speed .

sns.lineplot(data=date17, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('line plot of Distance (km) with Average Speed 0f 2017')
plt.show()

# scatter plot of Distance (km) with Average Speed.

sns.scatterplot(data=date17, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2017')
plt.show()

# lm-plot of Distance (km) with Average Speed.

sns.lmplot(data=date17, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('lm-plot of Distance (km) with Average Speed 0f 2017')
plt.show()

# hist plot of Distance (km) with Average Speed.

sns.histplot(data=date17, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('hist plot of Distance (km) with Average Speed 0f 2017')
plt.show()

# seaborn plot of Average Heart Rate (bpm) with Average Speed .

sns.lineplot(data=date17, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Speed 0f 2017 ')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Speed.

sns.scatterplot(data=date17, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2017')
plt.show()

# lm-plot of Average Heart Rate (bpm) with Average Speed.

sns.lmplot(data=date17, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Average Speed 0f 2017')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Speed.

sns.histplot(data=date17, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Speed 0f 2017')
plt.show()

# seaborn plot of Average Heart Rate (bpm) withAverage Pace.

sns.lineplot(data=date17, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Pace 0f 2017')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Pace.

sns.scatterplot(data=date17, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Average Pace 0f 2017')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Pace.

sns.histplot(data=date17, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Pace 0f 2017')
plt.show()

# line plot of Average Heart Rate (bpm) with Climb (m).

sns.lineplot(data=date17, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Climb (m) 0f 2017')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Climb (m).

sns.scatterplot(data=date17, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Climb (m) 0f 2017')
plt.show()

# hist plot of Average Heart Rate (bpm) with Climb (m).

sns.histplot(data=date17, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Climb (m) 0f 2017')
plt.show()

# # lm-plot of Average Heart Rate (bpm) with climb .

sns.lmplot(data=date17, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Climb (m) 0f 2017')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date17, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Climb (m) 0f 2017')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date17, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2017')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date17, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2017')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date17, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2017 ')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date17, x='Climb (m)', y='Calories Burned')
plt.title('line plot of Calories Burned with Climb (m) 0f 2017')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date17, x='Climb (m)', y='Calories Burned')
plt.title("scatter plot of Calories Burned with Climb (m) 0f 2017 ")
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date17, x='Climb (m)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2017')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date17, x='Climb (m)', y='Calories Burned')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2017')
plt.show()

# joint plot of Calories Burned with Climb (m).

sns.jointplot(data=date17, x='Climb (m)', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with climb .

sns.boxplot(data=date17, x='Climb (m)', y='Calories Burned')
plt.title('box-plot of Calories Burned with Climb (m) 0f 2017')
plt.show()


# line plot of Calories Burned with Type.

sns.lineplot(data=date17, x='Type', y='Calories Burned')
plt.title('line plot of Calories Burned with Type 0f 2017 ')
plt.show()

# scatter plot of Calories Burned with Type.

sns.scatterplot(data=date17, x='Type', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Type 0f 2017')
plt.show()

# hist plot of Calories Burned with Type.

sns.histplot(data=date17, x='Type', y='Calories Burned')
plt.title('hist plot of Calories Burned with Type 0f 2017')
plt.show()

# joint plot of Calories Burned with Type.

sns.jointplot(data=date17, x='Type', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with Type .

sns.boxplot(data=date17, x='Type', y='Calories Burned')
plt.title('box-plot of Calories Burned with Type 0f 2017')
plt.show()

# line plot of Calories Burned with Distance (km).

sns.lineplot(data=date17, x='Distance (km)', y='Calories Burned')
plt.title('line plot of Calories Burned with Distance (km) 0f 2017 ')
plt.show()

# scatter plot of Calories Burned with Distance (km).

sns.scatterplot(data=date17, x='Distance (km)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Distance (km) 0f 2017')
plt.show()

# hist plot of Calories Burned with Distance (km).

sns.histplot(data=date17, x='Distance (km)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Distance (km) 0f 2017')
plt.show()

# joint plot of Calories Burned with Distance (km).

sns.jointplot(data=date17, x='Distance (km)', y='Calories Burned', color='blue')
plt.show()

# Avg of 2016.

date16 = cardio[cardio['Date'] >= '2016']
# print(date.head())

numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
date16.set_index('Date', inplace=True)

annualAvg = date16.resample('A')[numbercols].mean()
weeklyAvg = date16.resample('W')[numbercols].mean()

print("How my average run looks in 2016: \n")
# print(annualAvg)
# print(weeklyAvg)

# line plot of Calories Burned with Average Speed .

sns.lineplot(data=date16, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Average Speed 0f 2016')
plt.show()

# scatter plot of Calories Burned with Average Speed.

sns.scatterplot(data=date16, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Average Speed 0f 2016')
plt.show()

# lm-plot of Calories Burned with Average Speed.

sns.lmplot(data=date16, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Average Speed 0f 2016')
plt.show()

# hist-plot of Calories Burned with Average Speed.

sns.histplot(data=date16, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Average Speed 0f 2016')
plt.show()

# seaborn plot of Distance (km) with Average Speed .

sns.lineplot(data=date16, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('line plot of Distance (km) with Average Speed 0f 2016')
plt.show()

# scatter plot of Distance (km) with Average Speed.

sns.scatterplot(data=date16, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2016')
plt.show()

# lm-plot of Distance (km) with Average Speed.

sns.lmplot(data=date16, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('lm-plot of Distance (km) with Average Speed 0f 2016')
plt.show()

# hist plot of Distance (km) with Average Speed.

sns.histplot(data=date16, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('hist plot of Distance (km) with Average Speed 0f 2016')
plt.show()

# seaborn plot of Average Heart Rate (bpm) with Average Speed .

sns.lineplot(data=date16, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Speed 0f 2016 ')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Speed.

sns.scatterplot(data=date16, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2016')
plt.show()

# lm-plot of Average Heart Rate (bpm) with Average Speed.

sns.lmplot(data=date16, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Average Speed 0f 2016')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Speed.

sns.histplot(data=date16, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Speed 0f 2016')
plt.show()

# seaborn plot of Average Heart Rate (bpm) withAverage Pace.

sns.lineplot(data=date16, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title("line plot of Average Heart Rate (bpm) with Average Pace 0f 2016")
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Pace.

sns.scatterplot(data=date16, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Average Pace 0f 2016')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Pace.

sns.histplot(data=date16, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Pace 0f 2016')
plt.show()

# line plot of Average Heart Rate (bpm) with Climb (m).

sns.lineplot(data=date16, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Climb (m) 0f 2016')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Climb (m).

sns.scatterplot(data=date16, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Climb (m) 0f 2016')
plt.show()

# hist plot of Average Heart Rate (bpm) with Climb (m).

sns.histplot(data=date16, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Climb (m) 0f 2016')
plt.show()

#  lm-plot of Average Heart Rate (bpm) with climb .

sns.lmplot(data=date16, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Climb (m) 0f 2016')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date16, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date16, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date16, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date16, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2016 ')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date16, x='Climb (m)', y='Calories Burned')
plt.title('line plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date16, x='Climb (m)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date16, x='Climb (m)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date16, x='Climb (m)', y='Calories Burned')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# joint plot of Calories Burned with Climb (m).

sns.jointplot(data=date16, x='Climb (m)', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with climb .

sns.boxplot(data=date16, x='Climb (m)', y='Calories Burned')
plt.title('box-plot of Calories Burned with Climb (m) 0f 2016')
plt.show()

# line plot of Calories Burned with Type.

sns.lineplot(data=date16, x='Type', y='Calories Burned')
plt.title('line plot of Calories Burned with Type 0f 2016')
plt.show()

# scatter plot of Calories Burned with Type.

sns.scatterplot(data=date16, x='Type', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Type 0f 2016')
plt.show()

# hist plot of Calories Burned with Type.

sns.histplot(data=date16, x='Type', y='Calories Burned')
plt.title('hist plot of Calories Burned with Type 0f 2016')
plt.show()

# joint plot of Calories Burned with Type.

sns.jointplot(data=date16, x='Type', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with Type .

sns.boxplot(data=date16, x='Type', y='Calories Burned')
plt.title('box-plot of Calories Burned with Type 0f 2016')
plt.show()

# line plot of Calories Burned with Distance (km).

sns.lineplot(data=date16, x='Distance (km)', y='Calories Burned')
plt.title('line plot of Calories Burned with Distance (km) 0f 2016 ')
plt.show()

# scatter plot of Calories Burned with Distance (km).

sns.scatterplot(data=date16, x='Distance (km)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Distance (km) 0f 2016')
plt.show()

# hist plot of Calories Burned with Distance (km).

sns.histplot(data=date16, x='Distance (km)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Distance (km) 0f 2016')
plt.show()

# joint plot of Calories Burned with Distance (km).

sns.jointplot(data=date16, x='Distance (km)', y='Calories Burned', color='blue')
plt.show()

# Avg of 2015.

date15 = cardio[cardio['Date'] >= '2015']
print(date.head())

numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
date15.set_index('Date', inplace=True)

annualAvg = date15.resample('A')[numbercols].mean()
weeklyAvg = date15.resample('W')[numbercols].mean()

print("How my average run looks in 2015: \n")
print(annualAvg)
print(weeklyAvg)

# line plot of Calories Burned with Average Speed .

sns.lineplot(data=date15, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Average Speed 0f 2015')
plt.show()

# scatter plot of Calories Burned with Average Speed.

sns.scatterplot(data=date15, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Average Speed 0f 2015')
plt.show()

# lm-plot of Calories Burned with Average Speed.

sns.lmplot(data=date15, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Average Speed 0f 2015')
plt.show()

# hist-plot of Calories Burned with Average Speed.

sns.histplot(data=date15, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Average Speed 0f 2015')
plt.show()

# seaborn plot of Distance (km) with Average Speed .

sns.lineplot(data=date15, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('line plot of Distance (km) with Average Speed 0f 2015')
plt.show()

# scatter plot of Distance (km) with Average Speed.

sns.scatterplot(data=date15, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2015')
plt.show()

# lm-plot of Distance (km) with Average Speed.

sns.lmplot(data=date15, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('lm-plot of Distance (km) with Average Speed 0f 2015')
plt.show()

# hist plot of Distance (km) with Average Speed.

sns.histplot(data=date15, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('hist plot of Distance (km) with Average Speed 0f 2015')
plt.show()

# seaborn plot of Average Heart Rate (bpm) with Average Speed .

sns.lineplot(data=date15, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Speed 0f 2015 ')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Speed.

sns.scatterplot(data=date15, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2015')
plt.show()

# lm-plot of Average Heart Rate (bpm) with Average Speed.

sns.lmplot(data=date15, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Average Speed 0f 2015')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Speed.

sns.histplot(data=date15, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Speed 0f 2015')
plt.show()

# seaborn plot of Average Heart Rate (bpm) withAverage Pace.

sns.lineplot(data=date15, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title("line plot of Average Heart Rate (bpm) with Average Pace 0f 2015")
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Pace.

sns.scatterplot(data=date15, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Average Pace 0f 2015')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Pace.

sns.histplot(data=date15, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Pace 0f 2015')
plt.show()

# line plot of Average Heart Rate (bpm) with Climb (m).

sns.lineplot(data=date15, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Climb (m) 0f 2015')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Climb (m).

sns.scatterplot(data=date15, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Climb (m) 0f 2015')
plt.show()

# hist plot of Average Heart Rate (bpm) with Climb (m).

sns.histplot(data=date15, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Climb (m) 0f 2015')
plt.show()

# # lm-plot of Average Heart Rate (bpm) with climb .

sns.lmplot(data=date15, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Climb (m) 0f 2015')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date15, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date15, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date15, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date15, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date15, x='Climb (m)', y='Calories Burned')
plt.title('line plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date15, x='Climb (m)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date15, x='Climb (m)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date15, x='Climb (m)', y='Calories Burned')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# joint plot of Calories Burned with Climb (m).

sns.jointplot(data=date15, x='Climb (m)', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with climb .

sns.boxplot(data=date15, x='Climb (m)', y='Calories Burned')
plt.title('box-plot of Calories Burned with Climb (m) 0f 2015')
plt.show()

# line plot of Calories Burned with Type.

sns.lineplot(data=date15, x='Type', y='Calories Burned')
plt.title('line plot of Calories Burned with Type 0f 2015 ')
plt.show()

# scatter plot of Calories Burned with Type.

sns.scatterplot(data=date15, x='Type', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Type 0f 2015')
plt.show()

# hist plot of Calories Burned with Type.

sns.histplot(data=date15, x='Type', y='Calories Burned')
plt.title('hist plot of Calories Burned with Type 0f 2015')
plt.show()

# joint plot of Calories Burned with Type.

sns.jointplot(data=date15, x='Type', y='Calories Burned', color='blue')
plt.show()

#  box-plot of Calories Burned with Type .

sns.boxplot(data=date15, x='Type', y='Calories Burned')
plt.title('box-plot of Calories Burned with Type 0f 2015')
plt.show()

# line plot of Calories Burned with Distance (km).

sns.lineplot(data=date15, x='Distance (km)', y='Calories Burned')
plt.title('line plot of Calories Burned with Distance (km) 0f 2015 ')
plt.show()

# scatter plot of Calories Burned with Distance (km).

sns.scatterplot(data=date15, x='Distance (km)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Distance (km) 0f 2015')
plt.show()

# hist plot of Calories Burned with Distance (km).

sns.histplot(data=date15, x='Distance (km)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Distance (km) 0f 2015')
plt.show()

# joint plot of Calories Burned with Distance (km) of 2015.

sns.jointplot(data=date15, x='Distance (km)', y='Calories Burned', color='blue')
plt.show()

# Avg of 2014.

date14 = cardio[cardio['Date'] >= '2014']
# print(date.head())

numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
date14.set_index('Date', inplace=True)

annualAvg = date14.resample('A')[numbercols].mean()
weeklyAvg = date14.resample('W')[numbercols].mean()

print("How my average run looks in 2014: \n")
print(annualAvg)
print(weeklyAvg)

# line plot of Calories Burned with Average Speed .

sns.lineplot(data=date14, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Average Speed 0f 2014')
plt.show()

# scatter plot of Calories Burned with Average Speed.

sns.scatterplot(data=date14, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Average Speed 0f 2014')
plt.show()

# lm-plot of Calories Burned with Average Speed.

sns.lmplot(data=date14, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Average Speed 0f 2014')
plt.show()

# hist-plot of Calories Burned with Average Speed.

sns.histplot(data=date14, x='Average Speed (km/h)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Average Speed 0f 2014')
plt.show()

# seaborn plot of Distance (km) with Average Speed .

sns.lineplot(data=date14, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('line plot of Distance (km) with Average Speed 0f 2014')
plt.show()

# scatter plot of Distance (km) with Average Speed.

sns.scatterplot(data=date14, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2014')
plt.show()

# lm-plot of Distance (km) with Average Speed.

sns.lmplot(data=date14, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('lm-plot of Distance (km) with Average Speed 0f 2014')
plt.show()

# hist plot of Distance (km) with Average Speed.

sns.histplot(data=date14, x='Average Speed (km/h)', y='Distance (km)', hue='Type')
plt.title('hist plot of Distance (km) with Average Speed 0f 2014')
plt.show()

# seaborn plot of Average Heart Rate (bpm) with Average Speed .

sns.lineplot(data=date14, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Average Speed 0f 2014 ')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Speed.

sns.scatterplot(data=date14, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Distance (km) with Average Speed 0f 2014')
plt.show()

# lm-plot of Average Heart Rate (bpm) with Average Speed.

sns.lmplot(data=date14, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Average Speed 0f 2014')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Speed.

sns.histplot(data=date14, x='Average Speed (km/h)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Speed 0f 2014')
plt.show()

# seaborn plot of Average Heart Rate (bpm) withAverage Pace.

sns.lineplot(data=date14, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title("line plot of Average Heart Rate (bpm) with Average Pace 0f 2014")
plt.show()

# scatter plot of Average Heart Rate (bpm) with Average Pace.

sns.scatterplot(data=date14, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Average Pace 0f 2014')
plt.show()

# hist plot of Average Heart Rate (bpm) with Average Pace.

sns.histplot(data=date14, x='Average Pace', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Average Pace 0f 2014')
plt.show()

# line plot of Average Heart Rate (bpm) with Climb (m).

sns.lineplot(data=date14, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('line plot of Average Heart Rate (bpm) with Climb (m) 0f 2014')
plt.show()

# scatter plot of Average Heart Rate (bpm) with Climb (m).

sns.scatterplot(data=date14, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('scatter plot of Average Heart Rate (bpm) with Climb (m) 0f 2014')
plt.show()

# hist plot of Average Heart Rate (bpm) with Climb (m).

sns.histplot(data=date14, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('hist plot of Average Heart Rate (bpm) with Climb (m) 0f 2014')
plt.show()

# # lm-plot of Average Heart Rate (bpm) with climb .

sns.lmplot(data=date14, x='Climb (m)', y='Average Heart Rate (bpm)', hue='Type')
plt.title('lm-plot of Average Heart Rate (bpm) with Climb (m) 0f 2014')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date14, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date14, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date14, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date14, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date14, x='Climb (m)', y='Calories Burned')
plt.title('line plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date14, x='Climb (m)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date14, x='Climb (m)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# # lm-plot of Calories Burned with climb .

sns.lmplot(data=date14, x='Climb (m)', y='Calories Burned')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# joint plot of Calories Burned with Climb (m).

sns.jointplot(data=date14, x='Climb (m)', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with climb .

sns.boxplot(data=date14, x='Climb (m)', y='Calories Burned')
plt.title('box-plot of Calories Burned with Climb (m) 0f 2014')
plt.show()

# line plot of Calories Burned with Type.

sns.lineplot(data=date14, x='Type', y='Calories Burned')
plt.title('line plot of Calories Burned with Type 0f 2014 ')
plt.show()

# scatter plot of Calories Burned with Type.

sns.scatterplot(data=date14, x='Type', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Type 0f 2014')
plt.show()

# hist plot of Calories Burned with Type.

sns.histplot(data=date14, x='Type', y='Calories Burned')
plt.title('hist plot of Calories Burned with Type 0f 2014')
plt.show()

# joint plot of Calories Burned with Type.

sns.jointplot(data=date14, x='Type', y='Calories Burned', color='blue')
plt.show()

# # box-plot of Calories Burned with Type .

sns.boxplot(data=date14, x='Type', y='Calories Burned')
plt.title('box-plot of Calories Burned with Type 0f 2014')
plt.show()

# line plot of Calories Burned with Distance (km).

sns.lineplot(data=date14, x='Distance (km)', y='Calories Burned')
plt.title('line plot of Calories Burned with Distance (km) 0f 2014')
plt.show()

# scatter plot of Calories Burned with Distance (km).

sns.scatterplot(data=date14, x='Distance (km)', y='Calories Burned')
plt.title('scatter plot of Calories Burned with Distance (km) 0f 2014')
plt.show()

# hist plot of Calories Burned with Distance (km).

sns.histplot(data=date14, x='Distance (km)', y='Calories Burned')
plt.title('hist plot of Calories Burned with Distance (km) 0f 2014')
plt.show()

# joint plot of Calories Burned with Distance (km) of 2015.

sns.jointplot(data=date14, x='Distance (km)', y='Calories Burned', color='blue')
plt.show()

# Avg of 2013.

date13 = cardio[cardio['Date'] >= '2013']
# print(date.head())

numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
date13.set_index('Date', inplace=True)

annualAvg13 = date13.resample('A')[numbercols].mean()
weeklyAvg13 = date13.resample('W')[numbercols].mean()

print("How my average run looks in 2013: \n")
print(annualAvg13)
print(weeklyAvg13)

# here is some plots of 2013

# line plot of Calories Burned with Climb (m).

sns.lineplot(data=date13, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('line plot of Calories Burned with Climb (m) 0f 2013')
plt.show()

# scatter plot of Calories Burned with Climb (m).

sns.scatterplot(data=date13, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('scatter plot of Calories Burned with Climb (m) 0f 2013')
plt.show()

# hist plot of Calories Burned with Climb (m).

sns.histplot(data=date13, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('hist plot of Calories Burned with Climb (m) 0f 2013')
plt.show()

# lm-plot of Calories Burned with climb .

sns.lmplot(data=date13, x='Climb (m)', y='Calories Burned', hue='Type')
plt.title('lm-plot of Calories Burned with Climb (m) 0f 2013')
plt.show()

# Avg of 2012.

date12 = cardio[cardio['Date'] >= '2012']
# print(date12.head())

numbercols = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']
# date13.set_index('Date', inplace=True)

annualAvg12 = date13.resample('A')[numbercols].mean()
weeklyAvg12 = date.resample('W')[numbercols].mean()

print("How my average run looks in 2012: \n")
print(annualAvg12)
print(weeklyAvg12)

#  Linear regression of Average Speed and Average Heart Rate.

x = cardio[['Average Speed (km/h)']]
y = cardio[['Average Heart Rate (bpm)']].fillna(value=0.0)
# print(x.shape)
# print(y.shape)
# print(x.head())
# print(y.head())
# print(x.info())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.shape)
# print(xTest.shape)
# print(yTrain.shape)
# print(yTest.shape)

lr = LinearRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
print(yPredict.shape)
#
data = {}
data['ytest'] = np.reshape(yTest, (-1))
data['ypredict'] = np.reshape(np.array(yPredict), (-1))
# print(data)
df = pd.DataFrame(data=data)
# print(df)

# reg plot between ytest and y predict.

sns.regplot(data=df, x='ytest', y='ypredict')
plt.title('reg plot between ytest and y predict')
plt.show()

sns.scatterplot(data=df, x='ytest', y='ypredict')
plt.title('scatter plot between ytest and y predict')
plt.show()

mse = mean_squared_error(yTest, yPredict)
print(mse)

d2Error = d2_absolute_error_score(yTest, yPredict)
print(d2Error)

medianError = median_absolute_error(yTest, yPredict)
print(medianError)

#  Linear regression of Average Speed calories burned.

x = cardio[['Average Speed (km/h)']]
y = cardio[['Calories Burned']]

# print(x.shape)
# print(y.shape)
# print(x.head())
# print(y.head())
# print(x.info())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.shape)
# print(xTest.shape)
# print(yTrain.shape)
# print(yTest.shape)

lr = LinearRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
print(yPredict.shape)
#
data = {}
data['ytest'] = np.reshape(yTest, (-1))
data['ypredict'] = np.reshape(np.array(yPredict), (-1))
# print(data)
df = pd.DataFrame(data=data)
# print(df)

# reg plot between ytest and y predict.

sns.regplot(data=df, x='ytest', y='ypredict')
plt.title('regplot between ytest and y predict')
plt.show()

sns.scatterplot(data=df, x='ytest', y='ypredict')
plt.title('scatter plot between ytest and y predict')
plt.show()

mse = mean_squared_error(yTest, yPredict)
print(mse)

d2Error = d2_absolute_error_score(yTest, yPredict)
print(d2Error)

medianError = median_absolute_error(yTest, yPredict)
print(medianError)

#  Linear regression of Average Speed with all types

x = cardio[['Distance (km)']]
y = cardio[['Average Speed (km/h)']]

# print(x.shape)
# print(y.shape)
# print(x.head())
# print(y.head())
# print(x.info())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.shape)
# print(xTest.shape)
# print(yTrain.shape)
# print(yTest.shape)

lr = LinearRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
print(yPredict.shape)
#
data = {}
data['ytest'] = np.reshape(yTest, (-1))
data['ypredict'] = np.reshape(np.array(yPredict), (-1))
# print(data)
df = pd.DataFrame(data=data)
# print(df)

# reg plot between ytest and y predict.

sns.regplot(data=df, x='ytest', y='ypredict')
plt.title('regplot between ytest and y predict')
plt.show()

sns.scatterplot(data=df, x='ytest', y='ypredict')
plt.title('scatter plot between ytest and y predict')
plt.show()

mse = mean_squared_error(yTest, yPredict)
print(mse)

d2Error = d2_absolute_error_score(yTest, yPredict)
print(d2Error)

medianError = median_absolute_error(yTest, yPredict)
print(medianError)

# LogisticRegression of type  and Calories Burned.

cardio['Type'] = cardio['Type'] == 'Running'

x = cardio[['Calories Burned']]
y = cardio[['Type']]

# print(x.head())
# print(y.head())
# print(x.info())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

lr = LogisticRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
prediction = np.reshape(yPredict, (-1, 1))
print(prediction.shape)

acc = accuracy_score(yTest, yPredict)
print(acc)

yPredict = lr.predict_proba(xTest)[:, 1]

plt.scatter(xTest, yTest, color='green')
xValue = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
yValue = lr.predict_proba(xValue)[:, 1]
plt.xlabel('x (Calories Burned)')
plt.ylabel('y (Type)')
plt.title('LogisticRegression sigmoid line')
plt.plot(xValue, yValue, color='blue')
plt.show()

sns.violinplot(data=cardio, x='Type', y='Calories Burned', color='green')
plt.title('violin plot of LogisticRegression between type and Calories Burned.')
sns.set(style='whitegrid')
plt.show()

# LogisticRegression of type (cycling) and Calories Burned.

cardio['Type'] = cardio['Type'] == 'Cycling'

x = cardio[['Calories Burned']]
y = cardio[['Type']]

# print(x.head())
# print(y.head())
# print(x.info())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

lr = LogisticRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
prediction = np.reshape(yPredict, (-1, 1))
print(prediction.shape)

acc = accuracy_score(yTest, yPredict)
print(acc)

yPredict = lr.predict_proba(xTest)[:, 1]

# sigmoid line between type (cycling) and Calories Burned.

plt.scatter(xTest, yTest, color='green')
xValue = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
yValue = lr.predict_proba(xValue)[:, 1]
plt.xlabel('x (Calories Burned)')
plt.ylabel('y (Type)')
plt.title('LogisticRegression sigmoid line')
plt.plot(xValue, yValue, color='blue')
plt.show()

# violin plot between type (cycling) and Calories Burned.

sns.violinplot(data=cardio, x='Type', y='Calories Burned', color='red')
plt.title('violin plot of LogisticRegression between type (cycling) and Calories Burned.')
sns.set(style='whitegrid')
plt.show()

# LogisticRegression of type (Walking) and Calories Burned.

cardio['Type'] = cardio['Type'] == 'Walking'

x = cardio[['Calories Burned']]
y = cardio[['Type']]

# print(x.head())
# print(y.head())
# print(x.info())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

lr = LogisticRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
prediction = np.reshape(yPredict, (-1, 1))
print(prediction.shape)

acc = accuracy_score(yTest, yPredict)
print(acc)

yPredict = lr.predict_proba(xTest)[:, 1]

# sigmoid line between type (Walking) and Calories Burned.

plt.scatter(xTest, yTest, color='red')
xValue = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
yValue = lr.predict_proba(xValue)[:, 1]
plt.xlabel('x (Calories Burned)')
plt.ylabel('y (Type)')
plt.title('LogisticRegression sigmoid line')
plt.plot(xValue, yValue, color='black')
plt.show()

# violin plot between type (Walking) and Calories Burned.

sns.violinplot(data=cardio, x='Type', y='Calories Burned', color='blue')
plt.title('violin plot of LogisticRegression between type (Walking) and Calories Burned.')
sns.set(style='whitegrid')
plt.show()

# LogisticRegression of Notes and Calories Burned.

cardio['Notes'] = cardio['Notes'] == 'TomTom MySports Watch'

x = cardio[['Calories Burned']]
y = cardio[['Notes']]

# print(x.head())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

lr = LogisticRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
prediction = np.reshape(yPredict, (-1, 1))
print(prediction.shape)

acc = accuracy_score(yTest, yPredict)
print(acc)

yPredict = lr.predict_proba(xTest)[:, 1]

# sigmoid line between Notes and Calories Burned.

plt.scatter(xTest, yTest, color='red')
xValue = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
yValue = lr.predict_proba(xValue)[:, 1]
plt.xlabel('x (Calories Burned)')
plt.ylabel('y (Notes)')
plt.title('LogisticRegression sigmoid line')
plt.plot(xValue, yValue, color='orange')
plt.show()

# violin plot between type (Walking) and Calories Burned.

sns.violinplot(data=cardio, x='Notes', y='Calories Burned', color='blue')
plt.title('violin plot of LogisticRegression between Notes and Calories Burned.')
sns.set(style='whitegrid')
plt.show()

# RandomForestClassifier for distance and Type.

x = cardio[np.array(['Distance (km)'])]
y = cardio[['Type']]

# print(x.head())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

rfc = RandomForestClassifier()
#
rfc.fit(xTrain, yTrain)
yprediction = rfc.predict(xTest)
prediction = np.reshape(np.array(yprediction), (-1, 1))
#
accScore = accuracy_score(yTest, yprediction)
print(accScore)
#
sns.violinplot(data=cardio, x='Distance (km)', y='Type', color='black')
plt.title('violin plot of RandomForestClassifier between Distance (km) and Type.')
sns.set(style='whitegrid')
plt.show()

# RandomForestClassifier for Average Speed (km/h) and Type.

x = cardio[np.array(['Average Speed (km/h)'])]
y = cardio[['Type']]
# print(x.head())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

rfc = RandomForestClassifier()
#
rfc.fit(xTrain, yTrain)
yprediction = rfc.predict(xTest)
prediction = np.reshape(np.array(yprediction), (-1, 1))
#
accScore = accuracy_score(yTest, yprediction)
print(accScore)
#
sns.violinplot(data=cardio, x='Average Speed (km/h)', y='Type', color='red')
plt.title('violin plot of RandomForestClassifier between Average Speed (km/h) and Type.')
sns.set(style='whitegrid')
plt.show()

# RandomForestClassifier for Calories Burned of all Type.

x = cardio[np.array(['Calories Burned'])]
y = cardio[['Type']]
# print(x.head())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

rfc = RandomForestClassifier()
#
rfc.fit(xTrain, yTrain)
yprediction = rfc.predict(xTest)
prediction = np.reshape(np.array(yprediction), (-1, 1))
#
accScore = accuracy_score(yTest, yprediction)
print(accScore)
#
sns.violinplot(data=cardio, x='Calories Burned', y='Type', color='red')
plt.title('violin plot of RandomForestClassifier between Calories Burned and Type.')
sns.set(style='whitegrid')
plt.show()

# DecisionTree for distance and Type.

x = cardio[np.array(['Distance (km)'])]
y = cardio[['Type']]
# print(x.head())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
print(xTrain.head())
print(xTest.head())
print(yTrain.head())
print(yTest.head())

clf = DecisionTreeClassifier()
clf.fit(xTrain, yTrain)
yprediction = clf.predict(xTest)
accScore = accuracy_score(yTest, yprediction)
# print(accScore)
# print(yprediction)

sns.catplot(data=cardio, x='Distance (km)', y='Type', color='black')
plt.title('cat-plot of DecisionTree between Distance and Type.')
plt.show()

sns.stripplot(data=cardio, x='Distance (km)', y='Type')
plt.title('strip plot of DecisionTree between Distance and Type.')
plt.show()

# DecisionTree for Average Speed (km/h) and Type.

x = cardio[np.array(['Average Speed (km/h)'])]
y = cardio[['Type']]
# print(x.head())
# print(x.info())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
print(xTrain.head())
print(xTest.head())
print(yTrain.head())
print(yTest.head())

clf = DecisionTreeClassifier()
clf.fit(xTrain, yTrain)
yprediction = clf.predict(xTest)
accScore = accuracy_score(yTest, yprediction)
# print(accScore)
# print(yprediction)

# cat-plot of DecisionTree between Average Speed and Type.

sns.catplot(data=cardio, x='Average Speed (km/h)', y='Type', color='red')
plt.title('cat-plot of DecisionTree between Average Speed and Type.')
plt.show()

# strip plot of DecisionTree between Average Speed  and Type.

sns.stripplot(data=cardio, x='Average Speed (km/h)', y='Type')
plt.title('strip plot of DecisionTree between Average Speed  and Type.')
plt.show()

# swarm plot of DecisionTree between Average Speed and Type.

sns.swarmplot(data=cardio, x='Average Speed (km/h)', y='Type', hue='Type')
plt.title('swarm plot of DecisionTree between Average Speed and Type.')
plt.show()

# DecisionTree for Calories Burned and Type.

x = cardio[np.array(['Calories Burned'])]
y = cardio[['Type']]
# print(x.head())
# print(x.info())
# print(y.head())
# print(y.info())
# print(x.describe())
# print(y.describe())

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
# print(xTrain.head())
# print(xTest.head())
# print(yTrain.head())
# print(yTest.head())

clf = DecisionTreeClassifier()
clf.fit(xTrain, yTrain)
yprediction = clf.predict(xTest)
accScore = accuracy_score(yTest, yprediction)
# print(accScore)
# print(yprediction)

# cat-plot of DecisionTree between Calories Burned and Type.

sns.catplot(data=cardio, x='Calories Burned', y='Type', color='red')
plt.title('cat-plot of DecisionTree between Calories Burned and Type.')
plt.show()

# strip plot of DecisionTree between Calories Burned and Type.

sns.stripplot(data=cardio, x='Calories Burned', y='Type')
plt.title('strip plot of DecisionTree between Calories Burned and Type.')
plt.show()

# swarm plot of DecisionTree between Calories Burned and Type.

sns.swarmplot(data=cardio, x='Calories Burned', y='Type', hue='Type')
plt.title('swarm plot of DecisionTree between Calories Burned and Type.')
plt.show()
