# Prediction of Solar Radiation using ANN
In this Deep Learning research, an Artificial Neural Network (ANN) is utilized based on a hyperparameter tuning-focused approach to discover a new solution for Solar Radiation using a highly noisy and skewed dataset. </br> 
[Published Conference Paper Link](https://iopscience.iop.org/article/10.1088/1742-6596/1767/1/012041)


![alt text](https://github.com/shahriar-rahman/Prediction-of-Solar-Radiation/blob/main/img/solarpanels.JPG)

## Abstract
Most solar applications and systems can be reliably used to generate electricity and
power in many homes and offices. Recently, there is an increase in many solar required
systems that can be found not only in electricity generation but other applications such as solar
distillation, water heating, heating of buildings, meteorology and producing solar conversion
energy. Prediction of solar radiation is very significant in order to accomplish the previously
mentioned objectives. In this paper, the main target is to present an algorithm that can be used
to predict an hourly activity of solar radiation. Using a dataset that consists of temperature of
air, time, humidity, wind speed, atmospheric pressure, direction of wind and solar radiation
data, an Artificial Neural Network (ANN) model is constructed to effectively forecast solar
radiation using the available weather forecast data. Two models are created to efficiently create
a system capable of interpreting patterns through supervised learning data and predict the
correct amount of radiation present in the atmosphere. The results of the two statistical
indicators: Mean Absolute Error (MAE) and Mean Squared Error (MSE) are performed and
compared with observed and predicted data. These two models were able to generate efficient
predictions with sufficient performance accuracy.


![alt text](https://github.com/shahriar-rahman/Prediction-Of-Solar-Radiation/blob/main/Diagrams/SolarPanel.PNG)

## Introduction
With the rise in technological advancements in our digital modern world, comes the rise in demand for
electricity. Solar Panels have become one of the most used devices for electricity production as it
has become more affordable and more efficient than ever. Solar radiation data is significant in
various sectors such as in conversion and generation of energy from sunlight, water heating, water
distillation and meteorology. Many solar technologies started to take full advantage of using
solar radiation as a foundation for producing electrical energy. Furthermore, insolation energy has
proven to be very valuable in other sectors as well, such as agricultural sectors and rainfall
measurement and detection. On the other hand, it also can have minor negative effects such as:
Radiation Exposure, UV and infra-red rays and climate change, thus making it more essential to
analyze the radiation data.

To address these problems and achieving proper radiation data for energy conversion and other
useful applications, many studies in the literature explored using various techniques of machine
learning to precisely obtain the necessary data. Generally, solar radiation data can be observed as a
time series produced by a random and stochastic process [5]. As a result, precise mathematical
modeling is necessary for efficient generalization. Hence, by using the historical data sample, it can be
mathematically interpreted as a conditional expectation by using a precise model.

![alt text](https://github.com/shahriar-rahman/Prediction-Of-Solar-Radiation/blob/main/Diagrams/DataAnalysis1.PNG) 

## Data Acquisition
The meteorological data consists of air pressure, time, humidity, wind speed, daily temperature, wind
direction and global solar radiation. These data were recorded by a meteorological station from the
Hawai’i Space Exploration Analog and Simulation weather station. Also known as HI-SEAS. The time period of the data collected is for four months (September through December, 2016) between Mission IV and Mission V. HI-SEAS is an environment located on a remote site on the Mauna Loa
side of the saddle area on the Big Island of Hawaii in around 8100 feet above sea level.

## Procedural Diagram
![alt text](https://github.com/shahriar-rahman/Prediction-Of-Solar-Radiation/blob/main/Diagrams/FlowChart.PNG)

## Results & Discussion
Initially, the model consisted of 8 layers with a regularizer values of 0.01 for both bias and kernel.
After running the simulation, it is observed that better performance is accomplished by adding an extra
layer with 15 depth and using a regularizing value of 0.009. We also faced some inconsistencies while
using the Adam optimizer, even though it performed far better than other optimizers. However, we managed to locate the inconsistency in Adam and used a technique to find a workaround using an amalgamation of RMSProp and Momentum known as Adam. RMSProp helps reduce the vertical
oscillation where Gradient descent with momentum adds momentum towards the horizontal direction,
which is good because it prevents overshooting. 

![alt text](https://github.com/shahriar-rahman/Prediction-Of-Solar-Radiation/blob/main/Diagrams/HyperparameterTuning.PNG)

However, even though we want to speed up the learning process at the start because it speeds up the learning process, it also needs to slow down after
a while otherwise it would have difficulties converging. This is why we introduced another
optimization technique known as learning rate decay. Learning rate decay allows the model to train at a much faster speed at the start, however, after
taking some steps, it starts to reduce the speed of the learning process. Thus, giving it more time to
converge properly.

![alt text](https://github.com/shahriar-rahman/Prediction-Of-Solar-Radiation/blob/main/Diagrams/Results.PNG)


