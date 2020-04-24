# mortality
Python implementation of the Deep Recurrent Neural Network fit and extrapolation of temporal trends. (C) Averisera Ltd 2017-2018.

The method uses a Recurrent Neural Network to extrapolate in time a sequence of positive rates(t, i), where t = time and i = label
(e.g. age group for mortality rates). The code is self-contained and requires only standard Python numerical libraries (Pandas, Numpy) and
TensorFlow.

The method was used to discover patters and forecast mortality and fertility trends in microsimulations of England and Wales in the following papers:

*Forecasting the impact of state pension reforms in post-Brexit England and Wales using microsimulation and deep learning* published in Proceedings of PenCon2018 (https://arxiv.org/abs/1802.09427)

*Microsimulations of demographic changes in England and Wales under various EU referendum scenarios* published in International Journal of Microsimulation 10(2), 103 (2017) (https://arxiv.org/abs/1606.04636)

Download the directory "sources" if you want to apply the model to mortality rates.

Licensed under GPL v3.
