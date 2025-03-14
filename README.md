Accurate river flow forecasting plays a critical role in the timely and effective management of water resources, guiding irrigation decisions, flood risk assessment, dam release timing, and various other applications. Predicting flow rates on an hourly, daily, or monthly basis—or even further out—supports both system optimization and short- and long-term planning.

Daily flow forecasts, in particular, are valuable tools for water managers who must allocate limited resources among competing users. Additionally, they are essential for flood-prone areas where early warnings can help mitigate flood impacts. Flooding, one of the most frequent natural disasters, has severely impacted many regions globally, including Iran, causing significant human and financial losses, especially in vulnerable areas.

Accurate data on each stage of river flow is crucial for analyzing, designing, and implementing water resource projects such as reservoirs, dams, flood control channels, and flow forecasts. However, stream discharge is highly non-linear, influenced by complex meteorological factors, and varies across spatial and temporal scales, making accurate short- and long-term predictions challenging.

Currently, two main approaches are used to address discharge prediction:

Model-Oriented Approach: This method relies on physical hydrological models of processes like precipitation-runoff. Despite their widespread use, model-oriented approaches face challenges with performance and uncertainty analysis. Additionally, calibrating these models to match specific basin characteristics can be difficult due to the need for extensive input data, including topographic, precipitation, discharge, and water level information.

Data-Driven Approach: Unlike the model-driven method, which depends on physical runoff models, the data-driven approach relies on statistical analysis of linear and non-linear relationships between input and output data, without assuming a specific physical process. This method has gained popularity as it minimizes assumptions about the rainfall-runoff process.

With significant advancements in computational science, data-driven methods—especially those utilizing artificial neural networks (ANN), machine learning (ML), and deep learning (DL)—have become particularly prominent. Among deep learning algorithms, long short-term memory (LSTM) networks, which address vanishing gradient issues by controlling information flow through input, forget, and output gates, have shown the ability to retain past information over extended periods.

LSTM networks are part of the recurrent neural network (RNN) family, designed to manage sequential data effectively, making them well-suited to time-series forecasting tasks. In addition to LSTM, gated recurrent unit (GRU) networks are another widely used RNN-based architecture, both offering variations that enhance RNN’s performance for time-series prediction.

Numerous studies have employed LSTM and GRU models in hydrological forecasting. Complex DL models combining these architectures, such as convolutional GRU and LSTM models, have been developed to further improve prediction accuracy.

In this project, the daily discharge of the Shapur River, located in the Helleh catchment area in southern Iran, is forecasted using time-series data on precipitation, temperature, wind, evaporation, relative humidity, and solar radiation, along with discharge data from the Jareh hydrometric station for the years 1981 to 2006. Unfortunately, no discharge data has been recorded at this station since 2006. Given that this river is a major water source in Bushehr province—a flood-prone area—and flows downstream of the Rais Ali Delwari Dam, accurate flow forecasting is vital for flood management and water release scheduling from the dam, which supports the province's date palm agriculture, a key national crop.

For this purpose, the project applies five models—CNN, LSTM, GRU, CNN-LSTM, and CNN-GRU—to forecast the flow of the Shapur River at Jareh station. The results will be compared in terms of predictive accuracy and mean squared error (MSE) criteria. The approach closely follows the work of Muhammad et al. (2019), who found that while LSTM models offered slightly better performance, GRU models provided comparable accuracy with faster execution, making GRU an efficient and reliable choice for hydrological applications.

1- To begin, I loaded my dataset into the Python environment and performed any necessary pre-processing.

2- In multivariate time series forecasting, the lookback window or window size specifies the number of previous time steps used to predict the current value. This helps capture patterns over a defined period to improve prediction accuracy. I created a function that takes in the dataset and window size, then generates features and labels based on this lookback period.

3- To develop the five mentioned models (CNN, LSTM, GRU, CNN-LSTM, and CNN-GRU) for the given dataset, I  used TensorFlow and Keras, which are popular libraries for building deep learning models. 

4- The comparison of the results obtained from the different models

