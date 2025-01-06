# Image Classification - Multi-class Weather Dataset

## Overview

This project focuses on building and evaluating machine learning models for classifying images of different weather conditions. The dataset used is the **Mendeley Weather Dataset (MWD)**, which contains 1,125 images representing four weather conditions: **cloudy**, **rain**, **sunrise**, and **shine**. The goal is to develop and compare various models to accurately classify these weather conditions.

## Dataset

The dataset consists of 1,125 images divided into four classes:
- **Cloudy**
- **Rain**
- **Sunrise**
- **Shine**

The dataset is split into three subsets:
- **Training Set**: 60% of the data
- **Validation Set**: 20% of the data
- **Test Set**: 20% of the data

The dataset can be downloaded from [Mendeley Data](https://data.mendeley.com/datasets/4drtyfjtfy/1).

## Project Structure

The project is divided into the following sections:

1. **Data Exploration and Preparation**:
   - Loading and cleaning the dataset.
   - Splitting the dataset into training, validation, and test sets.
   - Encoding image labels into numerical values.

2. **Model Development**:
   - **Simple Model**: A basic neural network with a flatten layer and a dense output layer.
   - **Complex Model**: A more advanced neural network with hidden layers and dropout regularization, tuned using Keras Tuner.
   - **Convolutional Neural Network (CNN)**: A CNN model with convolutional and pooling layers, optimized using Keras Tuner.
   - **Pre-trained Model (MobileNetV2)**: A transfer learning approach using MobileNetV2 with a custom classification head.

3. **Model Evaluation**:
   - Evaluating the performance of each model on the test set.
   - Comparing the accuracy and loss of different models.
   - Analyzing the confusion matrix to identify the most challenging weather condition to classify.

## Results

### Model Performance

| Model               | Test Accuracy | Test Loss  |
|---------------------|---------------|------------|
| Simple Model        | 61.54%        | 12.27      |
| Complex Model       | 84.62%        | 0.48       |
| CNN Model           | 91.12%        | 0.24       |
| MobileNetV2         | 96.45%        | 0.11       |

### Key Findings

- **Best Performing Model**: The **MobileNetV2** model achieved the highest test accuracy of **96.45%**, outperforming the other models.
- **Most Challenging Weather Condition**: The **shine** class (class 3) was the most difficult to detect, with an accuracy of **91.43%**.

## Libraries Used

- **TensorFlow**: For building and training neural networks.
- **Keras Tuner**: For hyperparameter tuning.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib/Seaborn**: For data visualization.

## How to Run the Code

1. **Download the Dataset**:
   - Download the Mendeley Weather Dataset from [here](https://data.mendeley.com/datasets/4drtyfjtfy/1).
   - Unzip the dataset and place it in a folder named `dataset2` in the same directory as the Jupyter notebook.

2. **Install Dependencies**:
   - Ensure you have the required libraries installed. You can install them using:
     ```bash
     pip install tensorflow keras-tuner pandas numpy matplotlib seaborn
     ```

3. **Run the Jupyter Notebook**:
   - Open the Jupyter notebook and run the cells sequentially to load the data, train the models, and evaluate their performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset is provided by [Mendeley Data](https://data.mendeley.com/datasets/4drtyfjtfy/1).
- Special thanks to the TensorFlow and Keras communities for their extensive documentation and support.
