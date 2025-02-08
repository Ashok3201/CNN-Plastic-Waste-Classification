# CNN Model for Plastic Waste Classification

<h1 align="center">Hi there, I'm Ashok Kumar</h1>

---


## Overview  
This project focuses on building a Convolutional Neural Network (CNN) model to classify images of plastic waste into various categories. The primary goal is to enhance waste management systems by improving the segregation and recycling process using deep learning technologies.  

---

## Table of Contents  
- [Project Description](#project-description)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Weekly Progress](#weekly-progress)  
- [How to Run](#how-to-run)  
- [Technologies Used](#technologies-used)  
- [Future Scope](#future-scope)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Description  
Plastic pollution is a growing concern globally, and effective waste segregation is critical to tackling this issue. This project employs a CNN model to classify plastic waste into distinct categories, facilitating automated waste management.  

## Dataset  
The dataset used for this project is the **Waste Classification Data** by Sashaank Sekar. It contains a total of 25,077 labeled images, divided into two categories: **Organic** and **Recyclable**. This dataset is designed to facilitate waste classification tasks using machine learning techniques.  


### Key Details:
- **Total Images**: 25,077  
  - **Training Data**: 22,564 images (85%)  
  - **Test Data**: 2,513 images (15%)  
- **Classes**: Organic and Recyclable  
- **Purpose**: To aid in automating waste management and reducing the environmental impact of improper waste disposal.
  
### Approach:  
- Studied waste management strategies and white papers.  
- Analyzed the composition of household waste.  
- Segregated waste into two categories (Organic and Recyclable).  
- Leveraged IoT and machine learning to automate waste classification.  

### Dataset Link:  
You can access the dataset here: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data).  

*Note: Ensure appropriate dataset licensing and usage guidelines are followed.*  


## Model Architecture  
The CNN architecture includes:  
- **Convolutional Layers:** Feature extraction  
- **Pooling Layers:** Dimensionality reduction  
- **Fully Connected Layers:** Classification  
- **Activation Functions:** ReLU and Softmax  

### Basic CNN Architecture  
Below is a visual representation of the CNN architecture used in this project:  

<p align="center">
  <img src="https://github.com/Ashok3201/CNN-Plastic-Waste-Classification/blob/main/images/CNN-Architecture.jpg" alt="Basic CNN Architecture" style="width:80%;">
</p>

## Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** Configurable (default: 25)  
- **Batch Size:** Configurable (default: 32)  

Data augmentation techniques were utilized to enhance model performance and generalizability.  

## Weekly Progress

### Week 1:
#### Objective:
* Set up the initial framework for a CNN model to classify waste into Organic and Recyclable categories.

#### Activities:
* Installed essential libraries and ensured TensorFlow was functional.
* Collected images for training and testing from specified paths.
* Gathered images using `glob` and labeled them accordingly.
* Stored image data and labels in a pandas DataFrame.

#### Outcome:
* Imported all necessary libraries and prepared a dataset of 22,564 images.
* Created a balanced dataset with 12,565 Organic and 9,999 Recyclable images.

- **Notebooks:**  
  - [Week1-Libraries-Importing-Data-Setup.ipynb](wasteclassification.ipynb)
    
### Week 2:
#### Objective:
* Visualize the dataset and build a Convolutional Neural Network (CNN) model.

#### Activities:
* Visualized sample images with their labels using `matplotlib`.
* Built a Sequential CNN model with multiple layers (Convolutional, Activation, MaxPooling, Flatten, Dense, Dropout).
* Compiled the model with `binary_crossentropy` loss and `adam` optimizer.
* Created training and testing data generators with rescaling using `ImageDataGenerator`.
* Trained the CNN model for 10 epochs and validated the performance.

#### Outcome:
* Successfully visualized the dataset and built a CNN model.
* Achieved a training accuracy of 97.35% and a validation accuracy of 89.97% after 10 epochs.
* Evaluated the model on test data with an accuracy of 87.50%.

- **Notebooks:**  
  - [Week2-build-Convolutional-Neural-Network-(CNN)-model.ipynb](wasteclassification.ipynb)
    
### Week 3 (Final):
#### Objective:
* Evaluate model performance with new metrics and improve prediction function.

#### Activities:
* Plotted training and validation accuracy over epochs using `matplotlib`.
* Plotted training and validation loss over epochs using `matplotlib`.
* Improved prediction function to display images and predictions.

#### Outcome:
* Successfully evaluated model performance with new metrics.
* Enhanced the prediction function to provide visual outputs.
* Verified the improved model's accuracy on new test images with accurate predictions.

- **Notebooks:**  
  - [Week3-prediction-function and test-predictions.ipynb](wasteclassification.ipynb)
## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Ashok3201/CNN-Plastic-Waste-Classification
   cd CNN-Plastic-Waste-Classification
   ```  
2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the training script:  *Details to be added after completion.*  
   ```bash  
   python train.py  
   ```  
4. For inference, use the following command:  *Details to be added after completion.*  
   ```bash  
   python predict.py --image_path /path/to/image.jpg  
   ```  

## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  

## Future Scope  
- Expanding the dataset to include more plastic waste categories.  
- Deploying the model as a web or mobile application for real-time use.  
- Integration with IoT-enabled waste management systems.  

## Contributing  
Contributions are welcome! If you would like to contribute, please open an issue or submit a pull request.  

## License  

