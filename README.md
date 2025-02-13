# **Dog Identifier Project**

## **Project Overview**
The Dog Identifier Project uses machine learning to classify breeds of domestic and wild dogs. Our goal was to create a practical and engaging AI application while refining our model training and optimization skills.

## **Technologies Used**
- **Models:** ResNet & MobileNet (pretrained)
- **Data Management:** Pandas, DataFrame stored as CSV
- **Libraries:** Pandas, OS, XML.etree.ElementTree, PIL, NumPy, TensorFlow
- **Dataset:** [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## **Data Preparation**
- **Dataset Acquisition:** Used the Stanford Dogs Dataset containing labeled images of various breeds.
- **Annotation Parsing:** Extracted labels from XML files with `xml.etree.ElementTree`.
- **Dataframe Construction:** Created a CSV containing image paths, labels, and bounding boxes.
- **Preprocessing:** Applied image resizing to `(224, 224)` and normalized pixel values to a `[0, 1]` range.
- **Splitting:** Divided data into training (80%) and validation (20%) sets.

## **Model Performance Summary**
| ID | Model Type            | Accuracy | Loss   | Notes                                          |
|----|-----------------------|----------|------- |------------------------------------------------|
| 1  | MobileNet (Baseline)  | 72%      | 0.75   | Baseline with no tuning                        |
| 2  | MobileNet (Tuned)     | 75%      | 0.68   | Added early stopping                          |
| 3  | MobileNet (Optimized) | 78%      | 0.65   | Applied L2 regularization, increased dropout  |
| 4  | ResNet50 (Attempt #1) | 80.04%   | 65.09  | Used 2 dense layers with 1 dropout layer      |
| 5  | ResNet50 (Attempt #2) | N/A      | >1     | Attempt failed (Conv2D incompatible)          |
| 6  | ResNet50 (Attempt #3) | 80%      | 65     | Added class weights for imbalance correction  |

## **How to Use the Flask Web App**
### **1. Install Dependencies:**
```sh
pip install -r requirements.txt
```
### **2. Run the Flask App:**
```sh
python flask_code.py
```
### **3. Access the App:**
Visit in your browser:
```
http://127.0.0.1:5000
```
### **4. Upload an Image:**
Use the web interface to upload an image and receive a breed prediction.

## **Next Steps**
- Refine hyperparameters to improve model accuracy
- Increase training set size or apply data augmentation
- Enhance the Flask app with clearer outputs and error handling

## **Final Thoughts**
This project was a valuable exercise in transfer learning, model tuning, and deploying an interactive web application. While model accuracy plateaued, the insights gained will guide future improvements.

