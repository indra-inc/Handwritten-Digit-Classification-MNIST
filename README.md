# Handwritten-Digit-Classification-MNIST
This a neural network implemented using TensorFlow that recognizes handwritten digits from 0-9 based on the MNIST dataset.
## About the MNIST Dataset:

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The MNIST database contains 60,000 training images and 10,000 testing images. The images are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker.


## Code Dependencies:
python 3.x with following modules installed

1. numpy
2. seaborn
3. tensorflow
4. keras
5. Pillow
6. opencv2

7. FastAPI
6. Uvicorn

* Run http://127.0.0.1:8000/ to see the final deployed model.

### API Testing:
Curl and Postman
![5](https://user-images.githubusercontent.com/97096513/210115894-d7eb01af-73dc-491c-b8d7-9f62390946be.png)
![4](https://user-images.githubusercontent.com/97096513/210115900-0c960c1a-05f3-47be-836d-647537e0867c.png)

### Execution:
Colab
Vscode
## Network Architecture:

![1](https://user-images.githubusercontent.com/97096513/210115853-c1419b91-3e71-46d4-93b0-43b2392aa577.png)
![MNIST](https://user-images.githubusercontent.com/97096513/210115884-4edc0e59-4909-452e-8433-679845e1f355.png)
![2](https://user-images.githubusercontent.com/97096513/210115859-6ddb14cb-268a-461f-8e6b-285b7a59a0db.png)
![3](https://user-images.githubusercontent.com/97096513/210115862-2cde1e54-7780-4757-b30a-e216ce8f58ac.png)


## Results:

After getting trained on a total of 42,000 images, a batch size of 128 and 20 epochs the model achieved an accuracy of 99.54%. 
Accuracy on the testing set was 99.26%. The model did not seem to overfit the traning data. 
Following are few snapshot made by the model:
