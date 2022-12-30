# 1. Library imports
import uvicorn
from tensorflow.keras.utils import load_img
from fastapi import FastAPI
from keras.models import load_model
import numpy as np

# 2. Create the app object
app = FastAPI()

## Loading the model
final_model = load_model('mnist.h5')


# 3. Index main route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Handwritten Digit Recognition'}

# 3. Expose the prediction functionality

@app.get('/{img_name}')
def predict_digit(img_name:str):

    try:
        img_path=(f'./images/{img_name}')

        img = load_img(img_path)
        #resize image to 28x28 pixels
        img = img.resize((28,28))
        #convert rgb to grayscale
        img = img.convert('L')
        img = np.array(img)
        #reshaping to support our model input and normalizing
        img = img.reshape(1,28,28,1).astype('float32')
        img = img/255.0
        #predicting the class
        res = final_model.predict([img])[0]
        a = np.argmax(res), 
        b = max(res)
        return {f"Predicted Image is: {a} and Maximum Probablity: {b}"}
    except FileNotFoundError as e:
        return(f"FileNotFoundError:{e}")


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn main:app --reload
#http://127.0.0.1:8000/docs