# Plant Disease Detection with Deep Learning

This project provides an end-to-end pipeline for detecting plant diseases from leaf images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It includes data preprocessing, model training, evaluation, and a user-friendly web interface for predictions using Streamlit.

## Features

- **Dataset Download & Preprocessing:**  
  Automatically downloads the PlantVillage dataset from Kaggle, unzips, and preprocesses the images for model training.

- **Model Training:**  
  Trains a CNN on the dataset to classify plant leaf images into various disease categories.

- **Model Evaluation:**  
  Evaluates the trained model and visualizes training/validation accuracy and loss.

- **Prediction System:**  
  Provides functions to preprocess images and predict their disease class.

- **Web Interface:**  
  Uses Streamlit to allow users to upload a plant leaf image and receive a disease prediction instantly.

- **Colab & Local Support:**  
  The code is compatible with Google Colab for training and can be run locally for inference.

## Project Structure

```
.
├── app.py
├── plant_disease_detection_project_file_.py
├── PLANT_DISEASE_DETECTION PROJECT File .ipynb
├── kaggle (2).json
├── project.py
```

- **app.py**: Streamlit web app for image upload and disease prediction.
- **plant_disease_detection_project_file_.py**: Main Colab notebook/script for data processing, model training, and evaluation.
- **PLANT_DISEASE_DETECTION PROJECT File .ipynb**: Jupyter notebook version of the project.
- **kaggle (2).json**: Kaggle API credentials (do not share publicly).
- **project.py**: (Empty placeholder for future expansion).

## How It Works

1. **Data Preparation**
    - Downloads the PlantVillage dataset from Kaggle.
    - Preprocesses images (resizing, normalization).
    - Splits data into training and validation sets.

2. **Model Training**
    - Defines a CNN architecture using Keras.
    - Trains the model on the preprocessed dataset.
    - Saves the trained model in `.keras` and `.h5` formats.
    - Saves class indices for mapping predictions to class names.

3. **Prediction**
    - Loads the trained model and class indices.
    - Preprocesses uploaded images to match training input.
    - Predicts the disease class of the uploaded image.

4. **Web Interface**
    - Users can upload a plant leaf image via the Streamlit app.
    - The app displays the uploaded image and the predicted disease class.

## Usage

### 1. Training (Google Colab)

- Open `plant_disease_detection_project_file_.py` or the Jupyter notebook in Colab.
- Ensure your Kaggle API key is available as `kaggle.json`.
- Run all cells to train the model and save the outputs.

### 2. Running the Web App (Locally)

1. Install dependencies:
    ```sh
    pip install streamlit tensorflow pillow numpy
    ```

2. Place the trained model (`plant_disease_prediction.keras`) and `class_indices.json` in your working directory.

3. Start the Streamlit app:
    ```sh
    streamlit run app.py
    ```

4. Open the provided local URL in your browser, upload a plant leaf image, and view the prediction.

## Example

![Streamlit UI Example](https://user-images.githubusercontent.com/your-username/your-image.png)

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Pillow
- Streamlit

## Notes

- The model expects images of size 224x224 pixels.
- The Kaggle API key file (`kaggle (2).json`) is required for dataset download.
- Do **not** share your Kaggle API key or model files publicly.

## License

This project is for educational and research purposes only.

---

**Author:**  
Jefranul Islam Rakib

---

> For questions or contributions, please open an issue or pull request on this repository.
