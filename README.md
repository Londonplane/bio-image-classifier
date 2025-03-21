# Bio Image Classifier

A simple web application that can classify images into three categories: Human, Animal, or Plant.

## Features

- **Image Classification**:
  - Uses Clarifai API for deep learning based classification
  - Has local backup system when API is unavailable
- **User-Friendly Interface**:
  - Easy to use web interface
  - Upload images and see results instantly
- **High Accuracy**:
  - Uses pre-trained deep learning models
  - Smart rules for better results

## How to Install

1. Clone this repository:
   ```bash
   git clone https://github.com/Londonplane/bio-image-classifier.git
   cd bio-image-classifier
   ```

2. Create virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   # OR
   venv\Scripts\activate  # For Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your Clarifai API key in `.env`:
     ```
     CLARIFAI_API_KEY=your_api_key_here
     ```
   - Get your API key from [Clarifai website](https://clarifai.com/)

## How to Use

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

3. Upload an image and click "Submit"
4. See the classification result

## Project Structure

- `app.py`: Main Flask application
- `model.py`: Image classification logic
- `templates/`: HTML files
- `uploads/`: Folder for uploaded images
- `requirements.txt`: Required Python packages

## Notes

1. **API Limits**:
   - Free Clarifai account has API call limits
   - Local system works when API is not available

2. **API Key Safety**:
   - Keep your API key secret
   - Never share your `.env` file

3. **Best Practices**:
   - Use clear images
   - Make sure main object is visible
   - For people: include face if possible
   - For animals: choose clear features
   - For plants: prefer green plants

## Future Plans

1. Add more categories
2. Add user feedback system
3. Add image pre-processing
4. Add batch processing
5. Make mobile-friendly

## License

[MIT License](LICENSE) 