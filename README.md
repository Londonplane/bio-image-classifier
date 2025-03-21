# Bio Image Classifier

A web application that classifies images into three categories: people, animals, and plants. The application uses Clarifai API for image recognition and provides a user-friendly interface for uploading and viewing classification results.

## Features

- Multiple image upload support
- Real-time image preview
- Image removal before upload
- Classification into three categories:
  - People
  - Animals
  - Plants
- Confidence score display
- Responsive design
- Error handling and user feedback

## Project Structure

```
bio-image-classifier/
├── app.py              # Main Flask application
├── model.py            # Image classification model
├── requirements.txt    # Python dependencies
├── uploads/           # Directory for uploaded images
└── templates/         # HTML templates
    ├── index.html     # Upload page
    └── result.html    # Results display page
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your Clarifai API key:
```
CLARIFAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and visit:
```
http://localhost:5000
```

## Usage

1. Click "选择文件" to select one or more images
2. Preview selected images
3. Remove unwanted images using the × button
4. Click "上传并分类" to process the images
5. View classification results with confidence scores

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF

## Error Handling

The application provides feedback for:
- No file selected
- Unsupported file types
- Processing errors
- API failures

## Dependencies

- Flask
- Pillow
- numpy
- requests
- scipy
- python-dotenv 