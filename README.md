# Video Learning Assistant

An AI-powered web application that enhances educational videos by generating transcripts, summaries, topic breakdowns, quizzes, and an interactive chat assistant.

## Features

* **Automatic Video Transcription** - Converts speech to text using OpenAI's Whisper model
* **AI-Powered Summaries** - Generates concise summaries of video content
* **Topic-Wise Breakdown** - Organizes content into logical topics with key points
* **Interactive Quiz Generation** - Creates quiz questions based on video content
* **PDF Summary Export** - Download organized notes as PDF
* **Learning Assistant Chat** - Ask questions about the video content

## Technologies Used

* **Backend**: Flask (Python)
* **AI Models**:
  * Mistral 7B for text generation and analysis
  * Whisper for audio transcription
* **Frontend**: HTML, CSS, JavaScript
* **Dependencies**:
  * FFmpeg for audio extraction
  * PyTorch
  * Transformers
  * ReportLab for PDF generation

## Installation

### Prerequisites

* Python 3.8+
* FFmpeg installed on your system
* GPU recommended for faster processing

```bash
# Clone the repository
git clone https://github.com/yourusername/video-learning-assistant.git
cd video-learning-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to http://localhost:5000

3. Upload an educational video

4. Wait for processing to complete (will take a few minutes depending on video length)

5. Explore the generated transcript, summary, topics, and quiz

6. Use the chat assistant to ask questions about the video content

7. Download a PDF summary if needed

## Project Structure

* `app.py` - Main Flask application with routes and AI functionality
* `templates/upload.html` - Frontend interface
* `uploads/` - Directory for temporary video storage

## Requirements

The requirements.txt file should include:
```
flask
flask-cors
torch
transformers
whisper
reportlab
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

[MIT](https://choosealicense.com/licenses/mit/)
