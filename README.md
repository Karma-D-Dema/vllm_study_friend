# Video Learning Assistant

An AI-powered web application that transforms educational videos into structured learning materials by generating transcripts, summaries, topic breakdowns, quizzes, and providing quality verification with an interactive chat assistant.

## Features

* **Automatic Video Transcription** - Converts speech to text using OpenAI's Whisper model with 95%+ accuracy
* **Dual Input Support** - Upload video files or process videos from URLs (including YouTube)
* **AI-Powered Content Generation** - Creates summaries, topic breakdowns, and quizzes using Mistral-7B
* **Quality Verification** - ROUGE metrics automatically assess content accuracy and flag potential issues
* **PDF Summary Export** - Download professionally formatted study materials
* **Interactive Learning Chat** - Ask follow-up questions about video content
* **Multi-Format Support** - Handles MP4, AVI, MOV, WMV, MKV, and direct video URLs

## System Architecture

### Core Components
* **Frontend**: HTML/CSS/JavaScript interface for uploads and results
* **Audio Processing**: FFmpeg extracts and normalizes audio
* **Speech-to-Text**: OpenAI Whisper model for transcription
* **Content Analysis**: Mistral-7B for summaries, topics, quiz generation, and chat
* **Quality Evaluation**: ROUGE metrics for automated content verification
* **PDF Generator**: ReportLab creates downloadable study notes
* **Web Framework**: Flask backend for request handling

### AI Models Used
* **Mistral-7B-Instruct-v0.2** - Text generation, summarization, and question answering
* **OpenAI Whisper (base model)** - Speech recognition and transcription
* **ROUGE Evaluation System** - Content quality assessment and hallucination detection

## Installation

### Prerequisites

* Python 3.8+
* FFmpeg installed on your system
* GPU recommended for faster processing (CUDA support)
* At least 8GB RAM for optimal performance

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Karma-D-Dema/vllm_study_friend.git
cd vllm_study_friend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### FFmpeg Installation

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Usage

### Basic Usage

1. **Start the application:**
```bash
python app.py
```

2. **Access the web interface:**
   - Open your browser and go to `http://localhost:5000`

3. **Process a video:**
   - **File Upload**: Click "Upload File" and select a video
   - **URL Processing**: Click "From URL", enter a video URL, and validate

4. **Review results:**
   - View transcript, summary, and topic breakdowns
   - Take the generated quiz
   - Chat with the AI assistant about the content
   - Download PDF study materials

### Supported Input Formats

**File Uploads:**
- Video: MP4, AVI, MOV, WMV, MKV
- Audio: MP3, WAV, M4A
- Maximum file size: 100MB

**URL Processing:**
- YouTube videos
- Direct video links (.mp4, .avi, .mov, etc.)
- Educational platform videos

## Project Structure

```
vllm_study_friend/
├── app.py                          # Main Flask application
├── rouge_bleu_evaluation.py        # Quality assessment system
├── templates/
│   └── upload.html                 # Frontend interface
├── uploads/                        # Temporary file storage
├── requirements.txt                # Python dependencies
├── content_verification.log        # Quality assessment logs
└── README.md                       # This file
```

## Performance Metrics

Based on evaluation of 50 educational videos:

* **Transcription Accuracy**: 95%+ across diverse accents and speakers
* **Content Quality**: 38% high quality, 60% moderate quality, 2% requiring review
* **Processing Speed**: 2-5 minutes for typical 10-minute educational video
* **Quality Verification**: 62% of content flagged for human review (conservative approach)

## API Endpoints

* `GET /` - Main upload interface
* `POST /analyze` - Process uploaded video file
* `POST /analyze-url` - Process video from URL
* `POST /validate-video-url` - Validate video URL before processing
* `POST /generate-pdf` - Create PDF summary
* `GET /download-pdf/<filename>` - Download generated PDF
* `POST /chat` - Interactive chat with AI assistant

## Requirements

```txt
flask==2.3.3
flask-cors==4.0.0
torch>=2.0.0
transformers>=4.30.0
openai-whisper>=20231117
reportlab>=4.0.0
nltk>=3.8.0
numpy>=1.24.0
requests>=2.31.0
yt-dlp>=2023.7.6
```

## Configuration

### Environment Variables (Optional)

```bash
export FLASK_ENV=development  # For development mode
export UPLOAD_FOLDER=uploads  # Custom upload directory
export MAX_CONTENT_LENGTH=104857600  # Max file size (100MB)
```

### Model Configuration

Models are automatically downloaded on first use:
- Mistral-7B (~13GB)
- Whisper base model (~140MB)

## Limitations

* Limited to clear audio quality videos
* No visual content analysis (slides, diagrams)
* Processing time increases with video length
* 100MB file size limit for uploads

## Future Enhancements

* Visual content analysis using computer vision
* GPU acceleration for faster processing
* Subject-specific model fine-tuning
* Cloud deployment with load balancing
* Advanced semantic evaluation metrics

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Update documentation if needed
5. Commit your changes: `git commit -m 'Add feature description'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## Troubleshooting

### Common Issues

**FFmpeg not found:**
- Ensure FFmpeg is installed and in system PATH
- Restart terminal/command prompt after installation

**Out of memory errors:**
- Reduce video length or resolution
- Close other applications to free RAM
- Use GPU if available

**Slow processing:**
- Enable GPU acceleration if available
- Process shorter video segments
- Reduce video quality before upload

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* OpenAI for the Whisper speech recognition model
* Mistral AI for the Mistral-7B language model
* The open-source community for supporting libraries and tools

## Citation

If you use this project in your research, please cite:

```bibtex
@software{vllm_study_friend,
  title={Video Learning Assistant: AI-Powered Educational Content Processing},
  author={Karma Dechen Dema},
  year={2025},
  url={https://github.com/Karma-D-Dema/vllm_study_friend}
}
```