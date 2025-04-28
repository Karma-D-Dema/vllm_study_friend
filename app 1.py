# app.py
from flask import Flask, render_template, request, jsonify, send_file
import torch
import os
import json
import re
import traceback
import subprocess
import tempfile
import io
from datetime import datetime
from transformers import pipeline, AutoTokenizer
import whisper
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from flask_cors import CORS  # Import CORS
from flask import request, jsonify

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Allow uploads up to 100 MB

# Initialize Mistral model only
MISTRAL_MODEL = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def check_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible in the system PATH."""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        return True
    except FileNotFoundError:
        return False


def transcribe_video(video_path):
    """
    Transcribe video using OpenAI's Whisper model.

    Args:
        video_path (str): Path to the video file

    Returns:
        str: Transcribed text from the video
    """
    try:
        # Check if FFmpeg is installed
        if not check_ffmpeg_installed():
            return """FFmpeg is not installed or not in your system PATH. 

Please install FFmpeg:
1. Download from https://ffmpeg.org/download.html
2. Add it to your system PATH
3. Restart your application

Alternatively, you can use our mock transcription function for testing."""

        # Step 1: Create a temporary file for the extracted audio
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "extracted_audio.mp3")

        # Step 2: Extract audio from video using ffmpeg subprocess
        print(f"Extracting audio from video: {video_path}")

        # More robust command that should work with a wider variety of videos
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'mp3',  # Use mp3 codec
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            audio_path
        ]

        # Run FFmpeg with verbose output for debugging
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"FFmpeg stderr: {process.stderr}")
            raise Exception(f"FFmpeg failed with error code {process.returncode}")

        # Check if audio was extracted successfully
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Failed to extract audio from video - output file is empty")

        # Step 3: Load Whisper model (can be "tiny", "base", "small", "medium", or "large")
        print("Loading Whisper model...")
        model_size = "base"

        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = whisper.load_model(model_size, device=device)

        # Step 4: Transcribe the audio
        print("Transcribing audio...")
        result = model.transcribe(audio_path)

        # Step 5: Clean up temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return result["text"]

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in transcription: {error_details}")
        # Return a friendly error message if transcription fails
        return f"Failed to transcribe video. Error: {str(e)}"


def generate_topic_wise_summary(text):
    """
    Generate a topic-wise summary of the transcript using Mistral model.

    Args:
        text (str): The transcript text

    Returns:
        dict: A dictionary with topics as keys and summaries as values
    """
    # Ensure we have text to process
    if not text or len(text.strip()) < 100:
        return {
            "title": "Video Content",
            "topics": [
                {
                    "heading": "Content Overview",
                    "summary": "The video content was too short or could not be properly transcribed for detailed topic analysis.",
                    "key_points": ["Insufficient content for detailed analysis"]
                }
            ]
        }

    prompt = f"""Analyze this educational content and organize it into 3-5 clear topics or sections.
For each topic:
1. Create a concise heading
2. Write a detailed summary of that section (100-150 words each)
3. Extract 2-3 key points or takeaways

Return the result in this exact JSON format:
{{
  "title": "Overall title of the content",
  "topics": [
    {{
      "heading": "Topic 1 Heading",
      "summary": "Detailed summary of this topic...",
      "key_points": ["Key point 1", "Key point 2", "Key point 3"]
    }},
    // more topics...
  ]
}}

Here is the content to analyze:
{text}"""

    try:
        response = MISTRAL_MODEL(
            prompt,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.5,
        )

        response_text = response[0]["generated_text"][len(prompt):]

        # Extract valid JSON using regex
        json_match = re.search(r'({[\s\S]*})', response_text)

        if not json_match:
            # Fallback structure if no JSON found
            default_summary = {
                "title": "Video Content Summary",
                "topics": [
                    {
                        "heading": "Main Content",
                        "summary": text[:500] + "..." if len(text) > 500 else text,
                        "key_points": ["Content could not be automatically organized into topics."]
                    }
                ]
            }
            return default_summary

        # Try to parse the extracted JSON
        extracted_json = json_match.group(1)
        try:
            parsed_json = json.loads(extracted_json)

            # Validate the structure
            if "title" not in parsed_json:
                parsed_json["title"] = "Video Content Summary"

            if "topics" not in parsed_json or not parsed_json["topics"]:
                parsed_json["topics"] = [{
                    "heading": "Main Content",
                    "summary": text[:500] + "..." if len(text) > 500 else text,
                    "key_points": ["Automated topic identification was not successful."]
                }]

            # Ensure each topic has all required fields
            for topic in parsed_json["topics"]:
                if "heading" not in topic:
                    topic["heading"] = "Untitled Section"
                if "summary" not in topic:
                    topic["summary"] = "No summary available."
                if "key_points" not in topic or not topic["key_points"]:
                    topic["key_points"] = ["No key points identified."]

            return parsed_json

        except json.JSONDecodeError:
            print("JSON parsing error in topic summary generation")
            # Return fallback on JSON parse error
            return {
                "title": "Video Content Summary",
                "topics": [
                    {
                        "heading": "Main Content",
                        "summary": text[:500] + "..." if len(text) > 500 else text,
                        "key_points": ["Could not parse the AI-generated topic structure."]
                    }
                ]
            }
    except Exception as e:
        print(f"Error generating topic-wise summary: {traceback.format_exc()}")
        # Return a valid fallback structure
        return {
            "title": "Video Content Summary",
            "topics": [
                {
                    "heading": "Main Content",
                    "summary": text[:500] + "..." if len(text) > 500 else text,
                    "key_points": ["Error generating topic summary."]
                }
            ]
        }


def generate_pdf_summary(transcript, topic_summary, video_filename):
    """
    Generate a PDF with topic-wise summary.

    Args:
        transcript (str): The full transcript
        topic_summary (dict): Topic-wise summary dictionary
        video_filename (str): Name of the original video file

    Returns:
        BytesIO: PDF file as bytes buffer
    """
    print(f"Starting PDF generation for {video_filename}")
    buffer = io.BytesIO()

    try:
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get sample stylesheet - already contains Heading1, Heading2, etc.
        styles = getSampleStyleSheet()

        # Modify existing styles instead of adding new ones
        styles['Heading1'].textColor = colors.darkblue
        styles['Heading1'].fontSize = 18
        styles['Heading1'].spaceAfter = 12

        styles['Heading2'].textColor = colors.darkblue
        styles['Heading2'].fontSize = 14
        styles['Heading2'].spaceAfter = 8

        # Only add custom styles that don't exist in the sample stylesheet
        keyPointStyle = ParagraphStyle(
            name='KeyPoint',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leftIndent=20,
            leading=14
        )

        # Build PDF content
        elements = []

        # Document title
        title = topic_summary.get("title", "Video Content Summary")
        elements.append(Paragraph(title, styles['Heading1']))

        # Video filename and date
        elements.append(Paragraph(f"Video: {video_filename}", styles['Normal']))
        current_date = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(f"Generated on: {current_date}", styles['Normal']))
        elements.append(Spacer(1, 0.25 * inch))

        # Add table of contents
        elements.append(Paragraph("Table of Contents", styles['Heading2']))
        topics = topic_summary.get("topics", [])
        if topics:
            for i, topic in enumerate(topics):
                elements.append(Paragraph(f"{i + 1}. {topic.get('heading', 'Untitled Topic')}", styles['Normal']))
        else:
            elements.append(Paragraph("1. Main Content", styles['Normal']))
        elements.append(Spacer(1, 0.25 * inch))

        # Add topics
        if topics:
            for i, topic in enumerate(topics):
                heading = topic.get('heading', f"Topic {i + 1}")
                summary = topic.get('summary', "No summary available.")
                key_points = topic.get('key_points', ["No key points available."])

                elements.append(Paragraph(f"{i + 1}. {heading}", styles['Heading2']))
                elements.append(Paragraph(summary, styles['Normal']))
                elements.append(Spacer(1, 0.1 * inch))

                # Add key points
                elements.append(Paragraph("Key Points:", keyPointStyle))
                for point in key_points:
                    bullet_text = f"• {point}"
                    elements.append(Paragraph(bullet_text, styles['Normal']))

                elements.append(Spacer(1, 0.2 * inch))
        else:
            elements.append(Paragraph("1. Main Content", styles['Heading2']))
            elements.append(
                Paragraph(transcript[:1000] + "..." if len(transcript) > 1000 else transcript, styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))

        # Add full transcript at the end
        elements.append(Paragraph("Full Transcript", styles['Heading2']))

        # Break transcript into manageable chunks to avoid ReportLab issues with very long paragraphs
        transcript_chunks = [transcript[i:i + 2000] for i in range(0, len(transcript), 2000)]
        for chunk in transcript_chunks:
            elements.append(Paragraph(chunk, styles['Normal']))
            elements.append(Spacer(1, 0.1 * inch))

        # Build the PDF
        print("Building PDF document...")
        doc.build(elements)
        print("PDF generation complete")
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Error in PDF generation: {str(e)}")
        print(traceback.format_exc())

        # Create a simple error PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        elements = []
        elements.append(Paragraph("Error Generating PDF", styles['Title']))
        elements.append(Paragraph(f"There was an error generating the PDF: {str(e)}", styles['Normal']))
        elements.append(Paragraph("Please try again or contact support.", styles['Normal']))

        doc.build(elements)
        buffer.seek(0)
        return buffer


def generate_summary(text):
    prompt = f"""Summarize this educational content into key points:
{text}
Summary:"""

    try:
        response = MISTRAL_MODEL(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        summary = response[0]["generated_text"][len(prompt):]

        # Ensure we have a valid summary
        if not summary or len(summary.strip()) < 10:
            return "Could not generate a summary for this content."

        return summary
    except Exception as e:
        print(f"Error generating summary: {traceback.format_exc()}")
        return f"Error generating summary: {str(e)}"


def generate_quiz(text):
    prompt = f"""Create a 5-question quiz based on this content:
{text}
Format as JSON with this exact structure:
{{
    "quiz": [
        {{
            "question": "...",
            "options": ["...", "...", "...", "..."],
            "answer": 0
        }}
    ]
}}"""

    try:
        response = MISTRAL_MODEL(
            prompt,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.5,
        )

        response_text = response[0]["generated_text"][len(prompt):]

        # Extract valid JSON using regex to find JSON object
        json_match = re.search(r'({[\s\S]*})', response_text)

        if not json_match:
            # Fallback to a basic quiz structure if no JSON found
            default_quiz = {
                "quiz": [
                    {
                        "question": "No questions could be generated from the content.",
                        "options": ["Try again", "Upload different content", "Contact support", "Check transcript"],
                        "answer": 0
                    }
                ]
            }
            return json.dumps(default_quiz)

        # Try to parse the extracted JSON
        extracted_json = json_match.group(1)
        parsed_json = json.loads(extracted_json)

        # Validate the structure
        if "quiz" not in parsed_json or not isinstance(parsed_json["quiz"], list):
            raise ValueError("Invalid quiz structure")

        return json.dumps(parsed_json)
    except Exception as e:
        print(f"Error generating quiz: {traceback.format_exc()}")
        # Return a valid fallback JSON on error
        default_quiz = {
            "quiz": [
                {
                    "question": "Error generating quiz. Please try again.",
                    "options": ["Retry", "Upload different content", "Check transcript", "Contact support"],
                    "answer": 0
                }
            ]
        }
        return json.dumps(default_quiz)


def process_video(video_path, video_filename=None):
    """
    Process a video file to extract transcript, generate summary, topic summary, and quiz.

    Args:
        video_path (str): Path to the video file
        video_filename (str, optional): Original filename of the video

    Returns:
        dict: Results including transcript, summary, topic summary, and quiz
    """
    try:
        # If video_filename is None, extract it from the path
        if video_filename is None:
            video_filename = os.path.basename(video_path)

        # Get absolute path
        absolute_path = os.path.abspath(video_path)

        print(f"Processing video at path: {absolute_path}")

        # Check if file exists
        if not os.path.exists(absolute_path):
            return {
                "error": f"File not found at path: {absolute_path}",
                "transcript": "",
                "summary": "",
                "topic_summary": {"title": "Error", "topics": []},
                "quiz": json.dumps({"quiz": []})
            }

        # Use real transcription
        transcript = transcribe_video(absolute_path)

        if not transcript or len(transcript.strip()) < 10:
            return {
                "error": "Could not transcribe video or transcription too short",
                "transcript": transcript or "",
                "summary": "",
                "topic_summary": {"title": "Error", "topics": []},
                "quiz": json.dumps({"quiz": []})
            }

        # Generate summary
        summary = generate_summary(transcript)

        # Generate topic-wise summary for PDF
        topic_summary = generate_topic_wise_summary(transcript)

        # Generate quiz
        quiz = generate_quiz(transcript)

        return {
            "transcript": transcript,
            "summary": summary,
            "topic_summary": topic_summary,
            "quiz": quiz,
            "video_filename": video_filename
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error processing video: {error_details}")
        return {
            "error": f"Error processing video: {str(e)}",
            "transcript": "",
            "summary": "",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze_video():
    print("=== /analyze endpoint called ===")
    print(f"Request method: {request.method}")
    print(f"Request content type: {request.content_type}")
    print(f"Files in request: {list(request.files.keys()) if request.files else 'None'}")

    if 'video' not in request.files:
        print("Error: No file uploaded")
        return jsonify({
            "error": "No file uploaded",
            "transcript": "",
            "summary": "",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }), 400

    video_file = request.files['video']
    print(f"Received file: {video_file.filename}, content type: {video_file.content_type}")

    if video_file.filename == '':
        print("Error: Empty filename")
        return jsonify({
            "error": "Empty filename",
            "transcript": "",
            "summary": "",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)

    # Ensure directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Save the file
    video_file.save(save_path)

    print(f"File saved to: {save_path}")

    try:
        # Make sure we pass both parameters
        results = process_video(save_path, video_file.filename)

        # Ensure all required fields are present
        if "error" in results and results["error"]:
            # If there's an error but the response is missing any required fields, add them
            if "transcript" not in results:
                results["transcript"] = ""
            if "summary" not in results:
                results["summary"] = ""
            if "topic_summary" not in results:
                results["topic_summary"] = {"title": "Error", "topics": []}
            if "quiz" not in results:
                results["quiz"] = json.dumps({"quiz": []})

            return jsonify(results), 500

        # Make sure all required fields are present in successful responses too
        if "topic_summary" not in results:
            results["topic_summary"] = generate_topic_wise_summary(results.get("transcript", ""))

        return jsonify(results)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in analyze_video: {error_details}")
        return jsonify({
            "error": str(e),
            "transcript": "",
            "summary": "Error processing video",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        if not data:
            print("Error: No JSON data received in request")
            return jsonify({"error": "No data received"}), 400

        transcript = data.get('transcript', '')
        topic_summary = data.get('topic_summary', {})
        video_filename = data.get('video_filename', 'video.mp4')

        print(f"PDF generation request received for video: {video_filename}")
        print(f"Topic summary has {len(topic_summary.get('topics', []))} topics")

        if not transcript:
            print("Error: Missing transcript")
            return jsonify({"error": "Missing transcript data"}), 400

        if not topic_summary or not isinstance(topic_summary, dict):
            print("Error: Missing or invalid topic summary")
            # Create a basic topic summary to avoid errors
            topic_summary = {
                "title": "Video Summary",
                "topics": [
                    {
                        "heading": "Content Summary",
                        "summary": transcript[:500] + "..." if len(transcript) > 500 else transcript,
                        "key_points": ["Automatically generated summary"]
                    }
                ]
            }

        # Generate PDF
        print("Generating PDF...")
        pdf_buffer = generate_pdf_summary(transcript, topic_summary, video_filename)

        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"summary_{timestamp}.pdf"

        # Save PDF to temporary file
        temp_dir = tempfile.gettempdir()
        temp_pdf_path = os.path.join(temp_dir, pdf_filename)
        print(f"Saving PDF to: {temp_pdf_path}")

        # Reset buffer position and write to file
        pdf_buffer.seek(0)
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_buffer.getbuffer())

        print(f"PDF successfully generated and saved to {temp_pdf_path}")

        # Return the path and filename
        return jsonify({
            "success": True,
            "pdf_path": temp_pdf_path,
            "filename": pdf_filename
        })
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error generating PDF: {error_details}")
        return jsonify({
            "error": f"Error generating PDF: {str(e)}"
        }), 500


@app.route('/download-pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    try:
        temp_dir = tempfile.gettempdir()
        temp_pdf_path = os.path.join(temp_dir, filename)
        print(f"Download request for PDF: {temp_pdf_path}")

        if not os.path.exists(temp_pdf_path):
            print(f"Error: PDF file not found at {temp_pdf_path}")
            return jsonify({"error": "PDF file not found"}), 404

        print(f"Sending file: {temp_pdf_path}")

        # Using Flask 2.x compatible parameters
        return send_file(
            temp_pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error downloading PDF: {error_details}")
        return jsonify({
            "error": f"Error downloading PDF: {str(e)}"
        }), 500


@app.route('/chat', methods=['POST'])
def chat_response():
    """
    Process user questions about video content and return relevant responses.

    Expected JSON input:
    {
        "query": "User's question text",
        "context": {
            "transcript": "Video transcript",
            "summary": "Video summary",
            "topic_summary": {topic summary object},
            "quiz": "quiz JSON string"
        }
    }

    Returns JSON:
    {
        "response": "Assistant's response"
    }
    """
    try:
        # Get data from request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract the query and content context
        query = data.get('query', '')
        context = data.get('context', {})

        if not query:
            return jsonify({"error": "No question provided"}), 400

        # Extract context components
        transcript = context.get('transcript', '')
        summary = context.get('summary', '')
        topic_summary = context.get('topic_summary', {})
        quiz_json = context.get('quiz', '{}')

        # Create a rich context for the AI model to use when generating a response
        combined_context = f"""TRANSCRIPT:
{transcript[:3000]}  # Limit transcript length to avoid token limits

SUMMARY:
{summary}

TOPIC BREAKDOWN:
"""

        # Add topics to context
        topics = topic_summary.get('topics', [])
        for i, topic in enumerate(topics):
            combined_context += f"""Topic {i + 1}: {topic.get('heading', '')}
Summary: {topic.get('summary', '')}
Key Points: {', '.join(topic.get('key_points', []))}

"""

        # Create prompt for the model
        prompt = f"""You are a helpful learning assistant that answers questions about educational video content.
Use ONLY the information provided in the context to answer the question. If the answer cannot be derived from the provided context, acknowledge that you don't have enough information.

CONTEXT:
{combined_context}

QUESTION: {query}

Your response should be helpful, concise, and directly address the question. If appropriate, cite specific parts of the video content.
"""

        # Generate response using Mistral model
        response = MISTRAL_MODEL(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
        )

        # Extract response text
        response_text = response[0]["generated_text"][len(prompt):]

        # Clean up response text - remove any extra prefixes like "RESPONSE:" or "ANSWER:"
        response_text = re.sub(r'^(RESPONSE:|ANSWER:|Assistant:|A:|>|\s)+', '', response_text,
                               flags=re.IGNORECASE).strip()

        return jsonify({"response": response_text})

    except Exception as e:
        print(f"Error in chat response: {traceback.format_exc()}")
        return jsonify({"error": f"Error processing your question: {str(e)}"}), 500


if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)