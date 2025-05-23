# app.py
from flask import Flask, render_template, request, jsonify, send_file
import torch
import os
import json
import re
import traceback
import subprocess
import io
from datetime import datetime
import time
from transformers import pipeline, AutoTokenizer
import whisper
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from flask_cors import CORS  # Import CORS
from flask import request, jsonify
from rouge_bleu_evaluation import ContentVerificationSystem, verify_content_accuracy
import json
import requests
import urllib.parse
from urllib.parse import urlparse, unquote
import tempfile
import shutil
import yt_dlp


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
            "quiz": json.dumps({"quiz": []})}


def process_video_with_verification(video_path, video_filename=None):
    """
    Enhanced version of process_video that includes automatic content verification
    """
    try:
        print(f"Processing video with verification: {video_filename}")

        # ✅ FIXED: Call the ORIGINAL process_video function, not itself
        results = process_video(video_path, video_filename)  # ← CHANGED THIS LINE

        if "error" in results and results["error"]:
            return results

        print("Video processing completed, starting content verification...")

        # Perform automatic content verification
        verification = verify_content_accuracy(
            results.get("transcript", ""),
            results.get("summary", ""),
            results.get("topic_summary", {})
        )

        # Add verification results to response
        results["content_verification"] = verification

        # Add quality flags based on verification
        results["quality_flags"] = generate_quality_flags(verification)

        # Log the verification results
        log_verification_results(video_filename, verification)

        print("Content verification completed")
        return results

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in enhanced video processing: {error_details}")
        return {
            "error": f"Error processing video with verification: {str(e)}",
            "transcript": "",
            "summary": "",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }


def generate_quality_flags(verification_result):
    """
    Generate quality flags based on verification results
    """
    flags = {
        "overall_quality": "unknown",
        "summary_quality": "unknown",
        "topic_quality": "unknown",
        "needs_review": False,
        "potential_hallucinations": False,
        "recommendations": []
    }

    try:
        # Check overall assessment
        overall = verification_result.get("overall_assessment", {})
        quality_status = overall.get("quality_status", "UNKNOWN")

        if quality_status == "HIGH_QUALITY":
            flags["overall_quality"] = "high"
        elif quality_status == "MODERATE_QUALITY":
            flags["overall_quality"] = "moderate"
        else:
            flags["overall_quality"] = "low"
            flags["needs_review"] = True
            flags["potential_hallucinations"] = True

        # Check summary quality
        summary_eval = verification_result.get("summary_verification", {}).get("summary_evaluation", {})
        summary_status = summary_eval.get("content_verification", {}).get("status", "UNKNOWN")

        if summary_status == "VERIFIED":
            flags["summary_quality"] = "verified"
        elif summary_status == "PARTIALLY_VERIFIED":
            flags["summary_quality"] = "partial"
        else:
            flags["summary_quality"] = "unverified"
            flags["potential_hallucinations"] = True

        # Check topic quality
        topic_eval = verification_result.get("topic_verification", {})
        avg_scores = topic_eval.get("average_topic_scores", {})
        avg_rouge1 = avg_scores.get("avg_rouge_1_f1", 0)

        if avg_rouge1 >= 0.25:
            flags["topic_quality"] = "verified"
        elif avg_rouge1 >= 0.15:
            flags["topic_quality"] = "partial"
        else:
            flags["topic_quality"] = "unverified"
            flags["potential_hallucinations"] = True

        # Generate recommendations
        if flags["potential_hallucinations"]:
            flags["recommendations"].append("Manual review recommended - potential inaccuracies detected")

        if flags["needs_review"]:
            flags["recommendations"].append("Content partially verified - cross-check important claims")

        if flags["overall_quality"] == "high":
            flags["recommendations"].append("Content appears accurate and well-supported by source material")

    except Exception as e:
        print(f"Error generating quality flags: {e}")
        flags["recommendations"].append("Error analyzing content quality")

    return flags


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






# At the very end of your file, before if __name__ == '__main__':
# Add this to verify routes are registered:
print("🚀 Registering Flask routes...")
print(f"📍 Available routes will be:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.endpoint}: {rule.rule} [{','.join(rule.methods)}]")


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


def verify_content():
    """
    Verify content accuracy using ROUGE/BLEU metrics
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract content from request
        transcript = data.get('transcript', '')
        summary = data.get('summary', '')
        topic_summary = data.get('topic_summary', {})

        if not transcript:
            return jsonify({"error": "Transcript is required for verification"}), 400

        print(f"Starting content verification...")
        print(f"Transcript length: {len(transcript)} characters")
        print(f"Summary length: {len(summary)} characters")
        print(f"Topics count: {len(topic_summary.get('topics', []))}")

        # Perform verification
        verification_result = verify_content_accuracy(transcript, summary, topic_summary)

        print("Verification completed successfully")
        return jsonify(verification_result)

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in content verification: {error_details}")
        return jsonify({
            "error": f"Error verifying content: {str(e)}",
            "details": error_details
        }), 500


# Additional utility functions for logging and monitoring
def log_verification_results(video_filename, verification_results):
    """
    Log verification results for monitoring and analysis
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "video_filename": video_filename,
            "overall_quality": verification_results.get("overall_assessment", {}).get("quality_status", "Unknown"),
            "summary_rouge_1": verification_results.get("summary_verification", {}).get("summary_evaluation", {}).get(
                "rouge_1", {}).get("f1", 0),
            "summary_rouge_2": verification_results.get("summary_verification", {}).get("summary_evaluation", {}).get(
                "rouge_2", {}).get("f1", 0),
            "avg_topic_rouge_1": verification_results.get("topic_verification", {}).get("average_topic_scores", {}).get(
                "avg_rouge_1_f1", 0),
            "needs_review": verification_results.get("overall_assessment", {}).get("quality_status") in [
                "MODERATE_QUALITY", "LOW_QUALITY"]
        }

        # Log to file (you can also log to database)
        log_file = "content_verification.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Verification results logged for {video_filename}")

    except Exception as e:
        print(f"Error logging verification results: {e}")


def get_verification_statistics():
    """
    Get statistics from verification logs
    """
    try:
        stats = {
            "total_processed": 0,
            "high_quality": 0,
            "moderate_quality": 0,
            "low_quality": 0,
            "avg_rouge_1": 0,
            "avg_rouge_2": 0,
            "needs_review_count": 0
        }

        log_file = "content_verification.log"
        if not os.path.exists(log_file):
            return stats

        rouge_1_scores = []
        rouge_2_scores = []

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    stats["total_processed"] += 1

                    quality = entry.get("overall_quality", "Unknown")
                    if quality == "HIGH_QUALITY":
                        stats["high_quality"] += 1
                    elif quality == "MODERATE_QUALITY":
                        stats["moderate_quality"] += 1
                    elif quality == "LOW_QUALITY":
                        stats["low_quality"] += 1

                    if entry.get("needs_review", False):
                        stats["needs_review_count"] += 1

                    rouge_1_scores.append(entry.get("summary_rouge_1", 0))
                    rouge_2_scores.append(entry.get("summary_rouge_2", 0))

                except json.JSONDecodeError:
                    continue

        if rouge_1_scores:
            stats["avg_rouge_1"] = round(sum(rouge_1_scores) / len(rouge_1_scores), 4)
        if rouge_2_scores:
            stats["avg_rouge_2"] = round(sum(rouge_2_scores) / len(rouge_2_scores), 4)

        return stats

    except Exception as e:
        print(f"Error getting verification statistics: {e}")
        return {"error": str(e)}


@app.route('/verification-stats', methods=['GET'])
def verification_statistics():
    """
    Get verification statistics endpoint
    """
    try:
        stats = get_verification_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def download_video_from_url(video_url, max_size_mb=100):
    """
    Download video from URL and save to temporary file
    """
    try:
        print(f"Downloading video from URL: {video_url}")

        # Parse URL to get filename
        parsed_url = urlparse(video_url)
        original_filename = unquote(parsed_url.path.split('/')[-1])

        # If no filename in URL, generate one
        if not original_filename or '.' not in original_filename:
            original_filename = f"video_{int(time.time())}.mp4"  # ← This should work now

        # Rest of the function stays the same...
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"download_{original_filename}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        with requests.get(video_url, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_size_mb:
                    return None, None, False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"

            # Download file
            with open(temp_file_path, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Check size during download
                        if downloaded_size > max_size_mb * 1024 * 1024:
                            f.close()
                            os.remove(temp_file_path)
                            return None, None, False, f"File too large during download"

        print(f"Successfully downloaded: {temp_file_path}")
        return temp_file_path, original_filename, True, None

    except requests.exceptions.RequestException as e:
        return None, None, False, f"Download failed: {str(e)}"
    except Exception as e:
        return None, None, False, f"Unexpected error: {str(e)}"


def download_youtube_audio(youtube_url, max_duration_minutes=20):
    """
    Download audio only from YouTube - Fast and reliable
    """
    try:
        print(f"🎵 Downloading audio from YouTube: {youtube_url}")

        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        output_template = os.path.join(temp_dir, f'yt_audio_{timestamp}.%(ext)s')

        # Audio-only options - much faster and more reliable
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',  # Audio only, any format
            'outtmpl': output_template,
            'max_filesize': 25 * 1024 * 1024,  # 25MB limit for audio
            'extract_flat': False,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            # Speed optimizations
            'quiet': True,
            'no_warnings': True,
            'retries': 1,
            'socket_timeout': 20,
            'http_chunk_size': 512 * 1024,  # 512KB chunks
        }

        print("📡 Extracting video information...")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(youtube_url, download=False)

            # Check duration
            duration = info.get('duration', 0)
            title = info.get('title', 'YouTube Audio')[:50]  # Limit title length

            print(f"📹 Video: {title}")
            print(f"⏱️ Duration: {duration // 60}:{duration % 60:02d}")

            if duration > max_duration_minutes * 60:
                return None, None, False, f"Audio too long: {duration // 60}min (max: {max_duration_minutes}min)"

            # Download audio
            print("⬬ Starting audio download...")
            start_time = time.time()

            ydl.download([youtube_url])

            download_time = time.time() - start_time
            print(f"✅ Audio download completed in {download_time:.1f} seconds")

            # Find the downloaded audio file
            import glob
            pattern = os.path.join(temp_dir, f'yt_audio_{timestamp}.*')
            matches = glob.glob(pattern)

            if matches:
                audio_file = matches[0]
                file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
                print(f"📁 Audio file: {audio_file}")
                print(f"💾 Size: {file_size:.1f}MB")

                return audio_file, title, True, None
            else:
                print("❌ Audio file not found after download")
                return None, None, False, "Audio file not found after download"

    except Exception as e:
        error_msg = str(e)
        print(f"❌ YouTube audio download error: {error_msg}")

        # Provide helpful error messages
        if "Video unavailable" in error_msg:
            return None, None, False, "Video is private, deleted, or not available in your region"
        elif "Sign in to confirm your age" in error_msg:
            return None, None, False, "Video is age-restricted. Try a different video"
        elif "Private video" in error_msg:
            return None, None, False, "Video is private. Try a public video"
        else:
            return None, None, False, f"Audio download failed: {error_msg}"


def is_youtube_url(url):
    """Check if URL is a YouTube URL"""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    parsed = urlparse(url)
    return any(domain in parsed.netloc.lower() for domain in youtube_domains)


def is_valid_video_url(url):
    """Basic validation for video URLs"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            return False
        if not parsed.netloc:
            return False

        # Check for common video file extensions in URL
        video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']
        url_lower = url.lower()

        # YouTube URLs are valid even without file extensions
        if is_youtube_url(url):
            return True

        # Check if URL ends with video extension
        return any(url_lower.endswith(ext) for ext in video_extensions)

    except Exception:
        return False


# New Flask route for URL processing
@app.route('/analyze-url', methods=['POST'])
@app.route('/analyze-url', methods=['POST'])
def analyze_video_from_url():
    """
    Analyze video from URL - Now using audio-only for YouTube
    """
    try:
        data = request.json
        if not data or 'video_url' not in data:
            return jsonify({
                "error": "No video URL provided",
                "transcript": "",
                "summary": "",
                "topic_summary": {"title": "Error", "topics": []},
                "quiz": json.dumps({"quiz": []})
            }), 400

        video_url = data['video_url'].strip()

        # Validate URL
        if not is_valid_video_url(video_url):
            return jsonify({
                "error": "Invalid video URL format",
                "transcript": "",
                "summary": "",
                "topic_summary": {"title": "Error", "topics": []},
                "quiz": json.dumps({"quiz": []})
            }), 400

        print(f"🎯 Processing URL: {video_url}")

        # Process based on URL type
        if is_youtube_url(video_url):
            # Use audio-only processing for YouTube
            print("🎵 Using audio-only processing for YouTube")
            results = process_youtube_audio(video_url)
        else:
            # Use video download for direct URLs
            print("🎬 Using video download for direct URL")
            temp_file_path, filename, success, error_msg = download_video_from_url(video_url)

            if not success:
                return jsonify({
                    "error": f"Failed to download video: {error_msg}",
                    "transcript": "",
                    "summary": "",
                    "topic_summary": {"title": "Error", "topics": []},
                    "quiz": json.dumps({"quiz": []})
                }), 400

            try:
                results = process_video_with_verification(temp_file_path, filename)

                # Clean up
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            except Exception as e:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                raise e

        # Add source URL to results
        results["source_url"] = video_url
        if "source_type" not in results:
            results["source_type"] = "youtube" if is_youtube_url(video_url) else "direct_url"

        # Add verification if not already present
        if "content_verification" not in results and "error" not in results:
            print("📊 Adding content verification...")
            verification = verify_content_accuracy(
                results.get("transcript", ""),
                results.get("summary", ""),
                results.get("topic_summary", {})
            )
            results["content_verification"] = verification
            results["quality_flags"] = generate_quality_flags(verification)

        return jsonify(results)

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Error in analyze_video_from_url: {error_details}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "transcript": "",
            "summary": "",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }), 500


def convert_audio_for_whisper(audio_path):
    """
    Convert downloaded audio to format suitable for Whisper
    """
    try:
        print(f"🔄 Converting audio for Whisper: {audio_path}")

        # Create output path for converted audio
        temp_dir = tempfile.gettempdir()
        converted_path = os.path.join(temp_dir, "whisper_audio.wav")

        # Use FFmpeg to convert to WAV format (best for Whisper)
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            converted_path
        ]

        print("🎛️ Converting audio format...")
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            print(f"FFmpeg conversion error: {process.stderr}")
            # If conversion fails, try using original file
            print("⚠️ Conversion failed, using original audio file")
            return audio_path

        print(f"✅ Audio converted successfully: {converted_path}")
        return converted_path

    except Exception as e:
        print(f"⚠️ Audio conversion error: {e}, using original file")
        return audio_path


def transcribe_youtube_audio(audio_path):
    """
    Transcribe audio using Whisper (optimized for YouTube audio)
    """
    try:
        print(f"🎤 Transcribing audio: {audio_path}")

        # Convert audio if needed
        whisper_audio_path = convert_audio_for_whisper(audio_path)

        # Load Whisper model
        print("🧠 Loading Whisper model...")
        model_size = "base"  # Good balance of speed and accuracy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {device}")

        model = whisper.load_model(model_size, device=device)

        # Transcribe the audio
        print("📝 Starting transcription...")
        start_time = time.time()

        result = model.transcribe(whisper_audio_path, language="en")  # Specify English for speed

        transcription_time = time.time() - start_time
        print(f"✅ Transcription completed in {transcription_time:.1f} seconds")

        # Clean up converted audio file if it's different from original
        if whisper_audio_path != audio_path and os.path.exists(whisper_audio_path):
            os.remove(whisper_audio_path)
            print("🗑️ Cleaned up converted audio file")

        transcript_text = result["text"].strip()
        print(f"📄 Transcript length: {len(transcript_text)} characters")

        return transcript_text

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Transcription error: {error_details}")
        return f"Failed to transcribe audio: {str(e)}"


def process_youtube_audio(youtube_url):
    """
    Complete pipeline: Download YouTube audio -> Transcribe -> Generate content
    """
    try:
        print(f"🚀 Starting YouTube audio processing pipeline for: {youtube_url}")

        # Step 1: Download audio
        audio_path, title, success, error_msg = download_youtube_audio(youtube_url)

        if not success:
            return {
                "error": f"Audio download failed: {error_msg}",
                "transcript": "",
                "summary": "",
                "topic_summary": {"title": "Error", "topics": []},
                "quiz": json.dumps({"quiz": []})
            }

        print(f"✅ Audio downloaded: {title}")

        try:
            # Step 2: Transcribe audio
            transcript = transcribe_youtube_audio(audio_path)

            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"🗑️ Cleaned up audio file: {audio_path}")

            if not transcript or len(transcript.strip()) < 10:
                return {
                    "error": "Could not transcribe audio or transcription too short",
                    "transcript": transcript or "",
                    "summary": "",
                    "topic_summary": {"title": "Error", "topics": []},
                    "quiz": json.dumps({"quiz": []})
                }

            # Step 3: Generate summary
            print("📝 Generating summary...")
            summary = generate_summary(transcript)

            # Step 4: Generate topic-wise summary
            print("📋 Generating topic summary...")
            topic_summary = generate_topic_wise_summary(transcript)

            # Step 5: Generate quiz
            print("❓ Generating quiz...")
            quiz = generate_quiz(transcript)

            print("🎉 YouTube audio processing completed successfully!")

            return {
                "transcript": transcript,
                "summary": summary,
                "topic_summary": topic_summary,
                "quiz": quiz,
                "video_filename": f"{title}.mp3",
                "source_type": "youtube_audio",
                "processing_method": "audio_only"
            }

        except Exception as e:
            # Clean up audio file on error
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

            error_details = traceback.format_exc()
            print(f"❌ Processing error: {error_details}")
            return {
                "error": f"Error processing audio: {str(e)}",
                "transcript": "",
                "summary": "",
                "topic_summary": {"title": "Error", "topics": []},
                "quiz": json.dumps({"quiz": []})
            }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Pipeline error: {error_details}")
        return {
            "error": f"YouTube processing failed: {str(e)}",
            "transcript": "",
            "summary": "",
            "topic_summary": {"title": "Error", "topics": []},
            "quiz": json.dumps({"quiz": []})
        }


# Update your analyze_video_from_url route to use audio processing

# Route to validate video URL before processing
@app.route('/validate-video-url', methods=['POST'])
def validate_video_url():
    """
    Validate if a video URL is processable
    """
    try:
        data = request.json
        if not data or 'video_url' not in data:
            return jsonify({"valid": False, "error": "No URL provided"}), 400

        video_url = data['video_url'].strip()

        # Basic validation
        if not is_valid_video_url(video_url):
            return jsonify({
                "valid": False,
                "error": "Invalid video URL format",
                "supported_formats": ["Direct video URLs (.mp4, .avi, .mov, etc.)", "YouTube URLs"]
            })

        # Check if URL is accessible (HEAD request)
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

            if is_youtube_url(video_url):
                # For YouTube, just check if it's a valid format
                return jsonify({
                    "valid": True,
                    "url_type": "youtube",
                    "note": "YouTube URL detected - will use yt-dlp for download"
                })
            else:
                # For direct URLs, check accessibility
                response = requests.head(video_url, headers=headers, timeout=10)
                content_type = response.headers.get('content-type', '')
                content_length = response.headers.get('content-length')

                size_info = ""
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    size_info = f" (Size: {size_mb:.1f}MB)"

                return jsonify({
                    "valid": True,
                    "url_type": "direct",
                    "content_type": content_type,
                    "note": f"Direct video URL{size_info}"
                })

        except requests.exceptions.RequestException as e:
            return jsonify({
                "valid": False,
                "error": f"URL not accessible: {str(e)}"
            })

    except Exception as e:
        return jsonify({
            "valid": False,
            "error": f"Validation error: {str(e)}"
        }), 500


# Combined route that handles both file upload and URL
@app.route('/analyze-video', methods=['POST'])
def analyze_video_combined():
    """
    Combined endpoint that handles both file uploads and URLs
    """
    try:
        # Check if it's a file upload
        if 'video' in request.files:
            # Handle file upload (existing functionality)
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({"error": "Empty filename"}), 400

            save_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            video_file.save(save_path)

            results = process_video_with_verification(save_path, video_file.filename)
            results["source_type"] = "file_upload"
            return jsonify(results)

        # Check if it's a URL request
        elif request.is_json:
            data = request.json
            if 'video_url' in data:
                # Handle URL (redirect to URL handler)
                return analyze_video_from_url()

        return jsonify({"error": "No video file or URL provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/debug-routes', methods=['GET'])
def debug_routes():
    """Debug route to see all available routes"""
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        line = urllib.parse.unquote("{:50s} {:20s} {}".format(rule.endpoint, methods, rule))
        output.append(line)

    return jsonify({
        "total_routes": len(output),
        "routes": output
    })


if __name__ == '__main__':
    # Test imports
    try:
        print("✅ Testing imports...")
        print(f"time.time() = {time.time()}")
        print(f"datetime.now() = {datetime.now()}")
        print("✅ All imports working")
    except Exception as e:
        print(f"❌ Import error: {e}")

    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    print("🌐 Starting Flask app on http://localhost:5000")
    print("🔍 Test endpoints:")
    print("  GET  http://localhost:5000/test-connection")
    print("  GET  http://localhost:5000/debug-routes")
    print("  POST http://localhost:5000/analyze-url")

    app.run(host='0.0.0.0', port=5000, debug=True)