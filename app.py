# ==============================================================================
#       SnackTrack & JamHack - AI-Powered Web App Backend (Updated)
# ==============================================================================
import cv2
import numpy as np
import base64
import os
import json
import urllib.parse
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from deepface import DeepFace

# --- Load environment variables and configure the API ---
load_dotenv()
try:
    # Configure the generative AI model with the API key from environment variables
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    # Raise an error if the API key is not found in the .env file
    raise KeyError("GEMINI_API_KEY not found. Please set it in your .env file.")

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- AI-Powered Recommendation Function ---
def get_ai_recommendations(mood):
    """
    Calls the Gemini AI to get dynamic food and music recommendations,
    including a specific YouTube video ID.
    """
    # Initialize the generative model. 
    # Switched to 'gemini-1.0-pro' to help avoid free-tier rate limits.
    model = genai.GenerativeModel('gemini-1.0-pro')

    # This is the "prompt" we send to the AI. 
    # We now ask for a "youtube_video_id" to create an embeddable link.
    prompt = f"""
    You are SnackTrack & JamHack, an AI that pairs moods with snacks and music.
    A user is feeling "{mood}".
    
    Give them one specific, creative, and fun recommendation.
    
    Provide your response as a JSON object with the following keys:
    - "snack": A specific food item (e.g., "Gourmet Spicy Popcorn with Parmesan").
    - "song": The name of a specific song (e.g., "Bohemian Rhapsody").
    - "artist": The name of the song's artist (e.g., "Queen").
    - "message": A short, witty, and empathetic message for the user.
    - "emoji": A single emoji that represents the mood.
    - "youtube_video_id": A valid YouTube video ID for the recommended song (e.g., "fJ9rUzIMcZQ" for "Bohemian Rhapsody").
    
    Example for 'happy':
    {{
        "snack": "A scoop of rainbow sherbet ice cream",
        "song": "Good as Hell",
        "artist": "Lizzo",
        "message": "You're glowing! Here's a combo to match that brilliant energy.",
        "emoji": "😊",
        "youtube_video_id": "SmbmeOgWsqE"
    }}
    
    Now, generate a recommendation for the mood: "{mood}".
    """
    
    try:
        # Call the AI to generate content based on the prompt
        response = model.generate_content(prompt)
        
        # Clean up the response and parse it as JSON
        # The AI sometimes wraps its JSON in json ... , so we strip that out.
        json_text = response.text.strip().replace("json", "").replace("", "")
        recommendation = json.loads(json_text)
        print("AI Recommendation:", recommendation)
        return recommendation
        
    except Exception as e:
        print(f"AI generation or JSON parsing failed: {e}")
        # Fallback in case the AI fails, ensuring the app doesn't crash
        return {
            "snack": "A handful of almonds",
            "song": "Weightless",
            "artist": "Marconi Union",
            "message": "The AI is pondering... here is a neutral vibe for you.",
            "emoji": "🤔",
            "youtube_video_id": "UfcAVejslrU" # A fallback video ID
        }

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # This assumes you have an 'index.html' in a 'templates' folder
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Receives image data, analyzes mood, gets AI recommendations, and returns them."""
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        # Decode the base64 image string
        img_str = data['image'].split(',')[1]
        img_decoded = base64.b64decode(img_str)
        nparr = np.frombuffer(img_decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze the image for emotion using DeepFace
        analysis_result = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False # Set to False to avoid errors if no face is detected
        )

        dominant_mood = "neutral" # Default mood
        if analysis_result and len(analysis_result) > 0:
            dominant_mood = analysis_result[0]['dominant_emotion']
            
        # Get dynamic recommendations from the AI based on the detected mood
        recommendation = get_ai_recommendations(dominant_mood)
        
        # --- MODIFICATION START ---
        # Construct the direct embed and image URLs
        
        # Create the YouTube embed URL from the video ID provided by the AI
        video_id = recommendation.get("youtube_video_id")
        if video_id:
            recommendation['song_embed_url'] = f"https://www.youtube.com/embed/{video_id}"
        else:
            # Fallback to a search link if the AI fails to provide an ID
            song_query = f"{recommendation.get('song', '')} by {recommendation.get('artist', '')}"
            recommendation['song_embed_url'] = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(song_query)}"

        # Create the direct image URL for the snack using Unsplash
        food_query = recommendation.get('snack', 'food')
        recommendation['food_image_url'] = f"https://source.unsplash.com/500x300/?{urllib.parse.quote_plus(food_query)}"
        # --- MODIFICATION END ---

        # Prepare the final JSON response to send back to the frontend
        response_data = {
            "status": "success",
            "mood": dominant_mood,
            "recommendation": recommendation
        }
        print("Final Response to Frontend:", response_data)
        return jsonify(response_data)

    except Exception as e:
        # Generic error handling for the endpoint
        print(f"An error occurred in /analyze: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(debug=True)