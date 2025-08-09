# ==============================================================================
#      SnackTrack & JamHack - AI-Powered Web App Backend
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
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise KeyError("GEMINI_API_KEY not found. Please set it in your .env file.")

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- AI-Powered Recommendation Function ---
def get_ai_recommendations(mood):
    """
    Calls the Gemini AI to get dynamic food and music recommendations.
    """
    # Initialize the generative model
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # This is the "prompt" we send to the AI. Notice how we ask for a JSON response.
    # This makes the AI's answer easy and reliable to use in our code.
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
    
    Example for 'happy':
    {{
        "snack": "A scoop of rainbow sherbet ice cream",
        "song": "Good as Hell",
        "artist": "Lizzo",
        "message": "You're glowing! Here's a combo to match that brilliant energy.",
        "emoji": "😊"
    }}
    
    Now, generate a recommendation for the mood: "{mood}".
    """
   
    try:
        # Call the AI
        response = model.generate_content(prompt)
        
        # Clean up the response and parse it as JSON
        # The AI sometimes wraps its JSON in ```json ... ```, so we strip that out.
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        recommendation = json.loads(json_text)
        print(recommendation)
        return recommendation
        
    except Exception as e:
        print(f"AI generation or JSON parsing failed: {e}")
        # Fallback in case the AI fails
        return {
            "snack": "A handful of almonds",
            "song": "Weightless",
            "artist": "Marconi Union",
            "message": "The AI is pondering... here is a neutral vibe for you.",
            "emoji": ""
        }

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Receives image data, analyzes mood, gets AI recommendations, and returns them."""
    try:
        data = request.get_json()
        img_str = data['image'].split(',')[1]
        img_decoded = base64.b64decode(img_str)
        nparr = np.frombuffer(img_decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze the image for emotion
        analysis_result = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            enforce_detection=False
        )

        dominant_mood = "neutral" # Default mood
        if analysis_result and len(analysis_result) > 0:
            dominant_mood = analysis_result[0]['dominant_emotion']
            
        # Get dynamic recommendations from the AI
        recommendation = get_ai_recommendations(dominant_mood)
        
        # Create the YouTube and Unsplash URLs
        song_query = f"{recommendation['song']} by {recommendation['artist']}"
        recommendation['song_link'] = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(song_query)}"
        
        food_query = recommendation['snack']
        recommendation['food_image_url'] = f"https://source.unsplash.com/500x300/?{urllib.parse.quote_plus(food_query)}"

        response_data = {
            "status": "success",
            "mood": dominant_mood,
            "recommendation": recommendation
        }
        print(response_data)
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred in /analyze: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)