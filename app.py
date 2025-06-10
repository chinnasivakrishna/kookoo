import json
import asyncio
import numpy as np
import wave
import struct
import io
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
from scipy.io.wavfile import write
import openai
import httpx
from deepgram import DeepgramClient, PrerecordedOptions, LiveTranscriptionEvents, LiveOptions
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# API Keys (Replace with your actual keys)
OPENAI_API_KEY = "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"
DEEPGRAM_API_KEY = "b40137a84624ef9677285b9c9feb3d1f3e576417"
LMNT_API_KEY = "e2b3ccd7d3ca4654a590309ac32320a5"

# Initialize clients
openai.api_key = OPENAI_API_KEY
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# Store active connections and their states
active_connections: Dict[str, WebSocket] = {}
connection_states: Dict[str, dict] = {}

class AICallHandler:
    def __init__(self, connection_id: str, websocket: WebSocket):
        self.connection_id = connection_id
        self.websocket = websocket
        self.audio_buffer = []
        self.conversation_history = [
            {"role": "system", "content": "You are a helpful AI assistant on a phone call. Keep responses concise and conversational, suitable for voice interaction. Be friendly and professional."}
        ]
        self.is_speaking = False
        self.silence_start_time = None
        self.min_silence_duration = 1.5  # seconds
        
    async def process_audio_chunk(self, audio_data):
        """Process incoming audio chunk"""
        try:
            # Add to buffer
            self.audio_buffer.extend(audio_data['samples'])
            
            # Check for silence detection (simple amplitude-based)
            avg_amplitude = np.mean(np.abs(audio_data['samples']))
            
            if avg_amplitude < 100:  # Silence threshold
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                elif time.time() - self.silence_start_time > self.min_silence_duration:
                    # Process accumulated audio
                    await self.process_speech()
                    self.silence_start_time = None
            else:
                self.silence_start_time = None
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def process_speech(self):
        """Process accumulated speech and generate response"""
        if not self.audio_buffer:
            return
            
        try:
            # Convert audio buffer to WAV format for Deepgram
            audio_array = np.array(self.audio_buffer, dtype=np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(8000)  # 8kHz
                wav_file.writeframes(audio_array.tobytes())
            
            wav_buffer.seek(0)
            audio_data = wav_buffer.read()
            
            # Clear buffer
            self.audio_buffer = []
            
            # Transcribe with Deepgram
            transcript = await self.transcribe_audio(audio_data)
            
            if transcript and transcript.strip():
                logger.info(f"User said: {transcript}")
                
                # Generate AI response
                ai_response = await self.generate_ai_response(transcript)
                logger.info(f"AI response: {ai_response}")
                
                # Convert to speech and send
                await self.text_to_speech_and_send(ai_response)
                
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio using Deepgram"""
        try:
            options = PrerecordedOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                diarize=False,
            )
            
            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_data, "mimetype": "audio/wav"}, options
            )
            
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            return transcript
            
        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            return ""
    
    async def generate_ai_response(self, user_message: str) -> str:
        """Generate AI response using OpenAI"""
        try:
            self.conversation_history.append({"role": "user", "content": user_message})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history,
                max_tokens=150,
                temperature=0.7
            )
            
            ai_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_message})
            
            return ai_message
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I'm sorry, I didn't catch that. Could you please repeat?"
    
    async def text_to_speech_and_send(self, text: str):
        """Convert text to speech using LMNT and send via WebSocket"""
        try:
            # LMNT API call
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.lmnt.com/v1/ai/speech",
                    headers={
                        "Authorization": f"Bearer {LMNT_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": text,
                        "voice": "lily",  # Choose appropriate voice
                        "format": "wav",
                        "sample_rate": 8000
                    }
                )
                
                if response.status_code == 200:
                    audio_content = response.content
                    await self.send_audio_to_caller(audio_content)
                else:
                    logger.error(f"LMNT API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
    
    async def send_audio_to_caller(self, audio_data: bytes):
        """Send audio data back to caller via WebSocket"""
        try:
            # Parse WAV file
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                
            # Convert to int16 samples
            samples = struct.unpack(f"<{len(frames)//2}h", frames)
            
            # Send in chunks of 80 samples (10ms at 8kHz)
            chunk_size = 80
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                
                if len(chunk) == chunk_size:
                    message = {
                        "type": "media",
                        "ucid": self.connection_id,
                        "data": {
                            "samples": list(chunk),
                            "bitsPerSample": 16,
                            "sampleRate": 8000,
                            "channelCount": 1,
                            "numberOfFrames": len(chunk),
                            "type": "data"
                        }
                    }
                    
                    await self.websocket.send_text(json.dumps(message))
                    await asyncio.sleep(0.01)  # 10ms delay between chunks
                    
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    connection_id = str(id(websocket))
    active_connections[connection_id] = websocket
    
    # Initialize AI call handler
    ai_handler = AICallHandler(connection_id, websocket)
    connection_states[connection_id] = {
        "handler": ai_handler,
        "start_time": time.time()
    }
    
    logger.info(f"New connection: {connection_id}")
    
    # Send welcome message
    welcome_message = "Hello! I'm your AI assistant. How can I help you today?"
    await ai_handler.text_to_speech_and_send(welcome_message)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                json_data = json.loads(data)
                
                # Handle different message types
                if json_data.get("event") == "start":
                    logger.info(f"Call started for {connection_id}")
                    
                elif json_data.get("event") == "stop":
                    logger.info(f"Call ended for {connection_id}")
                    break
                    
                elif json_data.get("type") == "media":
                    # Process audio data
                    await ai_handler.process_audio_chunk(json_data["data"])
                    
                else:
                    logger.info(f"Received: {json_data}")
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if connection_id in active_connections:
            del active_connections[connection_id]
        if connection_id in connection_states:
            del connection_states[connection_id]
        logger.info(f"Connection {connection_id} cleaned up")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_connections": len(active_connections)}

# Test endpoint to check API keys
@app.get("/test-apis")
async def test_apis():
    results = {}
    
    # Test OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        results["openai"] = "✓ Working"
    except Exception as e:
        results["openai"] = f"✗ Error: {str(e)}"
    
    # Test Deepgram (basic connection test)
    try:
        # This is a simple connection test
        dg_client = DeepgramClient(DEEPGRAM_API_KEY)
        results["deepgram"] = "✓ Client initialized"
    except Exception as e:
        results["deepgram"] = f"✗ Error: {str(e)}"
    
    # Test LMNT
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.lmnt.com/v1/ai/voices",
                headers={"Authorization": f"Bearer {LMNT_API_KEY}"}
            )
            if response.status_code == 200:
                results["lmnt"] = "✓ Working"
            else:
                results["lmnt"] = f"✗ HTTP {response.status_code}"
    except Exception as e:
        results["lmnt"] = f"✗ Error: {str(e)}"
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

from fastapi import FastAPI, Request
import json

# Add this to your main app.py or create a separate route handler

@app.route('/ivr', methods=['GET', 'POST'])
async def handle_ivr_request(request: Request):
    """Handle incoming IVR requests and return appropriate XML responses"""
    
    # Get request parameters
    event = request.query_params.get('event', '')
    did = request.query_params.get('did', '')
    caller_id = request.query_params.get('cid', '')
    
    print(f"IVR Event: {event}, DID: {did}, Caller ID: {caller_id}")
    
    if event == "NewCall":
        # Convert all request params to dictionary for UUI
        all_params = dict(request.query_params)
        uui_json = json.dumps(all_params)
        
        # Return XML to start WebSocket stream
        # Replace YOUR_WEBSOCKET_URL with your actual WebSocket server URL
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <start-record/>
    <stream is_sip="true" url="ws://YOUR_SERVER_IP:8080/ws" x-uui="{uui_json}">{did}</stream>
</response>'''
    
    elif event == "Stream":
        # Handle stream events if needed
        return '''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <hangup/>
</response>'''
    
    elif event == "Hangup":
        # Handle call hangup
        return '''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <hangup/>
</response>'''
    
    else:
        # Default response
        return '''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <hangup/>
</response>'''