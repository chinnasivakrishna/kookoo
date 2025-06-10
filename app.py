# main.py
import os
import json
import asyncio
import logging
from typing import Dict, List
from collections import deque
import numpy as np
from scipy.io.wavfile import write
import struct
import wave
import io
import base64
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import Response
import uvicorn

# AI Services
import openai
from deepgram import DeepgramClient, PrerecordedOptions, LiveOptions
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = "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"
DEEPGRAM_API_KEY = "b40137a84624ef9677285b9c9feb3d1f3e576417"
LMNT_API_KEY = "e2b3ccd7d3ca4654a590309ac32320a5"

# Initialize clients
openai.api_key = OPENAI_API_KEY
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

app = FastAPI(title="AI Telephonic System")

class AudioBuffer:
    def __init__(self):
        self.buffer = []
        self.sample_rate = 8000
        self.chunk_duration = 1.0  # seconds
        self.samples_per_chunk = int(self.sample_rate * self.chunk_duration)
    
    def add_samples(self, samples):
        self.buffer.extend(samples)
    
    def get_chunk_if_ready(self):
        if len(self.buffer) >= self.samples_per_chunk:
            chunk = self.buffer[:self.samples_per_chunk]
            self.buffer = self.buffer[self.samples_per_chunk:]
            return chunk
        return None
    
    def get_all_samples(self):
        return self.buffer.copy()

class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """You are a helpful AI assistant for phone calls. 
        Keep responses concise and natural for phone conversation. 
        Ask clarifying questions when needed and be friendly and professional."""
    
    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        # Keep only last 10 messages to manage context
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    async def get_ai_response(self, user_message: str) -> str:
        try:
            self.add_message("user", user_message)
            
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            self.add_message("assistant", ai_response)
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "I'm sorry, I'm having trouble processing your request right now."

class TTSManager:
    def __init__(self):
        self.lmnt_api_key = LMNT_API_KEY
        self.voice_id = "lily"  # Default LMNT voice
    
    async def text_to_speech(self, text: str) -> bytes:
        try:
            url = "https://api.lmnt.com/v1/ai/speech"
            headers = {
                "Authorization": f"Bearer {self.lmnt_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "voice": self.voice_id,
                "format": "wav",
                "sample_rate": 8000
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload)
                
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"LMNT API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def wav_to_samples(self, wav_data: bytes, target_sample_rate: int = 8000) -> List[int]:
        try:
            # Read WAV data
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    sample_width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    
                    # Convert to samples
                    if sample_width == 2:  # 16-bit
                        samples = struct.unpack(f'<{len(frames)//2}h', frames)
                    else:
                        # Convert other formats to 16-bit
                        samples = np.frombuffer(frames, dtype=np.int8)
                        samples = (samples.astype(np.float32) / 128.0 * 32767).astype(np.int16)
                    
                    # Convert stereo to mono if needed
                    if channels == 2:
                        samples = samples[::2]  # Take every other sample
                    
                    return list(samples)
                    
        except Exception as e:
            logger.error(f"WAV conversion error: {e}")
            return []

class STTManager:
    def __init__(self):
        self.deepgram = deepgram
    
    async def speech_to_text(self, audio_samples: List[int], sample_rate: int = 8000) -> str:
        try:
            # Convert samples to WAV bytes
            wav_data = self.samples_to_wav(audio_samples, sample_rate)
            
            options = PrerecordedOptions(
                model="nova-2",
                language="en",
                smart_format=True,
                punctuate=True
            )
            
            response = await self.deepgram.listen.asyncprerecorded.v("1").transcribe_url(
                {"buffer": wav_data, "mimetype": "audio/wav"},
                options
            )
            
            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip()
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return ""
    
    def samples_to_wav(self, samples: List[int], sample_rate: int) -> bytes:
        try:
            # Convert samples to numpy array
            audio_array = np.array(samples, dtype=np.int16)
            
            # Create WAV file in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_array.tobytes())
            
            wav_io.seek(0)
            return wav_io.read()
            
        except Exception as e:
            logger.error(f"WAV creation error: {e}")
            return b""

# Global managers
active_connections: Dict[str, WebSocket] = {}
connection_managers: Dict[str, dict] = {}

@app.get("/")
async def root():
    return {"message": "AI Telephonic System is running"}

@app.route("/ozonetel", methods=['GET', 'POST'])
async def handle_ozonetel_request(request: Request):
    try:
        # Get all query parameters
        params = dict(request.query_params)
        event = params.get('event', '')
        
        logger.info(f"Received event: {event}, params: {params}")
        
        if event == "NewCall":
            # Convert all request args to a dictionary and convert to JSON string
            uui_json = json.dumps(params)
            
            # Get the base URL from the request
            base_url = str(request.base_url).rstrip('/')
            websocket_url = base_url.replace('http', 'ws') + "/ws"
            
            # Return XML response to start streaming
            xml_response = f'''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <start-record/>
    <stream is_sip="true" url="{websocket_url}" x-uui="{uui_json}">+918049250961</stream>
</response>'''
            
            return Response(content=xml_response, media_type="application/xml")
            
        elif event == "Stream":
            # Handle stream event
            xml_response = '''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <cctransfer record="" moh="default" uui="sales" timeout="30" ringType="ring">general</cctransfer>
</response>'''
            return Response(content=xml_response, media_type="application/xml")
        
        else:
            # Default response
            xml_response = '''<?xml version="1.0" encoding="UTF-8"?>
<response>
    <hangup/>
</response>'''
            return Response(content=xml_response, media_type="application/xml")
            
    except Exception as e:
        logger.error(f"Error handling Ozonetel request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(id(websocket))
    active_connections[connection_id] = websocket
    
    # Initialize managers for this connection
    connection_managers[connection_id] = {
        'audio_buffer': AudioBuffer(),
        'conversation': ConversationManager(),
        'tts': TTSManager(),
        'stt': STTManager(),
        'call_data': [],
        'is_speaking': False,
        'silence_counter': 0
    }
    
    logger.info(f"New WebSocket connection: {connection_id}")
    
    # Send initial greeting
    await send_ai_response(connection_id, "Hello! How can I help you today?")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                json_data = json.loads(data)
                await handle_websocket_message(connection_id, json_data)
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
        await cleanup_connection(connection_id)
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await cleanup_connection(connection_id)

async def handle_websocket_message(connection_id: str, message: dict):
    managers = connection_managers[connection_id]
    
    if message.get("event") == "start":
        logger.info(f"Call started for connection {connection_id}")
        ucid = message.get("ucid")
        did = message.get("did")
        managers['ucid'] = ucid
        managers['did'] = did
        
    elif message.get("event") == "stop":
        logger.info(f"Call ended for connection {connection_id}")
        await cleanup_connection(connection_id)
        
    elif message.get("event") == "media" and message.get("type") == "media":
        # Handle audio data
        data = message.get("data", {})
        samples = data.get("samples", [])
        
        if samples:
            managers['call_data'].append(message)
            managers['audio_buffer'].add_samples(samples)
            
            # Check if we have enough audio for processing
            chunk = managers['audio_buffer'].get_chunk_if_ready()
            if chunk:
                await process_audio_chunk(connection_id, chunk)

async def process_audio_chunk(connection_id: str, audio_chunk: List[int]):
    managers = connection_managers[connection_id]
    
    # Check if user is speaking (simple voice activity detection)
    audio_level = np.mean(np.abs(audio_chunk))
    
    if audio_level > 100:  # Threshold for voice activity
        managers['is_speaking'] = True
        managers['silence_counter'] = 0
    else:
        managers['silence_counter'] += 1
        
        # If silence for 2 seconds (2 chunks), process speech
        if managers['is_speaking'] and managers['silence_counter'] >= 2:
            managers['is_speaking'] = False
            
            # Get all buffered audio for STT
            all_samples = managers['audio_buffer'].get_all_samples()
            if len(all_samples) > 8000:  # At least 1 second of audio
                await process_speech(connection_id, all_samples)
                managers['audio_buffer'] = AudioBuffer()  # Reset buffer

async def process_speech(connection_id: str, audio_samples: List[int]):
    managers = connection_managers[connection_id]
    
    try:
        # Convert speech to text
        transcript = await managers['stt'].speech_to_text(audio_samples)
        
        if transcript:
            logger.info(f"User said: {transcript}")
            
            # Get AI response
            ai_response = await managers['conversation'].get_ai_response(transcript)
            logger.info(f"AI response: {ai_response}")
            
            # Send AI response as speech
            await send_ai_response(connection_id, ai_response)
            
    except Exception as e:
        logger.error(f"Error processing speech: {e}")

async def send_ai_response(connection_id: str, text: str):
    managers = connection_managers[connection_id]
    websocket = active_connections.get(connection_id)
    
    if not websocket:
        return
    
    try:
        # Convert text to speech
        wav_data = await managers['tts'].text_to_speech(text)
        
        if wav_data:
            # Convert WAV to samples
            samples = managers['tts'].wav_to_samples(wav_data)
            
            # Send audio in chunks of 80 samples (10ms at 8kHz)
            chunk_size = 80
            ucid = managers.get('ucid', 'unknown')
            
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i + chunk_size]
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk.extend([0] * (chunk_size - len(chunk)))
                
                # Create WebSocket message
                message = {
                    "event": "media",
                    "type": "media",  
                    "ucid": ucid,
                    "data": {
                        "samples": chunk,
                        "bitsPerSample": 16,
                        "sampleRate": 8000,
                        "channelCount": 1,
                        "numberOfFrames": len(chunk),
                        "type": "data"
                    }
                }
                
                await websocket.send_text(json.dumps(message))
                await asyncio.sleep(0.01)  # 10ms delay between chunks
                
    except Exception as e:
        logger.error(f"Error sending AI response: {e}")

async def cleanup_connection(connection_id: str):
    try:
        # Save call recording
        if connection_id in connection_managers:
            call_data = connection_managers[connection_id]['call_data']
            if call_data:
                audio_samples = []
                for data in call_data:
                    samples = data.get("data", {}).get("samples", [])
                    audio_samples.extend(samples)
                
                if audio_samples:
                    audio_array = np.array(audio_samples, dtype=np.int16)
                    os.makedirs("recordings", exist_ok=True)
                    write(f"recordings/{connection_id}.wav", 8000, audio_array)
                    logger.info(f"Saved recording: recordings/{connection_id}.wav")
        
        # Cleanup
        if connection_id in active_connections:
            del active_connections[connection_id]
        if connection_id in connection_managers:
            del connection_managers[connection_id]
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
