import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, WhisperProcessor
import torch
import numpy as np
from pyannote.audio import Pipeline
from .utils.diarize import diarize_audio, post_process_segments_and_transcripts

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    device="cuda:0",  # Adjust as needed
    return_timestamps=True
)

# Initialize the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN_HERE"  # Replace with your Hugging Face token
)
diarization_pipeline.to(torch.device("cuda:0"))  # Adjust as needed

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = []
    context_buffer = ""
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "audio":
                audio_chunk = np.frombuffer(data["data"], dtype=np.float32)
                audio_buffer.append(audio_chunk)
                
                # Process audio when buffer reaches certain size
                if len(audio_buffer) >= 2:  # Adjust as needed
                    audio_data = np.concatenate(audio_buffer)
                    
                    # Transcribe
                    result = pipe(audio_data, return_timestamps=True)
                    
                    # Diarize if requested
                    if data.get("diarize", False):
                        audio_tensor = torch.from_numpy(audio_data).float()
                        diarization = diarize_audio(
                            audio_tensor,
                            diarization_pipeline,
                            num_speakers=data.get("num_speakers"),
                            min_speakers=data.get("min_speakers"),
                            max_speakers=data.get("max_speakers")
                        )
                        diarized_result = post_process_segments_and_transcripts(
                            diarization, result, group_by_speaker=True
                        )
                        
                        for segment in diarized_result:
                            context_buffer += f"\n{segment['speaker']}: {segment['text']}"
                    else:
                        context_buffer += " " + result["text"]
                    
                    # Keep only the last 1000 characters for context
                    context_buffer = context_buffer[-1000:]
                    
                    # Send transcription back to client
                    await websocket.send_json({
                        "type": "transcription",
                        "text": context_buffer
                    })
                    
                    # Clear audio buffer
                    audio_buffer = []
            
            elif data["type"] == "close":
                break
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)