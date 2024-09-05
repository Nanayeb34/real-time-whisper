import json
import argparse
from transformers import pipeline, WhisperProcessor
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import torch
import sounddevice as sd
import numpy as np
import queue
import threading
from pyannote.audio import Pipeline
from .utils.diarize import diarize_audio, post_process_segments_and_transcripts

from .utils.diarization_pipeline import diarize
from .utils.result import build_result

parser = argparse.ArgumentParser(description="Real-time Automatic Speech Recognition")

# Remove the --file-name argument
parser.add_argument(
    "--input-device",
    required=False,
    type=int,
    default=None,
    help="Input device index for the microphone. If not specified, the default system microphone will be used.",
)

# Modify the --transcript-path argument
parser.add_argument(
    "--output-file",
    required=False,
    default=None,
    type=str,
    help="Path to save the transcription output. If not specified, output will only be printed to console.",
)

# Add a new argument for chunk duration
parser.add_argument(
    "--chunk-duration",
    required=False,
    type=float,
    default=5.0,
    help="Duration of each audio chunk to process, in seconds. (default: 5.0)",
)

# Add a new argument for context size
parser.add_argument(
    "--context-size",
    required=False,
    type=int,
    default=3,
    help="Number of chunks to use for context. (default: 3)",
)

parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")',
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf-token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)
parser.add_argument(
    "--num-speakers",
    required=False,
    default=None,
    type=int,
    help="Specifies the exact number of speakers present in the audio file. Useful when the exact number of participants in the conversation is known. Must be at least 1. Cannot be used together with --min-speakers or --max-speakers. (default: None)",
)
parser.add_argument(
    "--min-speakers",
    required=False,
    default=None,
    type=int,
    help="Sets the minimum number of speakers that the system should consider during diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be less than or equal to --max-speakers if both are specified. (default: None)",
)
parser.add_argument(
    "--max-speakers",
    required=False,
    default=None,
    type=int,
    help="Defines the maximum number of speakers that the system should consider in diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be greater than or equal to --min-speakers if both are specified. (default: None)",
)

# Add this new argument to the parser
parser.add_argument(
    "--diarize",
    action="store_true",
    help="Enable diarization for the transcription.",
)

def main():
    args = parser.parse_args()

    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers cannot be used together with --min-speakers or --max-speakers.")

    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")

    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1.")

    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1.")

    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        if args.min_speakers > args.max_speakers:
            parser.error("--min-speakers cannot be greater than --max-speakers.")

    processor = WhisperProcessor.from_pretrained(args.model_name)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs={"attn_implementation": "flash_attention_2"} if args.flash else {"attn_implementation": "sdpa"},
    )

    if args.device_id == "mps":
        torch.mps.empty_cache()
    # elif not args.flash:
        # pipe.model = pipe.model.to_bettertransformer()

    ts = "word" if args.timestamp == "word" else True

    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task, "language": language}

    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        # outputs = pipe(
        #     args.file_name,
        #     chunk_length_s=30,
        #     batch_size=args.batch_size,
        #     generate_kwargs=generate_kwargs,
        #     return_timestamps=ts,
        # )

    # Set up diarization pipeline only if diarization is enabled
    diarization_pipeline = None
    if args.diarize:
        if args.hf_token == "no_token":
            parser.error("Diarization requires a HuggingFace token. Please provide --hf-token.")
        diarization_pipeline = Pipeline.from_pretrained(
            checkpoint_path=args.diarization_model,
            use_auth_token=args.hf_token,
        )
        diarization_pipeline.to(
            torch.device("mps" if args.device_id == "mps" else f"cuda:{args.device_id}")
        )

    # Set up audio stream
    samplerate = 16000  # Whisper expects 16kHz audio
    chunk_duration = args.chunk_duration  # Use the new chunk duration argument
    chunk_samples = int(samplerate * chunk_duration)
    context_size = args.context_size  # Use the new context size argument
    
    audio_buffer = queue.Queue()
    context_buffer = ""
    buffer_lock = threading.Lock()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_chunk = indata[:, 0]
        audio_buffer.put(audio_chunk)

    def process_audio():
        nonlocal context_buffer
        audio_context = []
        while True:
            # Get the next audio chunk
            chunk = audio_buffer.get()
            audio_context.append(chunk)
            
            # Limit the audio context to the last 3 chunks (15 seconds)
            if len(audio_context) > context_size:
                audio_context.pop(0)
            
            audio_data = np.concatenate(audio_context)
            # audio_tensor = torch.from_numpy(audio_data).float()

            # Transcribe
            result = pipe(audio_data, return_timestamps="chunk")

            # Diarize if enabled
            if args.diarize and diarization_pipeline:
                diarization = diarize_audio(
                    {"waveform": audio_tensor.unsqueeze(0), "sample_rate": samplerate}, 
                    diarization_pipeline, 
                    args.num_speakers, 
                    args.min_speakers, 
                    args.max_speakers
                )
                # Post-process diarization results
                diarized_result = post_process_segments_and_transcripts(
                    diarization, [result], group_by_speaker=True
                )
                
                with buffer_lock:
                    # Update context buffer with diarized result
                    for segment in diarized_result:
                        context_buffer += f"\n{segment['speaker']}: {segment['text']}"
                    # Keep only the last 1000 characters for context
                    context_buffer = context_buffer[-1000:]
                    
                    # Print the diarized transcription
                    print(f"Diarized Transcription: {context_buffer}")
            else:
                with buffer_lock:
                    # Update context buffer with non-diarized result
                    context_buffer += " " + result["text"]
                    # Keep only the last 1000 characters for context
                    context_buffer = context_buffer[-1000:]
                    
                    # Print the non-diarized transcription
                    print(f"Transcription: {context_buffer}")

    # Start the audio stream
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate,
                        blocksize=chunk_samples, device=args.input_device):
        print("Listening... Press Ctrl+C to stop.")
        
        # Start processing thread
        process_thread = threading.Thread(target=process_audio)
        process_thread.start()
        
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped listening.")

    # If an output file is specified, write the final transcription to it
    if args.output_file:
        with open(args.output_file, "w", encoding="utf8") as fp:
            json.dump({"transcription": context_buffer}, fp, ensure_ascii=False)
        print(f"Final transcription saved to {args.output_file}")

if __name__ == "__main__":
    main()
