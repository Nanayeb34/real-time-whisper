import requests
import torch
import numpy as np
from torchaudio import functional as F
from transformers.pipelines.audio_utils import ffmpeg_read
import sys


# Code lifted from https://github.com/huggingface/speechbox/blob/main/src/speechbox/diarize.py
# and from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py


def preprocess_inputs(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(waveform, diarization_pipeline, num_speakers, min_speakers, max_speakers):
    diarization = diarization_pipeline(
        {"waveform": waveform, "sample_rate": 16000},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    return new_segments


def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker) -> list:
    # Handle different possible structures of the transcript
    if isinstance(transcript, dict):
        # If transcript is a dictionary, it might contain a 'chunks' key
        chunks = transcript.get('chunks', [transcript])
    elif isinstance(transcript, list):
        chunks = transcript
    else:
        chunks = [transcript]

    # Get the end timestamps for each chunk from the ASR output
    end_timestamps = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            timestamp = chunk.get('timestamp', (None, None))
            if isinstance(timestamp, (list, tuple)) and len(timestamp) > 1 and timestamp[1] is not None:
                end_timestamps.append(timestamp[1])
            else:
                end_timestamps.append(sys.float_info.max)
        else:
            end_timestamps.append(sys.float_info.max)

    end_timestamps = np.array(end_timestamps)
    segmented_preds = []

    # Align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        end_time = segment["segment"]["end"]
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            text = "".join([chunk.get('text', '') for chunk in chunks[:upto_idx + 1]])
            timestamp = (
                chunks[0].get('timestamp', (None, None))[0] if chunks else None,
                chunks[upto_idx].get('timestamp', (None, None))[1] if chunks else None
            )
            segmented_preds.append({
                "speaker": segment["speaker"],
                "text": text,
                "timestamp": timestamp,
            })
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **chunks[i]})

        # Crop the chunks and timestamp lists
        chunks = chunks[upto_idx + 1:]
        end_timestamps = end_timestamps[upto_idx + 1:]

        if len(end_timestamps) == 0:
            break 

    return segmented_preds
