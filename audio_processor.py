import wave
import struct
import uuid
from collections import deque
import json
import time


class TTSAudioLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.examples = deque()
        # self.load_audio()

    def load_audio(self, window_size=80):
        with wave.open(self.file_path, 'rb') as wf:
            params = wf.getparams()
            sample_rate = params.framerate
            bits_per_sample = 16  # Ensuring int16 format
            channel_count = params.nchannels
            number_of_frames = params.nframes

            # Read audio frames
            raw_data = wf.readframes(number_of_frames)
            total_samples = number_of_frames * channel_count

            # Unpack audio frames into int16 samples
            samples = struct.unpack(f"<{total_samples}h", raw_data)

            # Create examples with 80 samples each
            for i in range(0, len(samples), window_size):
                sample_chunk = samples[i:i+window_size]
                if len(sample_chunk) == window_size:
                    example = {
                        'type': 'media',
                        'ucid': self.generate_ucid(),
                        'data': {
                            'samples': list(sample_chunk),
                            'bitsPerSample': bits_per_sample,
                            'sampleRate': sample_rate,
                            'channelCount': channel_count,
                            'numberOfFrames': len(sample_chunk),
                            'type': 'data'
                        }
                    }
                    self.examples.append(example)

    def pop_example(self):
        if self.examples:
            return self.examples.pop()
        else:
            return None

    def play_audio(self, ws):
        #for i in range(len(self.examples)):
        #    await ws.send_text(json.dumps(self.examples.pop()))
        #    time.sleep(0.05)
        pass

    def generate_ucid(self):
        # Make sure you are using valid ucid
        # Generate a unique identifier (UCID) for each example
        return str(uuid.uuid4())


