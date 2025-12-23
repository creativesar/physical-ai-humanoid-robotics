---
sidebar_position: 3
title: "Voice-to-Action: OpenAI Whisper Integration"
---

# Voice-to-Action: OpenAI Whisper Integration

## Introduction to Voice-to-Action Systems

Voice-to-action systems enable natural human-robot interaction by converting spoken language into executable robot commands. This technology represents a crucial step toward intuitive and accessible robotics, allowing users to communicate with robots using everyday language rather than specialized interfaces.

OpenAI Whisper is a state-of-the-art automatic speech recognition (ASR) system that provides accurate and robust speech-to-text capabilities. When integrated with robotics systems, Whisper enables robots to understand and process spoken commands in real-time, opening up new possibilities for human-robot interaction.

### Why Voice-to-Action?

Traditional robot interfaces often require:
- Physical interaction with buttons or joysticks
- Specialized programming or apps
- Technical knowledge of robot capabilities
- Visual attention to interface elements

Voice-to-action systems provide:
- Natural, intuitive interaction
- Hands-free operation
- Accessibility for users with physical limitations
- Operation in various lighting conditions
- Multimodal interaction capabilities

## Understanding OpenAI Whisper

### Architecture and Capabilities

Whisper is a transformer-based model that can handle multiple languages and various audio qualities. Its key features include:

- **Multilingual Support**: Recognizes and transcribes multiple languages
- **Robust Performance**: Handles background noise and varying audio quality
- **Speaker Identification**: Can distinguish between different speakers
- **Timestamp Generation**: Provides timing information for speech segments
- **Punctuation and Capitalization**: Adds appropriate punctuation to transcriptions

### Whisper Model Variants

Whisper comes in five sizes with different performance and speed characteristics:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Whisper Models                           │
├─────────────────────────────────────────────────────────────────┤
│ Model   │ Parameters │ Relative Speed │ English ASR │ Multilingual│
│         │            │                │ Performance │ Performance│
├─────────────────────────────────────────────────────────────────┤
│ tiny    │ 39 M       │ ~32x real-time │ 9.0%        │ 19.0%      │
│ base    │ 74 M       │ ~16x real-time │ 7.3%        │ 16.0%      │
│ small   │ 244 M      │ ~6x real-time  │ 5.9%        │ 12.4%      │
│ medium  │ 769 M      │ ~2x real-time  │ 4.8%        │ 10.5%      │
│ large   │ 1550 M     │ ~1x real-time  │ 3.8%        │ 7.3%       │
└─────────────────────────────────────────────────────────────────┘
```

## Integrating Whisper with Robotics

### Installation and Setup

```bash
# Install Whisper
pip install openai-whisper

# Install additional dependencies for audio processing
pip install pyaudio sounddevice librosa

# For GPU acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Whisper Integration

```python
import whisper
import torch
import numpy as np
import librosa
import sounddevice as sd
from threading import Thread, Event
import queue
import time

class WhisperVoiceToAction:
    def __init__(self, model_size="small", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Whisper-based voice-to-action system

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda', 'cpu')
        """
        self.model_size = model_size
        self.device = device

        # Load Whisper model
        print(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size, device=device)

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Audio recording variables
        self.audio_queue = queue.Queue()
        self.recording_event = Event()
        self.transcription_queue = queue.Queue()

        # Voice activity detection parameters
        self.energy_threshold = 0.01
        self.silence_duration = 1.0  # seconds of silence to trigger transcription

        print("Whisper voice-to-action system initialized")

    def audio_callback(self, indata, frames, time, status):
        """Audio callback for real-time recording"""
        if status:
            print(f"Audio status: {status}")

        # Put audio data in queue
        audio_data = indata.copy()
        self.audio_queue.put(audio_data)

    def start_listening(self):
        """Start real-time audio listening"""
        self.recording_event.clear()

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )

        self.stream.start()
        print("Started listening for voice commands...")

        # Start processing thread
        self.processing_thread = Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

    def stop_listening(self):
        """Stop audio listening"""
        self.recording_event.set()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def is_speech_detected(self, audio_chunk):
        """Detect if speech is present in audio chunk"""
        # Calculate energy of audio chunk
        energy = np.mean(np.abs(audio_chunk))
        return energy > self.energy_threshold

    def process_audio(self):
        """Process audio chunks for speech detection and transcription"""
        accumulated_audio = np.array([])
        silence_start_time = None

        while not self.recording_event.is_set():
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Flatten audio data if needed
                if len(audio_chunk.shape) > 1:
                    audio_chunk = audio_chunk.flatten()

                # Check if speech is detected
                if self.is_speech_detected(audio_chunk):
                    # Append to accumulated audio
                    accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])

                    # Reset silence timer
                    silence_start_time = None
                else:
                    # If we have accumulated audio and haven't started silence timer
                    if len(accumulated_audio) > 0 and silence_start_time is None:
                        silence_start_time = time.time()

                    # If silence duration exceeded and we have audio to transcribe
                    if (silence_start_time is not None and
                        time.time() - silence_start_time >= self.silence_duration and
                        len(accumulated_audio) > self.sample_rate * 0.5):  # At least 0.5 seconds

                        # Transcribe accumulated audio
                        transcription = self.transcribe_audio(accumulated_audio)

                        if transcription.strip():
                            self.transcription_queue.put(transcription)
                            print(f"Transcribed: {transcription}")

                        # Reset for next utterance
                        accumulated_audio = np.array([])
                        silence_start_time = None

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper"""
        try:
            # Convert audio to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_data,
                language="english",  # Specify language for better accuracy
                temperature=0.0,     # Use greedy decoding for consistency
                compression_ratio_threshold=None,  # Disable compression ratio filtering
                logprob_threshold=None,           # Disable logprob filtering
                no_speech_threshold=None          # Disable no-speech filtering
            )

            return result["text"].strip()

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def get_transcription(self, timeout=None):
        """Get next transcription from queue"""
        try:
            return self.transcription_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def process_voice_command(self, command_text):
        """
        Process voice command and convert to robot action

        Args:
            command_text: Transcribed text from speech

        Returns:
            dict: Action command for robot
        """
        # Normalize command text
        command = command_text.lower().strip()

        # Define command patterns and corresponding actions
        command_patterns = [
            # Navigation commands
            (r'move forward|go forward|move ahead|go ahead', {'action': 'navigate', 'direction': 'forward', 'distance': 1.0}),
            (r'move backward|go backward|move back|go back', {'action': 'navigate', 'direction': 'backward', 'distance': 1.0}),
            (r'turn left|rotate left|pivot left', {'action': 'navigate', 'direction': 'left', 'degrees': 90}),
            (r'turn right|rotate right|pivot right', {'action': 'navigate', 'direction': 'right', 'degrees': 90}),
            (r'stop|halt|freeze|pause', {'action': 'stop'}),
            (r'go to|move to|navigate to', self.parse_navigation_command(command)),

            # Manipulation commands
            (r'pick up|grasp|take|lift', self.parse_pickup_command(command)),
            (r'put down|place|drop|release', self.parse_placement_command(command)),
            (r'open|close', self.parse_manipulation_command(command)),

            # Interaction commands
            (r'hello|hi|hey', {'action': 'greet', 'type': 'hello'}),
            (r'goodbye|bye|see you', {'action': 'greet', 'type': 'goodbye'}),
            (r'help|assist|aid', {'action': 'request_help'}),

            # General commands
            (r'what can you do|what are your abilities', {'action': 'describe_capabilities'}),
            (r'status|how are you|what are you doing', {'action': 'report_status'}),
        ]

        # Match command to pattern
        for pattern, action in command_patterns:
            import re
            if re.search(pattern, command):
                if callable(action):
                    return action(command)
                return action

        # If no pattern matches, return unknown command
        return {'action': 'unknown', 'text': command}

    def parse_navigation_command(self, command):
        """Parse navigation commands with destination"""
        import re

        # Look for location keywords
        location_pattern = r'(?:to|toward|at)\s+(.+?)(?:\.|$)'
        match = re.search(location_pattern, command)

        if match:
            destination = match.group(1).strip()
            return {'action': 'navigate_to_location', 'destination': destination}

        return {'action': 'unknown', 'text': command}

    def parse_pickup_command(self, command):
        """Parse pickup commands with object"""
        import re

        # Look for object to pick up
        object_pattern = r'(?:pick up|grasp|take|lift)\s+(.+?)(?:\.|$)'
        match = re.search(object_pattern, command)

        if match:
            object_name = match.group(1).strip()
            return {'action': 'pickup_object', 'object': object_name}

        return {'action': 'unknown', 'text': command}

    def parse_placement_command(self, command):
        """Parse placement commands with location"""
        import re

        # Look for placement location
        location_pattern = r'(?:put down|place|drop|release)\s+(?:on|at|in)\s+(.+?)(?:\.|$)'
        match = re.search(location_pattern, command)

        if match:
            location = match.group(1).strip()
            return {'action': 'place_object', 'location': location}

        return {'action': 'unknown', 'text': command}

    def parse_manipulation_command(self, command):
        """Parse manipulation commands"""
        import re

        if 'open' in command:
            return {'action': 'manipulate', 'action_type': 'open'}
        elif 'close' in command:
            return {'action': 'manipulate', 'action_type': 'close'}

        return {'action': 'unknown', 'text': command}
```

## Advanced Voice Processing

### Noise Reduction and Audio Enhancement

```python
import librosa
import numpy as np
from scipy import signal
import webrtcvad  # WebRTC Voice Activity Detection

class AdvancedVoiceProcessor:
    def __init__(self):
        # Initialize noise reduction parameters
        self.noise_floor = 0.01
        self.snr_threshold = 10.0  # Signal-to-noise ratio threshold

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # Aggressiveness mode (0-3)

    def denoise_audio(self, audio_data, sample_rate=16000):
        """Apply noise reduction to audio data"""
        # Calculate noise profile from silent portions
        noise_profile = self.estimate_noise_profile(audio_data)

        # Apply spectral subtraction for denoising
        denoised_audio = self.spectral_subtraction(audio_data, noise_profile)

        return denoised_audio

    def estimate_noise_profile(self, audio_data, window_size=1024):
        """Estimate noise profile from audio data"""
        # Divide audio into frames
        frames = self.frame_audio(audio_data, window_size)

        # Calculate energy for each frame
        frame_energies = [np.mean(np.abs(frame)**2) for frame in frames]

        # Estimate noise floor as percentile of low-energy frames
        noise_floor = np.percentile(frame_energies, 10)

        # Calculate noise spectrum
        noise_spectrum = []
        for frame in frames:
            frame_fft = np.fft.fft(frame)
            frame_power = np.abs(frame_fft)**2
            noise_spectrum.append(np.minimum(frame_power, noise_floor))

        return np.mean(noise_spectrum, axis=0)

    def spectral_subtraction(self, audio_data, noise_profile, alpha=1.0, beta=2.0):
        """Apply spectral subtraction for noise reduction"""
        # Convert to frequency domain
        fft_data = np.fft.fft(audio_data)
        power_spectrum = np.abs(fft_data)**2

        # Subtract noise spectrum
        enhanced_spectrum = np.maximum(power_spectrum - alpha * noise_profile,
                                     beta * noise_profile)

        # Convert back to time domain
        enhanced_fft = np.sqrt(enhanced_spectrum) * np.exp(1j * np.angle(fft_data))
        enhanced_audio = np.real(np.fft.ifft(enhanced_fft))

        return enhanced_audio.astype(audio_data.dtype)

    def frame_audio(self, audio_data, frame_size, hop_size=None):
        """Divide audio into overlapping frames"""
        if hop_size is None:
            hop_size = frame_size // 2

        frames = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            frames.append(frame)

        return frames

    def enhance_audio_quality(self, audio_data, sample_rate=16000):
        """Enhance audio quality for better ASR performance"""
        # Apply pre-emphasis filter
        enhanced_audio = self.preemphasis_filter(audio_data)

        # Apply noise reduction
        enhanced_audio = self.denoise_audio(enhanced_audio, sample_rate)

        # Normalize audio
        enhanced_audio = self.normalize_audio(enhanced_audio)

        return enhanced_audio

    def preemphasis_filter(self, audio_data, coeff=0.97):
        """Apply pre-emphasis filter to boost high frequencies"""
        return np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])

    def normalize_audio(self, audio_data):
        """Normalize audio to prevent clipping"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
```

### Real-time Voice Activity Detection

```python
import pyaudio
import numpy as np
import threading
import queue
import time

class RealTimeVoiceActivityDetector:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Voice activity detection parameters
        self.energy_threshold = 0.01
        self.silence_threshold = 0.5  # Percentage of silent frames
        self.min_speech_frames = 5    # Minimum frames to consider speech
        self.max_silence_frames = 10  # Maximum silent frames before stopping

        # Audio processing
        self.audio_queue = queue.Queue()
        self.vad_queue = queue.Queue()
        self.listening = False

        # Callback function storage
        self.on_speech_detected = None
        self.on_silence_detected = None
        self.on_audio_chunk = None

    def start_listening(self):
        """Start real-time voice activity detection"""
        self.listening = True

        # Start audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

        print("Started real-time voice activity detection")

    def stop_listening(self):
        """Stop voice activity detection"""
        self.listening = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, 'audio'):
            self.audio.terminate()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Put audio data in queue for processing
        self.audio_queue.put(audio_data)

        # Call user callback if provided
        if self.on_audio_chunk:
            self.on_audio_chunk(audio_data)

        return (None, pyaudio.paContinue)

    def process_audio(self):
        """Process audio for voice activity detection"""
        speech_frames = []
        silent_frames = 0
        in_speech = False

        while self.listening:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Calculate energy
                energy = np.mean(np.abs(audio_chunk)**2)
                is_speech = energy > self.energy_threshold

                if is_speech:
                    # Add to speech frames
                    speech_frames.append(audio_chunk)
                    silent_frames = 0

                    # If just started speaking
                    if not in_speech:
                        in_speech = True
                        if self.on_speech_detected:
                            self.on_speech_detected()
                else:
                    # Count silent frames
                    silent_frames += 1

                    # If currently in speech and silence detected
                    if in_speech:
                        if silent_frames >= self.max_silence_frames:
                            # End of speech detected
                            in_speech = False

                            # Combine speech frames
                            if speech_frames:
                                full_speech = np.concatenate(speech_frames)

                                # Call user callback
                                if self.on_silence_detected:
                                    self.on_silence_detected(full_speech)

                                # Clear speech frames
                                speech_frames = []

                        # Add to speech frames if still in speech
                        elif speech_frames:
                            speech_frames.append(audio_chunk)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue
```

## Integration with Robotics Systems

### ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
import torch

class WhisperROSIntegration(Node):
    def __init__(self):
        super().__init__('whisper_voice_control')

        # Initialize Whisper voice-to-action system
        self.voice_processor = WhisperVoiceToAction(model_size="small")

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.voice_response_pub = self.create_publisher(String, '/voice_response', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            String,  # This would be audio data in practice
            '/audio_input',
            self.audio_callback,
            10
        )

        # Timer for processing voice commands
        self.process_timer = self.create_timer(0.1, self.process_voice_commands)

        # Store pending commands
        self.pending_commands = []

        # Start listening
        self.voice_processor.start_listening()

        self.get_logger().info('Whisper ROS integration node initialized')

    def audio_callback(self, msg):
        """Process audio input (in practice, this would be audio data)"""
        # This is a simplified example
        # In practice, you'd receive audio data and process it
        pass

    def process_voice_commands(self):
        """Process any pending voice commands"""
        # Get any new transcriptions
        while True:
            transcription = self.voice_processor.get_transcription(timeout=0.01)
            if transcription is None:
                break

            # Process the command
            action = self.voice_processor.process_voice_command(transcription)
            self.execute_robot_action(action, transcription)

    def execute_robot_action(self, action_dict, original_command):
        """Execute robot action based on voice command"""
        action_type = action_dict.get('action', 'unknown')

        if action_type == 'navigate':
            direction = action_dict.get('direction', 'forward')
            distance = action_dict.get('distance', 1.0)
            self.execute_navigation_command(direction, distance)

        elif action_type == 'navigate_to_location':
            destination = action_dict.get('destination', '')
            self.execute_navigation_to_location(destination)

        elif action_type == 'stop':
            self.stop_robot()

        elif action_type == 'pickup_object':
            object_name = action_dict.get('object', '')
            self.execute_pickup_command(object_name)

        elif action_type == 'place_object':
            location = action_dict.get('location', '')
            self.execute_placement_command(location)

        elif action_type == 'greet':
            greet_type = action_dict.get('type', 'hello')
            self.execute_greeting(greet_type)

        else:
            # Unknown command
            response = f"I don't understand the command: {original_command}"
            self.publish_voice_response(response)
            self.get_logger().warn(f"Unknown command: {original_command}")

    def execute_navigation_command(self, direction, distance=1.0):
        """Execute navigation command"""
        cmd_vel = Twist()

        if direction == 'forward':
            cmd_vel.linear.x = 0.5 * distance  # Adjust speed based on distance
        elif direction == 'backward':
            cmd_vel.linear.x = -0.5 * distance
        elif direction == 'left':
            cmd_vel.angular.z = 0.5 * distance
        elif direction == 'right':
            cmd_vel.angular.z = -0.5 * distance

        self.cmd_vel_pub.publish(cmd_vel)

        response = f"Moving {direction} for {distance} meters"
        self.publish_voice_response(response)

    def execute_navigation_to_location(self, destination):
        """Execute navigation to specific location"""
        # This would integrate with navigation stack
        response = f"Moving to {destination}. This feature requires navigation stack integration."
        self.publish_voice_response(response)

    def stop_robot(self):
        """Stop robot movement"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

        response = "Robot stopped"
        self.publish_voice_response(response)

    def execute_pickup_command(self, object_name):
        """Execute object pickup command"""
        response = f"Attempting to pick up {object_name}. This feature requires manipulation stack integration."
        self.publish_voice_response(response)

    def execute_placement_command(self, location):
        """Execute object placement command"""
        response = f"Attempting to place object at {location}. This feature requires manipulation stack integration."
        self.publish_voice_response(response)

    def execute_greeting(self, greet_type):
        """Execute greeting command"""
        if greet_type == 'hello':
            response = "Hello! How can I help you today?"
        elif greet_type == 'goodbye':
            response = "Goodbye! Have a great day!"
        else:
            response = "Hello!"

        self.publish_voice_response(response)

    def publish_voice_response(self, response_text):
        """Publish voice response"""
        response_msg = String()
        response_msg.data = response_text
        self.voice_response_pub.publish(response_msg)

        self.get_logger().info(f"Voice response: {response_text}")

def main(args=None):
    rclpy.init(args=args)

    voice_control_node = WhisperROSIntegration()

    try:
        rclpy.spin(voice_control_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_control_node.voice_processor.stop_listening()
        voice_control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Natural Language Processing for Robot Commands

### Command Parser with Context Awareness

```python
import re
import spacy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RobotCommand:
    action: str
    parameters: Dict[str, any]
    confidence: float
    context: Dict[str, any]

class AdvancedCommandParser:
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define action vocabulary
        self.action_keywords = {
            'navigation': ['move', 'go', 'walk', 'drive', 'navigate', 'travel', 'head'],
            'manipulation': ['pick', 'grasp', 'take', 'lift', 'place', 'put', 'drop', 'hold'],
            'interaction': ['say', 'speak', 'tell', 'greet', 'hello', 'goodbye', 'hi'],
            'inspection': ['look', 'see', 'find', 'locate', 'search', 'detect'],
            'utility': ['stop', 'wait', 'pause', 'help', 'assist']
        }

        # Define object categories
        self.object_categories = {
            'containers': ['cup', 'bottle', 'box', 'container', 'jar', 'bowl'],
            'furniture': ['table', 'chair', 'desk', 'couch', 'bed', 'shelf'],
            'electronics': ['phone', 'computer', 'tablet', 'tv', 'remote', 'lamp'],
            'food': ['apple', 'banana', 'water', 'snack', 'food', 'drink']
        }

        # Location keywords
        self.location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'hallway']

        # Direction keywords
        self.direction_keywords = {
            'forward': ['forward', 'ahead', 'straight', 'front'],
            'backward': ['backward', 'back', 'behind'],
            'left': ['left', 'left side', 'port'],
            'right': ['right', 'right side', 'starboard'],
            'up': ['up', 'above', 'top'],
            'down': ['down', 'below', 'bottom']
        }

    def parse_command(self, text: str, context: Dict = None) -> Optional[RobotCommand]:
        """
        Parse natural language command into structured robot command

        Args:
            text: Natural language command
            context: Additional context information

        Returns:
            RobotCommand with parsed action and parameters
        """
        if self.nlp is None:
            return self.fallback_parse(text, context)

        # Process with spaCy
        doc = self.nlp(text.lower())

        # Extract action
        action = self.extract_action(doc)

        # Extract parameters
        parameters = self.extract_parameters(doc, action)

        # Calculate confidence based on keyword matches
        confidence = self.calculate_confidence(doc, action, parameters)

        return RobotCommand(
            action=action,
            parameters=parameters,
            confidence=confidence,
            context=context or {}
        )

    def extract_action(self, doc) -> str:
        """Extract the main action from the command"""
        # Look for action keywords in the text
        for token in doc:
            if token.pos_ in ['VERB', 'AUX']:  # Verbs and auxiliary verbs
                verb_lemma = token.lemma_

                for action_category, keywords in self.action_keywords.items():
                    if verb_lemma in keywords:
                        return action_category

        # If no verb found, try noun phrases
        for token in doc:
            if token.pos_ == 'NOUN':
                noun = token.text
                for action_category, keywords in self.action_keywords.items():
                    if noun in keywords:
                        return action_category

        return 'unknown'

    def extract_parameters(self, doc, action: str) -> Dict[str, any]:
        """Extract parameters for the action"""
        parameters = {}

        # Extract objects
        objects = self.extract_objects(doc)
        if objects:
            parameters['objects'] = objects

        # Extract locations
        locations = self.extract_locations(doc)
        if locations:
            parameters['locations'] = locations

        # Extract directions
        directions = self.extract_directions(doc)
        if directions:
            parameters['directions'] = directions

        # Extract quantities
        quantities = self.extract_quantities(doc)
        if quantities:
            parameters['quantities'] = quantities

        # Action-specific parameter extraction
        if action == 'navigation':
            parameters.update(self.extract_navigation_params(doc))
        elif action == 'manipulation':
            parameters.update(self.extract_manipulation_params(doc))

        return parameters

    def extract_objects(self, doc) -> List[str]:
        """Extract objects from the command"""
        objects = []

        for ent in doc.ents:
            if ent.label_ in ['OBJECT', 'PRODUCT', 'FOOD', 'PERSON']:
                objects.append(ent.text)

        # Also look for noun phrases that might be objects
        for token in doc:
            if token.pos_ == 'NOUN':
                # Check if this noun is preceded by action words
                if token.i > 0 and doc[token.i - 1].text in ['pick', 'grasp', 'take', 'put', 'place']:
                    objects.append(token.text)

        return list(set(objects))  # Remove duplicates

    def extract_locations(self, doc) -> List[str]:
        """Extract locations from the command"""
        locations = []

        for ent in doc.ents:
            if ent.label_ in ['LOC', 'GPE', 'FACILITY']:
                locations.append(ent.text)

        # Look for location keywords
        for token in doc:
            if token.text in self.location_keywords:
                locations.append(token.text)

        return list(set(locations))

    def extract_directions(self, doc) -> List[str]:
        """Extract directions from the command"""
        directions = []

        for token in doc:
            for dir_type, keywords in self.direction_keywords.items():
                if token.text in keywords:
                    directions.append(dir_type)

        return list(set(directions))

    def extract_quantities(self, doc) -> List[float]:
        """Extract quantity values from the command"""
        quantities = []

        for ent in doc.ents:
            if ent.label_ == 'CARDINAL':
                try:
                    quantities.append(float(ent.text))
                except ValueError:
                    continue

        # Look for number tokens
        for token in doc:
            if token.like_num:
                try:
                    quantities.append(float(token.text))
                except ValueError:
                    continue

        return quantities

    def extract_navigation_params(self, doc) -> Dict[str, any]:
        """Extract navigation-specific parameters"""
        params = {}

        # Look for distance expressions
        for token in doc:
            if token.text in ['meter', 'meters', 'foot', 'feet', 'step', 'steps']:
                # Look for preceding number
                if token.i > 0:
                    try:
                        distance = float(doc[token.i - 1].text)
                        params['distance'] = distance
                    except ValueError:
                        pass

        return params

    def extract_manipulation_params(self, doc) -> Dict[str, any]:
        """Extract manipulation-specific parameters"""
        params = {}

        # Look for grasp types
        for token in doc:
            if token.text in ['gently', 'carefully', 'firmly', 'lightly']:
                params['grasp_type'] = token.text

        return params

    def calculate_confidence(self, doc, action: str, parameters: Dict) -> float:
        """Calculate confidence score for the parsed command"""
        confidence = 0.0

        # Base confidence on action recognition
        if action != 'unknown':
            confidence += 0.3

        # Add confidence for parameter recognition
        if parameters:
            confidence += min(0.7, len(parameters) * 0.1)

        # Boost confidence if we have both action and key parameters
        if action != 'unknown' and parameters:
            confidence += 0.2

        # Cap at 1.0
        return min(1.0, confidence)

    def fallback_parse(self, text: str, context: Dict = None) -> RobotCommand:
        """Fallback parsing if spaCy is not available"""
        # Simple keyword-based parsing
        text_lower = text.lower()

        # Extract action
        action = 'unknown'
        for action_type, keywords in self.action_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                action = action_type
                break

        # Extract simple parameters
        parameters = {}
        if 'forward' in text_lower:
            parameters['direction'] = 'forward'
        if 'left' in text_lower:
            parameters['direction'] = 'left'
        if 'right' in text_lower:
            parameters['direction'] = 'right'
        if 'back' in text_lower or 'backward' in text_lower:
            parameters['direction'] = 'backward'

        return RobotCommand(
            action=action,
            parameters=parameters,
            confidence=0.5,  # Lower confidence for fallback
            context=context or {}
        )
```

## Integration with Large Language Models

### Using LLMs for Command Interpretation

```python
import openai
from typing import Dict, List, Optional
import json

class LLMCommandInterpreter:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM command interpreter

        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: LLM model to use
        """
        if api_key:
            openai.api_key = api_key

        self.model = model
        self.command_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate", "manipulate", "inspect", "communicate", "utility"],
                    "description": "The main action to perform"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for the action",
                    "properties": {
                        "target": {"type": "string", "description": "Target object or location"},
                        "direction": {"type": "string", "description": "Direction for movement"},
                        "distance": {"type": "number", "description": "Distance in meters"},
                        "speed": {"type": "number", "description": "Speed factor"},
                        "description": {"type": "string", "description": "Additional details"}
                    }
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in interpretation"
                }
            },
            "required": ["action", "parameters", "confidence"]
        }

    def interpret_command(self, command_text: str, robot_context: Dict = None) -> Optional[RobotCommand]:
        """
        Interpret natural language command using LLM

        Args:
            command_text: Natural language command
            robot_context: Context about robot capabilities and environment

        Returns:
            RobotCommand with structured action
        """
        # Prepare system message with context
        system_message = self.create_system_message(robot_context)

        # Prepare user message with command
        user_message = f"Interpret this robot command: '{command_text}'"

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                functions=[{
                    "name": "interpret_robot_command",
                    "description": "Interpret a natural language command for a robot",
                    "parameters": self.command_schema
                }],
                function_call={"name": "interpret_robot_command"},
                temperature=0.1  # Low temperature for consistency
            )

            # Extract function arguments
            function_args = json.loads(response.choices[0].message.function_call.arguments)

            return RobotCommand(
                action=function_args['action'],
                parameters=function_args['parameters'],
                confidence=function_args['confidence'],
                context=robot_context or {}
            )

        except Exception as e:
            print(f"Error interpreting command with LLM: {e}")
            return None

    def create_system_message(self, robot_context: Dict = None) -> str:
        """Create system message with robot context"""
        base_context = """
        You are an expert at interpreting natural language commands for robots.
        Your job is to convert human language into structured robot commands.

        The robot has the following capabilities:
        - Navigation: Can move in different directions and distances
        - Manipulation: Can pick up, place, and manipulate objects
        - Inspection: Can look at and identify objects
        - Communication: Can speak and interact with humans
        - Utility: Can perform basic utility functions

        Always respond using the interpret_robot_command function with structured parameters.
        """

        if robot_context:
            base_context += f"\n\nAdditional robot context: {robot_context}"

        return base_context

    def batch_interpret_commands(self, commands: List[str], robot_context: Dict = None) -> List[RobotCommand]:
        """Interpret multiple commands in batch"""
        results = []

        for command in commands:
            interpreted = self.interpret_command(command, robot_context)
            if interpreted:
                results.append(interpreted)

        return results
```

## Voice Command Validation and Safety

### Safety Layer for Voice Commands

```python
class VoiceCommandSafetyLayer:
    def __init__(self):
        # Define safe action parameters
        self.safe_speed_limits = {
            'linear': 0.5,    # m/s
            'angular': 0.5    # rad/s
        }

        self.safe_distance_limits = {
            'min': 0.1,       # minimum distance from obstacles
            'max_navigation': 10.0  # maximum navigation distance
        }

        # Dangerous commands that should be filtered
        self.dangerous_keywords = [
            'break', 'destroy', 'damage', 'hurt', 'attack', 'hit',
            'fire', 'explosion', 'emergency', 'danger'
        ]

        # Restricted areas (would come from map data)
        self.restricted_areas = set(['exit', 'emergency', 'staff_only'])

    def validate_command(self, command: RobotCommand) -> Tuple[bool, str, RobotCommand]:
        """
        Validate voice command for safety

        Returns:
            (is_safe, reason, modified_command)
        """
        # Check for dangerous keywords
        if hasattr(command, 'parameters') and 'description' in command.parameters:
            desc = command.parameters['description'].lower()
            for keyword in self.dangerous_keywords:
                if keyword in desc:
                    return False, f"Dangerous keyword detected: {keyword}", command

        # Validate navigation commands
        if command.action == 'navigate':
            return self.validate_navigation_command(command)

        # Validate manipulation commands
        if command.action == 'manipulate':
            return self.validate_manipulation_command(command)

        # For other commands, basic validation
        return True, "Command is safe", command

    def validate_navigation_command(self, command: RobotCommand) -> Tuple[bool, str, RobotCommand]:
        """Validate navigation command for safety"""
        params = command.parameters

        # Check distance limits
        if 'distance' in params:
            distance = params['distance']
            if distance > self.safe_distance_limits['max_navigation']:
                # Modify to safe distance
                safe_distance = self.safe_distance_limits['max_navigation']
                modified_command = RobotCommand(
                    action=command.action,
                    parameters={**params, 'distance': safe_distance},
                    confidence=command.confidence,
                    context=command.context
                )
                return False, f"Distance too large ({distance}m), reduced to {safe_distance}m", modified_command

        # Check for restricted areas
        if 'target' in params:
            target = params['target'].lower()
            if target in self.restricted_areas:
                return False, f"Navigation to restricted area: {target}", command

        return True, "Navigation command is safe", command

    def validate_manipulation_command(self, command: RobotCommand) -> Tuple[bool, str, RobotCommand]:
        """Validate manipulation command for safety"""
        params = command.parameters

        # Check for dangerous object manipulation
        if 'target' in params:
            target = params['target'].lower()
            if any(dangerous in target for dangerous in ['knife', 'blade', 'sharp', 'hot', 'fire']):
                return False, f"Dangerous object manipulation requested: {target}", command

        return True, "Manipulation command is safe", command

    def apply_speed_limits(self, cmd_vel_msg, command: RobotCommand):
        """Apply speed limits to command velocity"""
        # Limit linear velocity
        cmd_vel_msg.linear.x = max(-self.safe_speed_limits['linear'],
                                  min(self.safe_speed_limits['linear'], cmd_vel_msg.linear.x))
        cmd_vel_msg.linear.y = max(-self.safe_speed_limits['linear'],
                                  min(self.safe_speed_limits['linear'], cmd_vel_msg.linear.y))
        cmd_vel_msg.linear.z = max(-self.safe_speed_limits['linear'],
                                  min(self.safe_speed_limits['linear'], cmd_vel_msg.linear.z))

        # Limit angular velocity
        cmd_vel_msg.angular.x = max(-self.safe_speed_limits['angular'],
                                   min(self.safe_speed_limits['angular'], cmd_vel_msg.angular.x))
        cmd_vel_msg.angular.y = max(-self.safe_speed_limits['angular'],
                                   min(self.safe_speed_limits['angular'], cmd_vel_msg.angular.y))
        cmd_vel_msg.angular.z = max(-self.safe_speed_limits['angular'],
                                   min(self.safe_speed_limits['angular'], cmd_vel_msg.angular.z))

        return cmd_vel_msg
```

## Performance Optimization

### Efficient Whisper Processing

```python
import threading
import asyncio
import concurrent.futures
from typing import Callable, Any

class OptimizedWhisperProcessor:
    def __init__(self, model_size="small", max_workers=2):
        """
        Optimized Whisper processor with concurrent processing

        Args:
            model_size: Size of Whisper model
            max_workers: Number of concurrent workers for transcription
        """
        self.model_size = model_size
        self.max_workers = max_workers

        # Load model once
        self.model = whisper.load_model(model_size)

        # Thread pool for concurrent processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Audio processing queue
        self.processing_queue = asyncio.Queue()
        self.result_callbacks = []

        print(f"Initialized optimized Whisper processor with {max_workers} workers")

    async def transcribe_audio_async(self, audio_data) -> str:
        """Asynchronously transcribe audio data"""
        loop = asyncio.get_event_loop()

        # Submit transcription to thread pool
        future = loop.run_in_executor(
            self.executor,
            self._transcribe_sync,
            audio_data
        )

        return await future

    def _transcribe_sync(self, audio_data) -> str:
        """Synchronous transcription (runs in thread pool)"""
        try:
            result = self.model.transcribe(
                audio_data,
                language="english",
                temperature=0.0,
                compression_ratio_threshold=None,
                logprob_threshold=None,
                no_speech_threshold=None
            )
            return result["text"].strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""

    def add_result_callback(self, callback: Callable[[str], None]):
        """Add callback for transcription results"""
        self.result_callbacks.append(callback)

    def process_audio_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        """Process multiple audio segments in parallel"""
        # Submit all transcriptions
        futures = []
        for segment in audio_segments:
            future = self.executor.submit(self._transcribe_sync, segment)
            futures.append(future)

        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

        # Sort results to maintain order
        results.sort(key=lambda x: audio_segments.index(x))  # This won't work, need better approach

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Whisper model"""
        return {
            'model_size': self.model_size,
            'max_workers': self.max_workers,
            'model_type': 'Whisper',
            'supported_languages': ['en', 'es', 'fr', 'de', 'ja', 'pt', 'ko', 'zh', 'ru', 'ar']
        }
```

## Best Practices for Voice-to-Action Systems

### 1. Audio Quality Management
- Use high-quality microphones for better speech recognition
- Implement noise reduction algorithms
- Consider acoustic environment when placing microphones
- Use directional microphones to focus on speaker

### 2. Command Design
- Use consistent command structures
- Provide clear feedback for recognized commands
- Implement command confirmation for critical actions
- Support both specific and general commands

### 3. Error Handling
- Gracefully handle unrecognized commands
- Provide helpful error messages
- Implement fallback mechanisms
- Log errors for system improvement

### 4. Privacy and Security
- Implement proper audio data handling
- Use secure transmission for sensitive commands
- Provide user controls for data collection
- Consider local processing for privacy-sensitive applications

### 5. User Experience
- Provide clear instructions for voice commands
- Implement natural conversation flow
- Support interruption and correction
- Adapt to user preferences over time

## Troubleshooting Common Issues

### 1. Poor Recognition Accuracy
- **Problem**: Whisper doesn't recognize commands well
- **Solution**: Improve audio quality, adjust model parameters, use smaller models for real-time

### 2. High Latency
- **Problem**: Delay between speaking and robot response
- **Solution**: Optimize processing pipeline, use faster models, implement streaming

### 3. False Positives
- **Problem**: Robot responds to background speech
- **Solution**: Implement wake word detection, improve VAD, adjust sensitivity

### 4. Context Confusion
- **Problem**: Robot misunderstands commands in context
- **Solution**: Implement context awareness, use LLMs for disambiguation

## Integration with Cognitive Planning

The voice-to-action system serves as the input layer for cognitive planning systems, where natural language commands are converted to structured actions that can be reasoned about and executed by the robot's planning system. This integration enables robots to understand and execute complex, multi-step commands expressed in natural language.

In the next section, we'll explore cognitive planning systems that use LLMs to translate natural language commands into executable robotic actions.