import logging
import os
import platform
import shutil
import subprocess
import threading
import time

import click
import mlx_whisper
import numpy as np
import pyaudio
from pynput import keyboard

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Configuration Constants ---
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
# Path to system sound files
START_SOUND = "/System/Library/Sounds/Blow.aiff"
STOP_SOUND = "/System/Library/Sounds/Frog.aiff"
# --- Globals for Recording Thread ---
frames = []
recording_active = False
p = None
stream = None
hotkey_listener = None

# --- MLX Whisper Models (Keep this list updated as needed) ---
MLX_WHISPER_MODELS = [
    "mlx-community/whisper-large-v3-mlx",
    "mlx-community/whisper-tiny-mlx-q4",
    "mlx-community/whisper-large-v2-mlx-fp32",
    "mlx-community/whisper-tiny.en-mlx-q4",
    "mlx-community/whisper-base.en-mlx-q4",
    "mlx-community/whisper-small.en-mlx-q4",
    "mlx-community/whisper-tiny-mlx-fp32",
    "mlx-community/whisper-base-mlx-fp32",
    "mlx-community/whisper-small-mlx-fp32",
    "mlx-community/whisper-medium-mlx-fp32",
    "mlx-community/whisper-base-mlx-2bit",
    "mlx-community/whisper-tiny-mlx-8bit",
    "mlx-community/whisper-tiny.en-mlx-4bit",
    "mlx-community/whisper-base-mlx",
    "mlx-community/whisper-base-mlx-8bit",
    "mlx-community/whisper-base.en-mlx-4bit",
    "mlx-community/whisper-small-mlx",
    "mlx-community/whisper-small-mlx-8bit",
    "mlx-community/whisper-small.en-mlx-4bit",
    "mlx-community/whisper-medium-mlx-8bit",
    "mlx-community/whisper-medium.en-mlx-8bit",
    "mlx-community/whisper-large-mlx-4bit",
    "mlx-community/whisper-large-v1-mlx",
    "mlx-community/whisper-large-v1-mlx-8bit",
    "mlx-community/whisper-large-v2-mlx-8bit",
    "mlx-community/whisper-large-v2-mlx-4bit",
    "mlx-community/whisper-large-v1-mlx-4bit",
    "mlx-community/whisper-large-mlx-8bit",
    "mlx-community/whisper-large-mlx",
    "mlx-community/whisper-medium.en-mlx-4bit",
    "mlx-community/whisper-small.en-mlx-8bit",
    "mlx-community/whisper-small.en-mlx",
    "mlx-community/whisper-small-mlx-4bit",
    "mlx-community/whisper-base.en-mlx-8bit",
    "mlx-community/whisper-base.en-mlx",
    "mlx-community/whisper-base-mlx-4bit",
    "mlx-community/whisper-tiny.en-mlx-8bit",
    "mlx-community/whisper-tiny.en-mlx",
    "mlx-community/whisper-tiny-mlx",
    "mlx-community/whisper-medium.en-mlx-fp32",
    "mlx-community/whisper-small.en-mlx-fp32",
    "mlx-community/whisper-base.en-mlx-fp32",
    "mlx-community/whisper-tiny.en-mlx-fp32",
    "mlx-community/whisper-medium-mlx-q4",
    "mlx-community/whisper-small-mlx-q4",
    "mlx-community/whisper-base-mlx-q4",
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-turbo",
]


# --- Sound Notification Function ---
def play_sound(sound_file):
    """Play a notification sound using macOS afplay command."""
    if not os.path.exists(sound_file):
        logging.warning(f"Sound file not found: {sound_file}")
        return

    try:
        # Use the macOS afplay command which can handle AIFF files
        subprocess.run(["afplay", sound_file], check=True)
    except Exception as e:
        logging.error(f"Error playing sound: {e}")


# --- Recording Function (to run in a separate thread) ---
def record_audio():
    """Reads audio chunks from the stream and stores them in frames."""
    global frames, recording_active, stream
    logging.info("Recording thread started.")
    frames = []  # Reset frames at the start of recording
    while recording_active and stream is not None:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except OSError as e:
            # Ignore input overflow errors, log others
            if e.errno != pyaudio.paInputOverflowed:
                logging.error(f"Recording IO error: {e}")
        except Exception as e:
            logging.error(f"Unexpected recording error: {e}", exc_info=True)
            # Decide if you want to stop recording on other errors
            # recording_active = False
    logging.info("Recording thread finished.")


# --- Hotkey Handling Functions ---
def start_stop_recording_hotkey():
    """Toggle recording state when hotkey is pressed."""
    global recording_active, recording_thread

    # Add timestamp to prevent multiple triggers in quick succession
    current_time = time.time()
    if hasattr(start_stop_recording_hotkey, "last_trigger_time"):
        # Ignore triggers that happen within 1 second of the last one
        if current_time - start_stop_recording_hotkey.last_trigger_time < 1.0:
            logging.info("Ignoring hotkey trigger (debounce)")
            return

    # Update the last trigger time
    start_stop_recording_hotkey.last_trigger_time = current_time

    logging.info(f"Toggle recording state. Current state: {recording_active}")

    if not recording_active:
        # Start recording
        play_sound(START_SOUND)
        recording_active = True
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        click.echo("🔴 Recording started via hotkey...")
    else:
        # Stop recording
        recording_active = False
        click.echo("Recording stopped via hotkey...")
        play_sound(STOP_SOUND)

        # Process the recording
        process_recording()


# Initialize the last trigger time
start_stop_recording_hotkey.last_trigger_time = 0


def setup_hotkey_listener(hotkey_combo):
    """Set up a keyboard listener for the specified hotkey combination."""
    try:
        # Parse the hotkey string into a list of keys
        keys = hotkey_combo.split("+")
        required_keys = set()

        # Map common key name variations
        key_mapping = {
            "cmd": keyboard.Key.cmd,
            "cmd_l": keyboard.Key.cmd,
            "command": keyboard.Key.cmd,
            "alt": keyboard.Key.alt,
            "option": keyboard.Key.alt,
            "ctrl": keyboard.Key.ctrl,
            "control": keyboard.Key.ctrl,
            "shift": keyboard.Key.shift,
        }

        for k in keys:
            k_lower = k.lower()
            if k_lower in key_mapping:
                required_keys.add(key_mapping[k_lower])
            elif hasattr(keyboard.Key, k_lower):
                required_keys.add(getattr(keyboard.Key, k_lower))
            else:
                required_keys.add(k)

        logging.info(f"Setting up hotkey with keys: {required_keys}")

        # Track currently pressed keys and last hotkey trigger time
        currently_pressed = set()
        last_trigger_time = 0

        def on_press(key):
            try:
                logging.info(f"Key pressed: {key}")
                currently_pressed.add(key)

                # Check if all required keys are pressed
                if required_keys.issubset(currently_pressed):
                    nonlocal last_trigger_time
                    current_time = time.time()
                    # Debounce at the listener level too
                    if current_time - last_trigger_time > 1.0:
                        logging.info("Hotkey combination detected!")
                        last_trigger_time = current_time
                        start_stop_recording_hotkey()
            except Exception as e:
                logging.error(f"Error in hotkey press handler: {e}")

        def on_release(key):
            try:
                logging.info(f"Key released: {key}")
                if key in currently_pressed:
                    currently_pressed.remove(key)

                # Stop listener if esc is pressed
                if key == keyboard.Key.esc:
                    logging.info("ESC key pressed, stopping listener")
                    # Instead of returning False, stop the listener directly
                    listener.stop()
            except Exception as e:
                logging.error(f"Error in hotkey release handler: {e}")

            # Don't return anything (implicitly returns None)

        # Create the listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        # Start the listener
        listener.start()
        logging.info(f"Keyboard listener started successfully: {listener}")
        return listener
    except Exception as e:
        logging.error(f"Failed to set up hotkey listener: {e}")
        return None


def process_recording():
    """Process the recorded audio and transcribe it."""
    global frames, p, stream

    if not frames:
        click.echo("No audio recorded.")
        return

    # Close audio resources
    if stream:
        stream.stop_stream()
        stream.close()
    if p:
        p.terminate()

    click.echo("🎙️ Processing audio...")

    # Convert audio data to numpy array
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio_data_fp32 = audio_data.astype(np.float32) / 32768.0

    # Transcribe Audio
    logging.info(f"Starting transcription with model {current_model}...")
    start_time = time.time()
    result = mlx_whisper.transcribe(
        audio_data_fp32,
        language=current_language,
        path_or_hf_repo=current_model,
    )
    end_time = time.time()
    logging.info(f"Transcription finished in {end_time - start_time:.2f} seconds.")

    transcribed_text = result.get("text", "")
    if isinstance(transcribed_text, list):
        transcribed_text = " ".join(transcribed_text)
    transcribed_text = transcribed_text.strip()

    # Output Result
    if transcribed_text:
        click.echo("\n--- Transcription ---")
        click.echo(transcribed_text)
        click.echo("---------------------\n")

        # Copy to clipboard if requested
        if current_copy_flag:
            copy_to_clipboard(transcribed_text)
    else:
        click.echo("🔇 Transcription result was empty.")

    # Reset for next recording
    frames = []

    # Reinitialize audio for next recording
    initialize_audio()


def copy_to_clipboard(text):
    """Copy text to clipboard using pbcopy."""
    pbcopy_path = shutil.which("pbcopy")
    if pbcopy_path:
        try:
            process = subprocess.Popen(pbcopy_path, stdin=subprocess.PIPE, text=True)
            process.communicate(input=text)
            click.echo("✅ Text copied to clipboard.")
        except Exception as e:
            logging.error(f"Failed to copy to clipboard using pbcopy: {e}")
            click.echo("⚠️ Failed to copy text to clipboard.", err=True)
    else:
        click.echo("⚠️ 'pbcopy' command not found. Cannot copy to clipboard.", err=True)


def initialize_audio():
    """Initialize the audio stream for recording."""
    global p, stream

    p = pyaudio.PyAudio()
    stream = p.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    logging.info("Audio stream opened.")


# --- Main CLI Application ---
@click.command(
    help="Records audio from the microphone, transcribes it using MLX Whisper, and prints the text."
)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(MLX_WHISPER_MODELS),
    default="mlx-community/whisper-large-v3-mlx",
    help="Specify the MLX Whisper model name or path.",
    show_default=True,
)
@click.option(
    "-l",
    "--language",
    default=None,  # Default to auto-detect
    help='Specify language code for transcription (e.g., "en", "fr", "de"). Default is auto-detect.',
)
@click.option(
    "--copy",
    is_flag=True,
    default=False,
    help="Copy the transcribed text to the clipboard (uses 'pbcopy' if available).",
)
@click.option(
    "--hotkey",
    default="cmd+alt" if platform.system() == "Darwin" else "ctrl+alt",
    help="Enable hotkey mode with the specified key combination (e.g., 'ctrl+shift+d').",
)
def main(model_name, language, copy, hotkey):
    """Runs the recording and transcription process."""
    global recording_active, p, stream, recording_thread, hotkey_listener
    global current_model, current_language, current_copy_flag

    # Store settings in globals for hotkey mode
    current_model = model_name
    current_language = language
    current_copy_flag = copy

    logging.info(f"Using model: {model_name}")
    if language:
        logging.info(f"Specified language: {language}")
    else:
        logging.info("Language: Auto-detect")

    # Check model compatibility if language is specified
    if language and ".en" in model_name and language != "en":
        click.echo(
            f"Error: Cannot use language '{language}' with an English-only model ('{model_name}').",
            err=True,
        )
        return

    recording_thread = None

    # Check if we're in hotkey mode
    if hotkey:
        try:
            click.echo(
                f"🔑 Hotkey mode enabled. Press {hotkey} to start/stop recording."
            )
            click.echo("Press ESC to exit the program.")

            # Initialize audio
            initialize_audio()

            # Set up the hotkey listener
            hotkey_listener = setup_hotkey_listener(hotkey)
            if not hotkey_listener:
                click.echo("Failed to set up hotkey listener. Exiting.", err=True)
                return

            click.echo("Hotkey listener is now active. Waiting for hotkey presses...")

            # Keep the main thread alive until ESC is pressed
            while hotkey_listener.is_alive():
                time.sleep(0.1)

            return
        except KeyboardInterrupt:
            click.echo("Keyboard interrupt received. Exiting.")
            return

    try:
        # --- Initialize Audio ---
        p = pyaudio.PyAudio()
        stream = p.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        logging.info("Audio stream opened.")

        # --- Play start sound and start recording ---
        play_sound(START_SOUND)
        recording_active = True
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()
        click.echo("🔴 Recording... Press Enter to stop.")

        # Wait for user to press Enter
        input()  # This blocks until Enter is pressed
        recording_active = False
        logging.info("Stopping recording...")

        # Wait for the recording thread to finish processing last chunks
        if recording_thread:
            recording_thread.join(timeout=2)  # Add a timeout
        logging.info("Recording stopped.")

        # Play stop sound
        play_sound(STOP_SOUND)

        # --- Stop and Close Audio Stream ---
        if stream:
            stream.stop_stream()
            stream.close()
            logging.info("Audio stream closed.")
        if p:
            p.terminate()
            logging.info("PyAudio terminated.")

        if not frames:
            click.echo("No audio recorded.")
            return

        click.echo("🎙️ Processing audio...")

        # --- Convert audio data to numpy array ---
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0
        logging.info(
            f"Audio data converted to numpy array with shape {audio_data_fp32.shape}"
        )

        # --- Transcribe Audio ---
        logging.info(f"Starting transcription with model {model_name}...")
        start_time = time.time()
        result = mlx_whisper.transcribe(
            audio_data_fp32,
            language=language,  # Pass None for auto-detect
            path_or_hf_repo=model_name,
        )
        end_time = time.time()
        logging.info(f"Transcription finished in {end_time - start_time:.2f} seconds.")

        transcribed_text = result.get("text", "")
        if isinstance(transcribed_text, list):
            transcribed_text = " ".join(transcribed_text)
        transcribed_text = transcribed_text.strip()

        # --- Output Result ---
        if transcribed_text:
            click.echo("\n--- Transcription ---")
            click.echo(transcribed_text)
            click.echo("---------------------\n")

            # Copy to clipboard if requested and pbcopy is available
            if copy:
                pbcopy_path = shutil.which("pbcopy")
                if pbcopy_path:
                    try:
                        process = subprocess.Popen(
                            pbcopy_path, stdin=subprocess.PIPE, text=True
                        )
                        process.communicate(input=transcribed_text)
                        click.echo("✅ Text copied to clipboard.")
                    except Exception as e:
                        logging.error(f"Failed to copy to clipboard using pbcopy: {e}")
                        click.echo("⚠️ Failed to copy text to clipboard.", err=True)
                else:
                    click.echo(
                        "⚠️ 'pbcopy' command not found. Cannot copy to clipboard.",
                        err=True,
                    )
        else:
            click.echo("🔇 Transcription result was empty.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        click.echo(f"An error occurred: {e}", err=True)
        # Ensure recording is stopped if an error happens mid-way
        recording_active = False
        if recording_thread and recording_thread.is_alive():
            recording_thread.join(timeout=1)

    finally:
        # --- Cleanup ---
        # Ensure stream and PyAudio are closed even on error
        if stream:
            try:
                # Only try to stop if not already stopped
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                logging.debug(f"Error during stream cleanup: {e}")
                pass
        if p:
            try:
                p.terminate()
            except Exception as e:
                logging.debug(f"Error terminating PyAudio: {e}")
                pass

        # Stop hotkey listener if active
        if hotkey_listener:
            try:
                hotkey_listener.stop()
            except Exception as e:
                logging.debug(f"Error stopping hotkey listener: {e}")
                pass


if __name__ == "__main__":
    main()
