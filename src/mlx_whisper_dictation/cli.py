import logging
import shutil
import subprocess
import threading
import time

import click
import mlx_whisper
import numpy as np
import pyaudio

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
# --- Globals for Recording Thread ---
frames = []
recording_active = False
p = None
stream = None

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
def main(model_name, language, copy):
    """Runs the recording and transcription process."""
    global recording_active, p, stream

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

        # --- Start Recording ---
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


if __name__ == "__main__":
    main()
