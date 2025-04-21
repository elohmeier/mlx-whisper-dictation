import asyncio
import concurrent.futures
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
import wave

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
COMPLETE_SOUND = "/System/Library/Sounds/Ping.aiff"


class DictationApp:
    """Main application class for MLX Whisper Dictation."""

    def __init__(
        self,
        model_name: str,
        language: str | None,
        copy_flag: bool,
        hotkey: str,
        debug: bool,
    ):
        # Configuration
        self.model_name = model_name
        self.language = language
        self.copy_flag = copy_flag
        self.hotkey = hotkey
        self.debug = debug

        # State
        self.recording = False
        self.frames: list[bytes] = []
        self.p: pyaudio.PyAudio | None = None
        self.stream = None
        self.hotkey_listener = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.last_hotkey_trigger = 0
        self.currently_pressed: set = set()
        self.required_keys: set = set()
        self.hotkey_queue = asyncio.Queue()  # Initialize the queue here

        # Set up logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug mode enabled - will play back recorded audio")

    async def play_sound(self, sound_file):
        """Play a notification sound using macOS afplay command."""
        if not os.path.exists(sound_file):
            logging.warning(f"Sound file not found: {sound_file}")
            return

        def _play():
            try:
                # Use the macOS afplay command which can handle AIFF files
                subprocess.run(["afplay", sound_file], check=True)
            except Exception as e:
                logging.error(f"Error playing sound: {e}")

        # Run in executor since subprocess is blocking
        await asyncio.get_event_loop().run_in_executor(self.executor, _play)

    async def initialize_audio(self):
        """Initialize the audio stream for recording."""

        def _init_audio():
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            return self.stream

        # Run in executor since PyAudio is blocking
        await asyncio.get_event_loop().run_in_executor(self.executor, _init_audio)
        logging.info("Audio stream opened.")

    async def record_audio(self):
        """Reads audio chunks from the stream and stores them in frames."""
        logging.info("Recording started.")
        self.frames = []  # Reset frames at the start of recording

        def _read_chunk():
            if self.stream is None:
                return None
            try:
                return self.stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                # Ignore input overflow errors, log others
                if e.errno != pyaudio.paInputOverflowed:
                    logging.error(f"Recording IO error: {e}")
                return None
            except Exception as e:
                logging.error(f"Unexpected recording error: {e}", exc_info=True)
                return None

        while self.recording:
            data = await asyncio.get_event_loop().run_in_executor(
                self.executor, _read_chunk
            )
            if data:
                self.frames.append(data)
            # Small sleep to prevent CPU hogging
            await asyncio.sleep(0.001)

        logging.info("Recording finished.")

    async def toggle_recording(self):
        """Toggle recording state when hotkey is pressed."""
        # Add timestamp to prevent multiple triggers in quick succession
        current_time = time.time()
        if current_time - self.last_hotkey_trigger < 1.0:
            logging.info("Ignoring hotkey trigger (debounce)")
            return

        # Update the last trigger time
        self.last_hotkey_trigger = current_time

        logging.info(f"Toggle recording state. Current state: {self.recording}")

        if not self.recording:
            # Start recording first
            self.recording = True
            # Start recording in a task
            asyncio.create_task(self.record_audio())
            # Then play sound
            await self.play_sound(START_SOUND)
            click.echo("🔴 Recording started via hotkey...")
        else:
            # Stop recording first
            self.recording = False
            click.echo("Recording stopped via hotkey...")
            # Then play sound
            await self.play_sound(STOP_SOUND)
            # Process the recording
            await self.process_recording()

    async def setup_hotkey_listener(self):
        """Set up a keyboard listener for the specified hotkey combination."""
        try:
            # Parse the hotkey string into a list of keys
            keys = self.hotkey.split("+")

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
                    self.required_keys.add(key_mapping[k_lower])
                elif hasattr(keyboard.Key, k_lower):
                    self.required_keys.add(getattr(keyboard.Key, k_lower))
                else:
                    self.required_keys.add(k)

            logging.info(f"Setting up hotkey with keys: {self.required_keys}")

            # Queue is already initialized in __init__

            def on_press(key):
                try:
                    logging.debug(f"Key pressed: {key}")
                    self.currently_pressed.add(key)

                    # Check if all required keys are pressed
                    if self.required_keys.issubset(self.currently_pressed):
                        current_time = time.time()
                        # Debounce at the listener level too
                        if current_time - self.last_hotkey_trigger > 1.0:
                            logging.info("Hotkey combination detected!")
                            # Just put a message in the queue instead of trying to create a task
                            try:
                                # Use a non-blocking put to avoid hanging if queue is full
                                self.hotkey_queue.put_nowait("toggle")
                            except asyncio.QueueFull:
                                logging.warning(
                                    "Hotkey queue is full, ignoring hotkey press"
                                )
                except Exception as e:
                    logging.error(f"Error in hotkey press handler: {e}")

            def on_release(key):
                try:
                    logging.debug(f"Key released: {key}")
                    if key in self.currently_pressed:
                        self.currently_pressed.remove(key)

                    # Stop listener if esc is pressed
                    if key == keyboard.Key.esc:
                        logging.info("ESC key pressed, stopping listener")
                        # Instead of returning False, stop the listener directly
                        if self.hotkey_listener:
                            self.hotkey_listener.stop()
                except Exception as e:
                    logging.error(f"Error in hotkey release handler: {e}")

            # Create and start the listener in a separate thread
            def _setup_listener():
                self.hotkey_listener = keyboard.Listener(
                    on_press=on_press, on_release=on_release
                )
                self.hotkey_listener.start()
                return self.hotkey_listener

            await asyncio.get_event_loop().run_in_executor(
                self.executor, _setup_listener
            )

            logging.info(
                f"Keyboard listener started successfully: {self.hotkey_listener}"
            )
            return self.hotkey_listener
        except Exception as e:
            logging.error(f"Failed to set up hotkey listener: {e}")
            return None

    async def play_recorded_audio(self):
        """Play back the recorded audio for debugging purposes."""
        if not self.frames:
            click.echo("No audio to play back.")
            return

        def _play_audio():
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Write the audio frames to the WAV file
            with wave.open(temp_filename, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.frames))

            try:
                # Use the macOS afplay command to play the WAV file
                subprocess.run(["afplay", temp_filename], check=True)
                return True
            except Exception as e:
                logging.error(f"Error playing back audio: {e}")
                return False
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    logging.debug(f"Error removing temporary audio file: {e}")

        click.echo("🔊 Playing back recorded audio...")
        success = await asyncio.get_event_loop().run_in_executor(
            self.executor, _play_audio
        )

        if success:
            click.echo("✅ Audio playback complete.")
        else:
            click.echo("⚠️ Failed to play back audio.", err=True)

    async def process_recording(self):
        """Process the recorded audio and transcribe it."""
        if not self.frames:
            click.echo("No audio recorded.")
            return

        # Close audio resources
        def _close_audio():
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()

        await asyncio.get_event_loop().run_in_executor(self.executor, _close_audio)

        click.echo("🎙️ Processing audio...")

        # Play back the audio if debug mode is enabled
        if self.debug:
            await self.play_recorded_audio()

        # Convert audio data to numpy array
        audio_data = np.frombuffer(b"".join(self.frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0

        # Transcribe Audio - run in executor since it's CPU intensive
        def _transcribe():
            logging.info(f"Starting transcription with model {self.model_name}...")
            start_time = time.time()
            result = mlx_whisper.transcribe(
                audio_data_fp32,
                language=self.language,
                path_or_hf_repo=self.model_name,
            )
            end_time = time.time()
            logging.info(
                f"Transcription finished in {end_time - start_time:.2f} seconds."
            )
            return result

        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _transcribe
        )

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
            if self.copy_flag:
                await self.copy_to_clipboard(transcribed_text)

            # Play completion sound to signal transcription is done
            await self.play_sound(COMPLETE_SOUND)
        else:
            click.echo("🔇 Transcription result was empty.")

        # Reset for next recording
        self.frames = []

        # Reinitialize audio for next recording
        await self.initialize_audio()

    async def copy_to_clipboard(self, text):
        """Copy text to clipboard using pbcopy."""

        def _copy():
            pbcopy_path = shutil.which("pbcopy")
            if pbcopy_path:
                try:
                    process = subprocess.Popen(
                        pbcopy_path, stdin=subprocess.PIPE, text=True
                    )
                    process.communicate(input=text)
                    return True
                except Exception as e:
                    logging.error(f"Failed to copy to clipboard using pbcopy: {e}")
                    return False
            return None

        result = await asyncio.get_event_loop().run_in_executor(self.executor, _copy)

        if result is True:
            click.echo("✅ Text copied to clipboard.")
        elif result is False:
            click.echo("⚠️ Failed to copy text to clipboard.", err=True)
        else:
            click.echo(
                "⚠️ 'pbcopy' command not found. Cannot copy to clipboard.", err=True
            )

    async def cleanup(self):
        """Clean up resources."""
        logging.debug("Starting cleanup...")

        # First stop the executor from accepting new tasks
        self.executor.shutdown(wait=False)

        def _cleanup():
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logging.debug(f"Error during stream cleanup: {e}")

            if self.p:
                try:
                    self.p.terminate()
                except Exception as e:
                    logging.debug(f"Error terminating PyAudio: {e}")

            if self.hotkey_listener:
                try:
                    self.hotkey_listener.stop()
                except Exception as e:
                    logging.debug(f"Error stopping hotkey listener: {e}")

        # Create a new executor just for cleanup
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as cleanup_executor:
            await asyncio.get_event_loop().run_in_executor(cleanup_executor, _cleanup)

        logging.debug("Cleanup completed")

    async def run_hotkey_mode(self):
        """Run the application in hotkey mode."""
        try:
            click.echo(
                f"🔑 Hotkey mode enabled. Press {self.hotkey} to start/stop recording."
            )
            click.echo("Press ESC to exit the program.")

            # Initialize audio
            await self.initialize_audio()

            # Set up the hotkey listener
            listener = await self.setup_hotkey_listener()
            if not listener:
                click.echo("Failed to set up hotkey listener. Exiting.", err=True)
                return

            click.echo("Hotkey listener is now active. Waiting for hotkey presses...")

            # Process hotkey events from the queue
            while listener.is_alive():
                try:
                    # Wait for a short time to check for hotkey events
                    # This allows us to also check if the listener is still alive
                    message = await asyncio.wait_for(
                        self.hotkey_queue.get(), timeout=0.1
                    )
                    if message == "toggle":
                        await self.toggle_recording()
                    self.hotkey_queue.task_done()
                except TimeoutError:
                    # No hotkey event, just continue the loop
                    pass
                except Exception as e:
                    logging.error(f"Error processing hotkey event: {e}")

        except asyncio.CancelledError:
            logging.info("Hotkey mode task cancelled")
        finally:
            await self.cleanup()

    async def run_interactive_mode(self):
        """Run the application in interactive mode (press Enter to stop)."""
        try:
            # Initialize audio
            await self.initialize_audio()

            # Start recording
            self.recording = True
            recording_task = asyncio.create_task(self.record_audio())

            # Play start sound
            await self.play_sound(START_SOUND)
            click.echo("🔴 Recording... Press Enter to stop.")

            # Wait for user to press Enter (in a non-blocking way)
            def _wait_for_enter():
                input()  # This blocks until Enter is pressed
                return True

            # Run the blocking input() in a separate thread
            await asyncio.get_event_loop().run_in_executor(
                self.executor, _wait_for_enter
            )

            # Stop recording
            self.recording = False
            logging.info("Stopping recording...")

            # Wait for recording task to complete
            try:
                await asyncio.wait_for(recording_task, timeout=2)
            except TimeoutError:
                logging.warning("Recording task did not complete in time")

            # Play stop sound
            await self.play_sound(STOP_SOUND)

            # Process the recording
            await self.process_recording()

        except asyncio.CancelledError:
            logging.info("Interactive mode task cancelled")
            self.recording = False
        finally:
            await self.cleanup()

    async def run(self):
        """Run the application in the appropriate mode."""
        logging.info(f"Using model: {self.model_name}")
        if self.language:
            logging.info(f"Specified language: {self.language}")
        else:
            logging.info("Language: Auto-detect")

        # Check model compatibility if language is specified
        if self.language and ".en" in self.model_name and self.language != "en":
            click.echo(
                f"Error: Cannot use language '{self.language}' with an English-only model ('{self.model_name}').",
                err=True,
            )
            return

        try:
            if self.hotkey:
                await self.run_hotkey_mode()
            else:
                await self.run_interactive_mode()
        except KeyboardInterrupt:
            click.echo("Keyboard interrupt received. Exiting.")


# --- Main CLI Application ---
@click.command(
    help="Records audio from the microphone, transcribes it using MLX Whisper, and prints the text."
)
@click.option(
    "-m",
    "--model-name",
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
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with audio playback after recording.",
)
def main(model_name, language, copy, hotkey, debug):
    """Runs the recording and transcription process."""
    # Create the application instance
    app = DictationApp(
        model_name=model_name,
        language=language,
        copy_flag=copy,
        hotkey=hotkey,
        debug=debug,
    )

    # Run the application with asyncio
    try:
        asyncio.run(app.run())
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        click.echo(f"An error occurred: {e}", err=True)


if __name__ == "__main__":
    main()
