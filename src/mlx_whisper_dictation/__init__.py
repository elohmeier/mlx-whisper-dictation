import logging
import os
import platform
import tempfile
import time
import wave

import click
import mlx_whisper
import pyaudio
import rumps
from pynput import keyboard

# Configure basic logging with thread ID
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Thread %(thread)d] - %(message)s",
)

# rumps.debug_mode(True)


class StatusBarApp(rumps.App):
    def __init__(
        self,
        model_name,
        key_combination=None,
        use_double_cmd=False,
        languages=None,
    ):
        super().__init__("whisper", "⏯")
        # Model and keyboard settings
        self.model_name = model_name
        self.pykeyboard = keyboard.Controller()

        # Keyboard listener settings
        self.use_double_cmd = use_double_cmd
        self.last_press_time = 0
        self.cmd_key = keyboard.Key.cmd_r
        self.key1_pressed = False
        self.key2_pressed = False
        self._toggle_triggered = False

        # Parse key combination if provided
        if not use_double_cmd and key_combination:
            try:
                key1_name, key2_name = key_combination.split("+")
                self.key1 = getattr(
                    keyboard.Key, key1_name, keyboard.KeyCode.from_char(key1_name)
                )
                self.key2 = getattr(
                    keyboard.Key, key2_name, keyboard.KeyCode.from_char(key2_name)
                )
            except Exception as e:
                logging.error(f"Failed to parse key combination: {e}")
                raise ValueError(f"Invalid key combination: {key_combination}") from e

        # Language settings
        self.languages = languages or []
        self.current_language = self.languages[0] if self.languages else None

        # Recording state
        self.started = False
        self.recording = False
        self.frames = []

        # Audio resources
        try:
            self.p = pyaudio.PyAudio()
            self.stream = None
        except Exception as e:
            logging.error(f"Failed to initialize PyAudio: {e}", exc_info=True)
            self.p = None
            raise RuntimeError("Could not initialize audio system.") from e

        # Timers
        self.title_update_timer = None
        self.record_timer = None
        self.start_time = 0
        self.elapsed_time = 0

        # Build menu
        menu_items = [
            rumps.MenuItem("Start Recording", callback=self.start_app),
            rumps.MenuItem("Stop Recording", callback=None),
            None,
        ]

        # Add language menu items if needed
        if self.languages:
            self.language_menu_items = {}
            for lang in self.languages:
                item = rumps.MenuItem(lang, callback=self.change_language)
                item.state = 1 if lang == self.current_language else 0
                self.language_menu_items[lang] = item
                menu_items.append(item)
            menu_items.append(None)

        self.quit_button = rumps.MenuItem("Quit", callback=self.quit_app)
        self.menu = menu_items

    def change_language(self, sender):
        new_language = sender.title
        if new_language == self.current_language:
            return

        # Update language selection
        if self.current_language in self.language_menu_items:
            self.language_menu_items[self.current_language].state = 0
        self.current_language = new_language
        sender.state = 1

    def start_app(self, sender):
        if self.started or not self.p:
            return

        self.started = True
        self.recording = True
        self.frames = []
        self.menu["Start Recording"].set_callback(None)
        self.menu["Stop Recording"].set_callback(self.stop_app)

        # Initialize audio stream
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                frames_per_buffer=1024,
                input=True,
            )
        except Exception as e:
            logging.error(f"Failed to open audio stream: {e}", exc_info=True)
            self.stop_app(None)
            return

        # Start recording timer (runs every 0.05 seconds to collect audio)
        if self.record_timer:
            self.record_timer.stop()
        self.record_timer = rumps.Timer(self._record_audio_chunk, 0.05)
        self.record_timer.start()

        # Start updating title
        self.start_time = time.time()
        self.elapsed_time = 0

        # Create and start the title update timer
        if self.title_update_timer:
            self.title_update_timer.stop()
        self.title_update_timer = rumps.Timer(self.update_title, 1)
        self.title_update_timer.start()
        self.update_title()  # Update immediately

    def _record_audio_chunk(self, timer=None):
        if not self.recording or not self.stream or self.p is None:
            return

        try:
            data = self.stream.read(1024, exception_on_overflow=False)
            self.frames.append(data)
        except Exception as e:
            logging.error(f"Recording error: {e}", exc_info=True)
            # Don't stop recording on a single error, just log it

    def _transcribe_audio(self, audio_file_path):
        logging.info(f"Starting transcription for model {self.model_name}...")
        try:
            result = mlx_whisper.transcribe(
                audio_file_path,
                language=self.current_language,
                path_or_hf_repo=self.model_name,
            )

            text_to_type = result.get("text", "").strip()  # type: ignore
            if not text_to_type:
                logging.warning("Transcription result was empty.")
                return

            logging.info(f"Typing result: '{text_to_type[:50]}...'")
            for char in text_to_type:
                self.pykeyboard.type(char)
                time.sleep(0.002)
        except Exception as e:
            logging.error(f"Error during transcription: {e}", exc_info=True)

    def stop_app(self, sender):
        if not self.started:
            return

        # Cancel timers
        if self.title_update_timer:
            self.title_update_timer.stop()

        if self.record_timer:
            self.record_timer.stop()

        # Update UI state
        self.title = "Processing..."
        self.started = False
        self.recording = False
        self.menu["Stop Recording"].set_callback(None)
        self.menu["Start Recording"].set_callback(self.start_app)

        # Stop and close the audio stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logging.error(f"Error closing stream: {e}")

        # Process the recorded audio if we have any frames
        if self.frames:
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_filename = temp_file.name

                    with wave.open(temp_filename, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(b"".join(self.frames))

                self._transcribe_audio(temp_filename)

                try:
                    os.unlink(temp_filename)
                except Exception:
                    pass
            except Exception as e:
                logging.error(f"Transcription error: {e}", exc_info=True)
            finally:
                self.frames = []

        self.title = "⏯"

    def update_title(self, sender=None):
        if self.started:
            self.elapsed_time = int(time.time() - self.start_time)
            minutes, seconds = divmod(self.elapsed_time, 60)
            self.title = f"({minutes:02d}:{seconds:02d}) 🔴"

    def toggle(self):
        if self.started:
            self.stop_app(None)
        else:
            self.start_app(None)

    def on_key_press(self, key):
        try:
            if self.use_double_cmd:
                self._handle_double_cmd_press(key)
            else:
                self._handle_key_combination_press(key)
        except Exception as e:
            logging.error(f"Error in key press handler: {e}", exc_info=True)

    def _handle_double_cmd_press(self, key):
        if key == self.cmd_key:
            current_time = time.time()
            is_listening = self.started

            if not is_listening and current_time - self.last_press_time < 0.5:
                self.toggle()
                self.last_press_time = 0
            elif is_listening:
                self.toggle()
                self.last_press_time = 0
            else:
                self.last_press_time = current_time

    def _handle_key_combination_press(self, key):
        if key == self.key1:
            self.key1_pressed = True
        elif key == self.key2:
            self.key2_pressed = True

        if self.key1_pressed and self.key2_pressed and not self._toggle_triggered:
            self.toggle()
            self._toggle_triggered = True

    def on_key_release(self, key):
        if not self.use_double_cmd:
            if key == self.key1:
                self.key1_pressed = False
                self._toggle_triggered = False
            elif key == self.key2:
                self.key2_pressed = False
                self._toggle_triggered = False

    def quit_app(self, sender):
        # Stop recording if active
        if self.started:
            self.stop_app(None)

        # Cancel any timers
        if self.title_update_timer:
            self.title_update_timer.stop()

        # Clean up audio resources
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logging.debug(f"Error closing stream: {e}")
            finally:
                self.stream = None

        if self.record_timer:
            self.record_timer.stop()

        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                logging.debug(f"Error terminating PyAudio: {e}")
            finally:
                self.p = None

        # Quit application
        rumps.quit_application()


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


@click.command(help="Dictation app using the MLX OpenAI Whisper model.")
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(MLX_WHISPER_MODELS),
    default="mlx-community/whisper-large-v3-mlx",
    help="Specify the MLX Whisper model name or path.",
)
@click.option(
    "-k",
    "--key-combination",
    default=lambda: "cmd_l+alt" if platform.system() == "Darwin" else "ctrl+alt",
    help="Key combination to toggle recording (e.g., 'cmd_l+alt' or 'ctrl+alt').",
)
@click.option(
    "--k-double-cmd",
    is_flag=True,
    help="Use double Right Command key press on macOS to toggle.",
)
@click.option(
    "-l",
    "--language",
    help='Comma-separated list of language codes (e.g., "en,fr,de"). First is default.',
)
def main(model_name, key_combination, k_double_cmd, language):
    """Run the MLX Whisper dictation app."""
    logging.info(f"Starting with model: {model_name}")

    # Parse languages
    languages = []
    if language:
        languages = [lang.strip() for lang in language.split(",") if lang.strip()]

    # Check model language compatibility
    if ".en" in model_name and languages and any(lang != "en" for lang in languages):
        click.echo(
            "Error: Cannot use non-English languages with an English-only model.",
            err=True,
        )
        return

    # Determine if we should use double cmd
    use_double_cmd = k_double_cmd and platform.system() == "Darwin"

    pynput_listener = None
    app = None

    try:
        # Create the status bar app
        app = StatusBarApp(
            model_name=model_name,
            key_combination=key_combination,
            use_double_cmd=use_double_cmd,
            languages=languages,
        )

        # Create the pynput listener
        pynput_listener = keyboard.Listener(
            on_press=app.on_key_press, on_release=app.on_key_release
        )

        # Start the keyboard listener
        pynput_listener.start()

        # Run the app
        app.run()

    except RuntimeError as e:
        click.echo(f"Error during initialization: {e}", err=True)
        if pynput_listener is not None and pynput_listener.is_alive():
            pynput_listener.stop()
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        logging.error(f"Error: {e}", exc_info=True)
        if app is not None:
            app.quit_app(None)
        if pynput_listener is not None and pynput_listener.is_alive():
            pynput_listener.stop()


if __name__ == "__main__":
    main()
