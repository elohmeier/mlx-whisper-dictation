import platform
import threading
import time

import click
import mlx_whisper
import numpy as np
import pyaudio
import rumps
from pynput import keyboard


class SpeechTranscriber:
    # removed model from arguments
    def __init__(self, model_name):
        # self.model = model
        self.pykeyboard = keyboard.Controller()
        self.model_name = model_name

    def transcribe(self, audio_data, language=None):
        # changed because of MLX
        result = mlx_whisper.transcribe(
            audio_data, language=language, path_or_hf_repo=self.model_name
        )

        is_first = True
        for element in result["text"]:
            if is_first and element == " ":
                is_first = False
                continue

            try:
                self.pykeyboard.type(element)
                time.sleep(0.0025)
            except:
                pass


class Recorder:
    def __init__(self, transcriber):
        self.recording = False
        self.transcriber = transcriber

    def start(self, language=None):
        thread = threading.Thread(target=self._record_impl, args=(language,))
        thread.start()

    def stop(self):
        self.recording = False

    def _record_impl(self, language):
        self.recording = True
        frames_per_buffer = 1024
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            frames_per_buffer=frames_per_buffer,
            input=True,
        )
        frames = []

        while self.recording:
            data = stream.read(frames_per_buffer)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0
        self.transcriber.transcribe(audio_data_fp32, language)


class GlobalKeyListener:
    def __init__(self, app, key_combination):
        self.app = app
        self.key1, self.key2 = self.parse_key_combination(key_combination)
        self.key1_pressed = False
        self.key2_pressed = False

    def parse_key_combination(self, key_combination):
        key1_name, key2_name = key_combination.split("+")
        key1 = getattr(keyboard.Key, key1_name, keyboard.KeyCode(char=key1_name))
        key2 = getattr(keyboard.Key, key2_name, keyboard.KeyCode(char=key2_name))
        return key1, key2

    def on_key_press(self, key):
        if key == self.key1:
            self.key1_pressed = True
        elif key == self.key2:
            self.key2_pressed = True

        if self.key1_pressed and self.key2_pressed:
            self.app.toggle()

    def on_key_release(self, key):
        if key == self.key1:
            self.key1_pressed = False
        elif key == self.key2:
            self.key2_pressed = False


class DoubleCommandKeyListener:
    def __init__(self, app):
        self.app = app
        self.key = keyboard.Key.cmd_r
        self.pressed = 0
        self.last_press_time = 0

    def on_key_press(self, key):
        is_listening = self.app.started
        if key == self.key:
            current_time = time.time()
            if (
                not is_listening and current_time - self.last_press_time < 0.5
            ):  # Double click to start listening
                self.app.toggle()
            elif is_listening:  # Single click to stop listening
                self.app.toggle()
            self.last_press_time = current_time

    def on_key_release(self, key):
        pass


class StatusBarApp(rumps.App):
    def __init__(self, recorder, languages=None, max_time=None):
        super().__init__("whisper", "⏯")
        self.languages = languages
        self.current_language = languages[0] if languages is not None else None

        menu = [
            "Start Recording",
            "Stop Recording",
            None,
        ]

        if languages is not None:
            for lang in languages:
                callback = (
                    self.change_language if lang != self.current_language else None
                )
                menu.append(rumps.MenuItem(lang, callback=callback))
            menu.append(None)

        self.menu = menu
        self.menu["Stop Recording"].set_callback(None)

        self.started = False
        self.recorder = recorder
        self.max_time = max_time
        self.timer = None
        self.elapsed_time = 0

    def change_language(self, sender):
        self.current_language = sender.title
        for lang in self.languages:
            self.menu[lang].set_callback(
                self.change_language if lang != self.current_language else None
            )

    @rumps.clicked("Start Recording")
    def start_app(self, _):
        click.echo("Listening...")
        self.started = True
        self.menu["Start Recording"].set_callback(None)
        self.menu["Stop Recording"].set_callback(self.stop_app)
        self.recorder.start(self.current_language)

        if self.max_time is not None:
            self.timer = threading.Timer(self.max_time, lambda: self.stop_app(None))
            self.timer.start()

        self.start_time = time.time()
        self.update_title()

    @rumps.clicked("Stop Recording")
    def stop_app(self, _):
        if not self.started:
            return

        if self.timer is not None:
            self.timer.cancel()

        click.echo("Transcribing...")
        self.title = "⏯"
        self.started = False
        self.menu["Stop Recording"].set_callback(None)
        self.menu["Start Recording"].set_callback(self.start_app)
        self.recorder.stop()
        click.echo("Done.\n")

    def update_title(self):
        if self.started:
            self.elapsed_time = int(time.time() - self.start_time)
            minutes, seconds = divmod(self.elapsed_time, 60)
            self.title = f"({minutes:02d}:{seconds:02d}) 🔴"
            threading.Timer(1, self.update_title).start()

    def toggle(self):
        if self.started:
            self.stop_app(None)
        else:
            self.start_app(None)


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


@click.command(
    help="Dictation app using the MLX OpenAI Whisper model. By default the keyboard shortcut cmd+option starts and stops dictation"
)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(MLX_WHISPER_MODELS),
    default="mlx-community/whisper-large-v3-mlx",
    help="""Specify the MLX Whisper model to use. Example: mlx-community/whisper-large-v3-mlx.
    To see the most up to date list of models visit https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc?utm_source=chatgpt.com. 
    Note that the models ending in .en are trained only on English speech and will perform better on English language.""",
)
@click.option(
    "-k",
    "--key-combination",
    default=lambda: "cmd_l+alt" if platform.system() == "Darwin" else "ctrl+alt",
    help="Specify the key combination to toggle the app. Example: cmd_l+alt for macOS ctrl+alt for other platforms. Default: cmd_l+alt (macOS) or ctrl+alt (others).",
)
@click.option(
    "--k-double-cmd",
    is_flag=True,
    help="If set, use double Right Command key press on macOS to toggle the app (double click to begin recording, single click to stop recording). Ignores the --key-combination argument.",
)
@click.option(
    "-l",
    "--language",
    help='Specify the two-letter language code (e.g., "en" for English) to improve recognition accuracy. This can be especially helpful for smaller model sizes. Multiple languages can be specified with commas (e.g., "en,fr,de").',
)
@click.option(
    "-t",
    "--max-time",
    type=float,
    default=30.0,
    help="Specify the maximum recording time in seconds. The app will automatically stop recording after this duration. Default: 30 seconds.",
)
def main(model_name, key_combination, k_double_cmd, language, max_time):
    """Run the MLX Whisper dictation app."""

    if language is not None:
        language = language.split(",")

    if (
        model_name.endswith(".en")
        and language is not None
        and any(lang != "en" for lang in language)
    ):
        raise click.UsageError(
            "If using a model ending in .en, you cannot specify a language other than English."
        )

    transcriber = SpeechTranscriber(model_name)
    recorder = Recorder(transcriber)

    app = StatusBarApp(recorder, language, max_time)
    if k_double_cmd:
        key_listener = DoubleCommandKeyListener(app)
    else:
        key_listener = GlobalKeyListener(app, key_combination)
    listener = keyboard.Listener(
        on_press=key_listener.on_key_press, on_release=key_listener.on_key_release
    )
    listener.start()

    click.echo("Running...")
    app.run()


if __name__ == "__main__":
    main()
