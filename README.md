# MLX Whisper Dictation - Installation and Usage Guide

## Step 1: Install Homebrew
1. Open your terminal and run:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Add Homebrew to your `PATH`:
   ```bash
   export PATH="/opt/homebrew/bin:$PATH"
   ```

---

## Step 2: Configure Zsh
1. Open the Zsh configuration file:
   ```bash
   nano ~/.zshrc
   ```
2. Add the following line:
   ```bash
   source ~/.zshrc
   ```
3. Save and exit:
   - Press `Ctrl + X`
   - Press `Y`
   - Press `Enter`
4. Reload the configuration:
   ```bash
   source ~/.zshrc
   ```

---

## Step 3: Install Required Packages
Run this command to install the necessary packages:
```bash
brew install portaudio llvm
```

---

## Step 4: Clone the Repository
1. Navigate to your `Documents` folder:
   ```bash
   cd ~/Documents
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/computerstimulation/mlx-whisper-dictation
   ```
3. Navigate into the project folder:
   ```bash
   cd mlx-whisper-dictation
   ```

---

## Step 5: Set Up a Virtual Environment
1. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

---

## Step 6: Install Dependencies
Install the app's required dependencies:
```bash
pip install -r requirements.txt
```
Wait for the dependencies to finish downloading.

---

## Step 7: Run the App
Run the application:
```bash
python whisper-dictation.py
```

---

## Step 8: Use the App
1. Open a text field and place your cursor in it.
2. Press `Command + Option` to start dictation.
3. If prompted with “Terminal would like to access the microphone,” press **Allow**.
4. Speak into your microphone.
5. Press `Command + Option` again to stop dictation.

---

### Notes:
- The first time you use the app, the model may take some time to download.
- The default model is **MLX Whisper Large** (highest quality but slower processing time).
- You can change the model in the app configuration.

If your cursor is on a text field, transcribed text will be automatically pasted.

To stop the app, press `Ctrl + C` in the terminal.