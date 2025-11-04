# YouTube Video Dubber

This project is a command-line tool for automatically dubbing YouTube videos from a source language to a target language. It leverages Google's Gemini AI for translation and text-to-speech, and `yt-dlp` for downloading YouTube videos.

## Features

- **Video and Audio Downloading**: Downloads the specified YouTube video and its audio track.
- **Transcript Generation**: Fetches existing transcripts or generates them if not available.
- **Translation**: Translates the video transcript to a specified target language using Google Gemini.
- **Text-to-Speech**: Converts the translated text into speech using Google Gemini's TTS capabilities.
- **Audio Dubbing**: Merges the generated speech with the original video's background audio.
- **Final Video Creation**: Combines the dubbed audio with the original video to create the final dubbed video.

## Project Inspiration

This project was created so that older generations, like my mom, can enjoy high-quality documentary content from YouTube in their own language, without being restricted by language barriers.

## Requirements

- Python 3.13+
- FFmpeg

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/youtube-nepali.git
    cd youtube-nepali
    ```

2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    _(Note: A `requirements.txt` file is not present, but the dependencies are listed in `pyproject.toml`. The command above is a standard practice.)_

3.  Make sure you have FFmpeg installed and available in your system's PATH.

## Usage

The main script is `custom_dubber/main.py`. You can run it from the command line with the following arguments:

```bash
python -m custom_dubber.main \
    --youtube_id <YOUTUBE_VIDEO_ID> \
    --target_language <TARGET_LANGUAGE_CODE> \
    --api_keys <YOUR_GEMINI_API_KEY_1> <YOUR_GEMINI_API_KEY_2> \
    [--source_language <SOURCE_LANGUAGE_CODE>] \
    [--output_directory <OUTPUT_DIRECTORY>] \
    [--voice <VOICE_NAME>] \
    [--name <FINAL_VIDEO_NAME>] \
    [--move_directory <MOVE_TO_DIRECTORY>] \
    [--cleanup]
```

### Arguments

- `--youtube_id`: (Required) The ID of the YouTube video you want to dub.
- `--target_language`: (Required) The language code of the target language (e.g., `ne` for Nepali).
- `--api_keys`: (Required) One or more Google Gemini API keys.
- `--source_language`: The language code of the source video. Defaults to `en-US`.
- `--output_directory`: The directory where the output files will be saved. Defaults to `output/`.
- `--voice`: The voice to be used for the text-to-speech. Defaults to `Charon`.
- `--name`: The name for the final output video file.
- `--move_directory`: A directory where the final video will be moved.
- `--cleanup`: If set, intermediate files will be deleted after the process is complete.

### Example

```bash
python -m custom_dubber.main \
    --youtube_id "dQw4w9WgXcQ" \
    --target_language "ne" \
    --api_keys "YOUR_API_KEY" \
    --name "dubbed_video" \
    --cleanup
```

## Project Structure

```
.
├── custom_dubber/
│   ├── main.py                   # Main script to run the dubbing process
│   ├── video_downloader.py       # Handles downloading of video, audio, and subtitles
│   ├── translation_gemini.py     # Handles the translation of transcripts using Gemini
│   ├── text_to_speech_gemini.py  # Handles the text-to-speech conversion using Gemini
│   ├── audio_processing.py       # Manages audio-related tasks
│   ├── video_processing.py       # Manages video-related tasks
│   └── ...
├── pyproject.toml                # Project metadata and dependencies
└── README.md                     # This file
```

## Dependencies

- `google-genai`
- `google-generativeai`
- `yt-dlp`
- `moviepy`
- `pydub`
- `youtube-transcript-api`

## How it Works

1.  **Download**: The tool starts by downloading the YouTube video, its audio, and any available subtitles using `yt-dlp`.
2.  **Transcribe/Translate**: If subtitles are not available, it transcribes the audio. The transcript is then translated into the target language using the Google Gemini API.
3.  **Synthesize Speech**: The translated text segments are converted into audio using Google Gemini's text-to-speech models. This is done in parallel to speed up the process.
4.  **Process Audio**: The newly generated audio clips are inserted at their corresponding timestamps into the original audio track.
5.  **Finalize Video**: The final step involves combining the newly created dubbed audio track with the original video file, resulting in a fully dubbed video.
