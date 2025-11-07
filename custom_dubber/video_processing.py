# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import time
import warnings

from typing import Final

from moviepy import (
    AudioFileClip,
    VideoFileClip,
)

_DEFAULT_FPS: Final[int] = 30
_DEFAULT_DUBBED_VIDEO_FILE: Final[str] = "dubbed_video"
_DEFAULT_OUTPUT_FORMAT: Final[str] = ".mp4"


class VideoProcessing:
    @staticmethod
    def split_audio_video(*, video_file: str, output_directory: str) -> tuple[str, str]:
        """Splits an audio/video file into separate audio and video files."""

        base_filename = os.path.basename(video_file)
        filename, _ = os.path.splitext(base_filename)
        with VideoFileClip(video_file) as video_clip, warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            audio_clip = video_clip.audio
            audio_output_file = os.path.join(output_directory, filename + "_audio.mp3")
            audio_clip.write_audiofile(audio_output_file, logger=None)
            video_clip_without_audio = video_clip.with_audio(None)
            fps = video_clip.fps or _DEFAULT_FPS

            video_output_file = os.path.join(output_directory, filename + "_video.mp4")
            video_clip_without_audio.write_videofile(
                video_output_file, codec="libx264", fps=fps, logger=None
            )
        return video_output_file, audio_output_file

    @staticmethod
    def combine_audio_video(
        *,
        video_file: str,
        dubbed_audio_file: str,
        output_directory: str,
        target_language: str,
    ) -> str:
        """Combines an audio file with a video file, ensuring they have the same duration.

        Returns:
          The path to the output video file with dubbed audio.
        """
        print("Combining audio and video...")

        # Get video and audio durations using MoviePy
        video = VideoFileClip(video_file)
        audio = AudioFileClip(dubbed_audio_file)
        video_duration = video.duration
        audio_duration = audio.duration
        video.close()
        audio.close()

        target_language_suffix = "_" + target_language.replace("-", "_").lower()
        dubbed_video_file = os.path.join(
            output_directory,
            _DEFAULT_DUBBED_VIDEO_FILE
            + target_language_suffix
            + _DEFAULT_OUTPUT_FORMAT,
        )

        print("Started final video rendering...")
        start_time = time.time()

        # Use FFmpeg directly for much faster processing with stream copying
        # This avoids re-encoding the video, making it 10-100x faster
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file without asking
            "-i",
            video_file,  # Input video
            "-i",
            dubbed_audio_file,  # Input audio
            "-map",
            "0:v:0",  # Map video stream from first input
            "-map",
            "1:a:0",  # Map audio stream from second input
            "-c:v",
            "copy",  # Copy video codec (no re-encoding)
            "-c:a",
            "aac",  # Encode audio to AAC
            "-b:a",
            "192k",  # Audio bitrate
            "-shortest",  # Finish encoding when shortest stream ends
            "-movflags",
            "+faststart",  # Enable streaming optimization (moov atom at start)
            "-threads",
            "0",  # Use all available CPU threads
            dubbed_video_file,
        ]

        # If audio is shorter than video, we need to pad it with silence
        if audio_duration < video_duration:
            print(
                f"Audio is {video_duration - audio_duration:.2f}s shorter than video, padding with silence..."
            )
            # Create a temporary audio file with silence padding
            temp_audio = os.path.join(output_directory, "temp_padded_audio.aac")
            pad_duration = video_duration - audio_duration

            # Use FFmpeg to pad audio with silence
            pad_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                dubbed_audio_file,
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={pad_duration}",
                "-filter_complex",
                "[0:a][1:a]concat=n=2:v=0:a=1",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                temp_audio,
            ]
            subprocess.run(
                pad_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Update the command to use padded audio
            ffmpeg_cmd[4] = temp_audio

        # Run FFmpeg command
        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Clean up temporary file if created
            if audio_duration < video_duration:
                temp_audio = os.path.join(output_directory, "temp_padded_audio.aac")
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

        except subprocess.CalledProcessError as e:
            print(f"Error during video rendering: {e}")
            raise

        end_time = time.time()
        print(
            f"Final video rendering completed in {end_time - start_time:.2f} seconds."
        )
        return dubbed_video_file
