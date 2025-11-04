from pathlib import Path
import yt_dlp
import os


class VideoDownloader:
    def __init__(self, youtube_id, output_dir=None):
        self.youtube_id = youtube_id
        self.output_path = output_dir or os.path.join("__output", self.youtube_id)

    def download_audio(self):
        audio_path = os.path.join(self.output_path, "audio")

        ydl_audio_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "quiet": True,
            "keepvideo": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
        with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={self.youtube_id}"])

        return os.path.join(self.output_path, Path(audio_path).name + ".mp3")

    def download_video(self):
        video_path = os.path.join(self.output_path, "video.mp4")
        if os.path.exists(video_path):
            print(f"Video already downloaded at {video_path}")
        else:
            os.makedirs(self.output_path, exist_ok=True)

            ydl_opts = {
                "format": "bestvideo[height=1080][ext=mp4]/best",
                "outtmpl": video_path,
                "quiet": True,
                "addmetadata": True,
                "embedthumbnail": True,
                "embedsubtitles": True,
                "keepvideo": True,
                "postprocessors": [
                    {
                        "key": "FFmpegVideoConvertor",
                        "preferedformat": "mp4",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={self.youtube_id}"])
        return os.path.join(self.output_path, Path(video_path).stem + ".mp4")

    def download_subtitles(self, language_codes):
        paths = []

        try:
            # fetch available subtitles info first including generated ones
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "json3",
                "subtitleslangs": language_codes,
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={self.youtube_id}", download=False
                )
                available_subs = info_dict.get("subtitles", {})
                print(f"Available subtitles: {available_subs}")
        except Exception as e:
            print(f"Error fetching available subtitles: {e}")
            available_subs = {}

        for lang in language_codes:
            subtitle_path = os.path.join(self.output_path, "subtitles", f"{lang}.json")
            if os.path.exists(subtitle_path):
                print(f"Subtitles for {lang} already downloaded at {subtitle_path}")
                continue

            ydl_opts = {
                "writesubtitles": True,
                "subtitleslangs": [lang],
                "subtitlesformat": "json3",
                "skip_download": True,
                "outtmpl": subtitle_path,
                "quiet": True,
            }
            try:
                print(f"Downloading subtitles for {lang}...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={self.youtube_id}"])
                paths.append(subtitle_path)

            except Exception as e:
                print(f"Error downloading subtitles for {lang}: {e}")

        return paths
