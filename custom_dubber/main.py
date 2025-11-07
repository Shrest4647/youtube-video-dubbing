import argparse
import os
import shutil
import json
from pathlib import Path
import concurrent.futures
from itertools import cycle

from custom_dubber.utils import (
    extract_transcripts,
    translate_transcripts,
    concat_close_transcripts,
    transcribe_using_ytt,
    yt_download,
    synthesize_speech_worker,
)
from . import audio_processing
from .video_processing import VideoProcessing


def main(
    youtube_id: str,
    source_language: str,
    target_language: str,
    output_directory: str,
    voice: str,
    api_keys: list[str],
    name: str | None = None,
    move_directory: str | None = None,
    cleanup: bool = False,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    # copy the ignore file to output directory

    shutil.copyfile(
        os.path.join(Path(__file__).parent.parent, ".gitignore.template"),
        os.path.join(output_directory, ".gitignore"),
    )

    download_metadata_file = os.path.join(output_directory, "download_metadata.json")
    download_metadata = {}

    if os.path.exists(download_metadata_file):
        with open(download_metadata_file, "r", encoding="utf-8") as f:
            download_metadata = json.load(f)

    video_path = download_metadata.get("video_path", "")
    audio_path = download_metadata.get("audio_path", "")
    subtitle_paths = download_metadata.get("subtitle_paths", [])

    if not video_path or not audio_path:
        print("Video or audio file not found, downloading...")
        video_path, audio_path, subtitle_paths = yt_download(
            youtube_id, output_directory, source_language, target_language
        )

        with open(download_metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "subtitle_paths": subtitle_paths,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    transcripts = download_metadata.get("transcripts", [])
    if not transcripts:
        # Fetch the transcript
        transcripts = []
        if len(subtitle_paths) == 0:
            transcripts = transcribe_using_ytt(
                youtube_id, source_language, target_language, api_keys
            )
        elif len(subtitle_paths) == 1:
            source_subtitle_path = subtitle_paths[0]
            transcripts = extract_transcripts(source_subtitle_path)
            transcripts = translate_transcripts(
                transcripts, source_language, target_language, api_keys
            )
        elif len(subtitle_paths) == 2:
            # Assume first is source, second is target
            target_subtitle_path = subtitle_paths[1]
            transcripts = extract_transcripts(target_subtitle_path)
            transcripts = [
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "translated_text": item["text"],
                }
                for item in transcripts
            ]
        else:
            raise Exception(
                "Multiple subtitle paths found, unable to determine source and target."
            )

        print(f"Total transcript segments: {len(transcripts)}")
        transcripts = concat_close_transcripts(transcripts, threshold=3.5)
        print(f"Total transcript segments after concatenation: {len(transcripts)}")
        with open(download_metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "subtitle_paths": subtitle_paths,
                    "transcripts": transcripts,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    if not download_metadata.get("tts_complete", False):
        print("Starting text-to-speech synthesis...")
        # Convert text to speech in parallel
        updated_transcripts = []
        api_key_cycle = cycle(api_keys)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1  # len(api_keys)
        ) as executor:
            futures = [
                executor.submit(
                    synthesize_speech_worker,
                    item,
                    i,
                    len(transcripts),
                    output_directory,
                    voice,
                    target_language,
                    next(api_key_cycle),
                )
                for i, item in enumerate(transcripts)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    updated_transcripts.append(future.result())
                except Exception as e:
                    print(f"Error during TTS synthesis: {e}")

        # Sort transcripts back to original order based on start time
        transcripts = sorted(updated_transcripts, key=lambda x: x["start"])
        print("All segments processed.")

        with open(download_metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "subtitle_paths": subtitle_paths,
                    "transcripts": transcripts,
                    "tts_complete": True,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    print("Inserting dubbed vocals into background audio...")

    dubbed_audio_vocals_file = audio_processing.insert_audio_at_timestamps(
        utterance_metadata=transcripts,
        background_audio_file=audio_path,
        output_directory=output_directory,
    )
    print(f"Dubbed vocals audio saved to {dubbed_audio_vocals_file}")
    dubbed_audio_file = audio_processing.merge_background_and_vocals(
        background_audio_file=audio_path,
        dubbed_vocals_audio_file=dubbed_audio_vocals_file,
        output_directory=output_directory,
        target_language=target_language,
        vocals_volume_adjustment=5.0,
        background_volume_adjustment=0.0,
    )

    print(f"Final dubbed audio saved to {dubbed_audio_file}")
    dubbed_video_file = VideoProcessing.combine_audio_video(
        video_file=video_path,
        dubbed_audio_file=dubbed_audio_file,
        output_directory=output_directory,
        target_language=target_language,
    )
    print(f"Dubbed video saved to {dubbed_video_file}")

    if name:
        final_video_path = os.path.join(output_directory, Path(name).stem + ".mp4")

        os.rename(dubbed_video_file, final_video_path)
        dubbed_video_file = final_video_path
        print(f"Renamed final video to {dubbed_video_file}")

    if move_directory:
        os.makedirs(move_directory, exist_ok=True)
        final_move_path = os.path.join(move_directory, Path(dubbed_video_file).name)
        shutil.move(dubbed_video_file, final_move_path)
        dubbed_video_file = final_move_path
        print(f"Moved final video to {dubbed_video_file}")

    if cleanup:
        print("Cleaning up intermediate files...")
        for item in transcripts:
            dubbed_path = item.get("dubbed_path")
            if dubbed_path and os.path.exists(dubbed_path):
                os.remove(dubbed_path)
        if os.path.exists(dubbed_audio_vocals_file):
            os.remove(dubbed_audio_vocals_file)
        print("Cleanup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Dubber")
    parser.add_argument(
        "--youtube_id", type=str, required=True, help="YouTube video ID"
    )
    parser.add_argument(
        "--source_language", type=str, default="en-US", help="Source language code"
    )
    parser.add_argument(
        "--target_language", type=str, required=True, help="Target language code"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="output/",
        help="Directory to save output files",
    )
    parser.add_argument("--voice", type=str, default="Charon", help="Voice for dubbing")
    parser.add_argument(
        "--api_keys",
        # Nargs='+' means we expect one or more values, which will be collected into a list.
        nargs="+",
        required=True,  # Ensure the user provides this argument
        help="A list of API keys (separated by spaces).",
    )
    parser.add_argument("--name", default=None, help="Final Video name")
    parser.add_argument(
        "--move_directory", default=None, help="Move output to directory"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Cleanup intermediate files",
    )
    args = parser.parse_args()

    main(
        youtube_id=args.youtube_id,
        source_language=args.source_language,
        target_language=args.target_language,
        output_directory=args.output_directory,
        voice=args.voice,
        api_keys=args.api_keys,
        name=args.name,
        move_directory=args.move_directory,
        cleanup=args.cleanup,
    )
