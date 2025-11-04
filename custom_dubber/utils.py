import json5
import random
import re
import os
from .translation_gemini import TranslationGemini
from .video_downloader import VideoDownloader
from .youtube_to_text import YoutubeToText
from .text_to_speech_gemini import TextToSpeechGemini

_BREAK_MARKER = "<break>"


def extract_transcripts(subtitle_path):
    def filter_bracket_text(text):
        # Remove text within square brackets and parentheses
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        return text.strip()

    transcripts = []
    with open(subtitle_path, "r", encoding="utf-8") as f:
        subtitle_data = json5.load(f)
        for item in subtitle_data["events"]:
            if "segs" in item:
                text = "".join([seg["utf8"] for seg in item["segs"]])

                if filter_bracket_text(text) == "":
                    continue

                transcripts.append(
                    {
                        "start": item["tStartMs"] / 1000.0,
                        "end": (item["tStartMs"] + item["dDurationMs"]) / 1000.0,
                        "text": text,
                    }
                )
    return transcripts


def translate_transcripts(transcripts, source_language, target_language, api_keys):
    results = []

    all_texts = _BREAK_MARKER.join([item["text"] + "\n" for item in transcripts])
    translator = TranslationGemini(api_key=random.choice(api_keys))
    translated_text = translator._translate_script(
        script=all_texts,
        source_language=source_language,
        target_language=target_language,
    )
    print("Translated text:")
    translated_texts = translated_text.split(_BREAK_MARKER)
    assert len(translated_texts) == len(transcripts)
    print("Building transcripts with translated text...", translated_texts)
    for i in range(len(transcripts)):
        results.append(
            {
                "start": transcripts[i]["start"],
                "end": transcripts[i]["end"],
                "text": transcripts[i]["text"],
                "translated_text": translated_texts[i].strip(),
            }
        )
    return results


def concat_close_transcripts(transcripts, threshold=2.0):
    if not transcripts:
        return []

    concatenated = [transcripts[0]]
    max_duration = 2 * 60  # 2 minutes

    for current in transcripts[1:]:
        previous = concatenated[-1]
        diff = current["start"] - previous["end"]
        threshold_factor = 1
        if len(previous["text"].split()) < 3 or len(current["text"].split()) < 3:
            previous["start"] = previous["start"] + threshold
            threshold_factor = 2

        if (
            diff <= threshold * threshold_factor
            and (current["end"] - previous["start"]) <= max_duration
        ):
            diff_seconds = max(0, diff // 1)  # Round down to nearest second
            # Concatenate texts
            previous["end"] = current["end"]
            previous["text"] += (
                f"\n (pause for {diff_seconds} seconds). \n" + current["text"]
            )
            previous["translated_text"] += (
                f"\n (pause for {diff_seconds} seconds). \n"
                + current["translated_text"]
            )
        else:
            concatenated.append(current)

    return concatenated


def yt_download(
    youtube_id: str,
    output_directory: str,
    source_language: str,
    target_language: str,
) -> tuple[str, str, list[str]]:
    downloader = VideoDownloader(youtube_id=youtube_id, output_dir=output_directory)
    video_path = downloader.download_video()
    print(f"Video downloaded to {video_path}")
    audio_path = downloader.download_audio()
    print(f"Audio downloaded to {audio_path}")

    subtitle_paths = []
    source_file = os.path.join(output_directory, "subtitles", f"{source_language}.json")
    target_file = os.path.join(output_directory, "subtitles", f"{target_language}.json")
    # check if the subtitles already exist
    if os.path.exists(source_file):
        subtitle_paths.append(source_file)
    if os.path.exists(target_file):
        subtitle_paths.append(target_file)
    if not subtitle_paths:
        subtitle_paths = downloader.download_subtitles(
            language_codes=[source_language, target_language]
        )
        print(f"Subtitles downloaded to {subtitle_paths}")
    return video_path, audio_path, subtitle_paths


def transcribe_using_ytt(youtube_id, source_language, target_language, api_keys):
    transcripts = []

    try:
        ytt = YoutubeToText(youtube_id=youtube_id)
        # First check if the transcript is available in the target language
        _transcript = ytt.get_transcript(language_codes=[target_language])
        transcripts = [
            {
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
                "translated_text": item["text"],
            }
            for item in _transcript
        ]
    except Exception as e:
        print(f"Error fetching transcript in {target_language}: {e}")
        # Fallback to source language
        _transcript = ytt.get_transcript(language_codes=[source_language])
        transcripts = translate_transcripts(
            _transcript, source_language, target_language, api_keys
        )
        print("Transcript fetched:")
    return transcripts


def synthesize_speech_worker(
    item, i, total, output_directory, voice, target_language, api_key
):
    """Worker function to synthesize speech for a single transcript item."""
    tts = TextToSpeechGemini(api_key=api_key)
    print(f"Synthesizing speech for segment {i+1}/{total}")
    output_path = (
        f"{output_directory}/segment_{i+1}_{item['start']//1}_{item['end']//1}.wav"
    )
    if os.path.exists(output_path):
        print(f"Audio already exists for segment {i+1}, skipping synthesis.")
        item["dubbed_path"] = output_path
        item["for_dubbing"] = True
        return item

    _audio_data = tts._convert_text_to_speech_without_end_silence(
        assigned_voice=voice,
        target_language=target_language,
        output_filename=output_path,
        text=item["translated_text"] or item["text"],
        speed=1.0,
    )
    item["dubbed_path"] = output_path
    item["for_dubbing"] = True
    print(f"Saved synthesized speech to {output_path} {_audio_data}")
    return item
