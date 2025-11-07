# Copyright 2025 Jordi Mas i Hernàndez <jmas@softcatala.org>
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

import io
from time import sleep
from typing import List, NamedTuple
import mimetypes
import struct
import hashlib
import json
from pathlib import Path
import shutil

from google import genai
from google.genai import types, errors
from pydub import AudioSegment
from logging import Logger
from .ffmpeg import FFmpeg


def logger() -> Logger:
    return Logger("text_to_speech_gemini")


class Voice(NamedTuple):
    name: str
    gender: str
    region: str = ""


def _write_wav_to_mp3(wav_data: bytes, filename: str):
    """Converts WAV audio data to MP3 format.

    Args:
        wav_data: The raw WAV audio data as a bytes object.

    Returns:
        None
    """

    audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")
    audio_segment.export(filename, format="mp3")


def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = _parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",  # ChunkID
        chunk_size,  # ChunkSize (total file size - 8 bytes)
        b"WAVE",  # Format
        b"fmt ",  # Subchunk1ID
        16,  # Subchunk1Size (16 for PCM)
        1,  # AudioFormat (1 for PCM)
        num_channels,  # NumChannels
        sample_rate,  # SampleRate
        byte_rate,  # ByteRate
        block_align,  # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",  # Subchunk2ID
        data_size,  # Subchunk2Size (size of audio data)
    )
    return header + audio_data


def _parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts:  # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass  # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass  # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


# Documentation: https://ai.google.dev/docs/gemini_api_overview
class TextToSpeechGemini:
    def __init__(self, device="cpu", server="", api_key="", cache_dir=".tts_cache"):
        self._SSML_MALE: str = "Male"
        self._SSML_FEMALE: str = "Female"
        self._DEFAULT_SPEED: float = 1.0
        self.client = genai.Client(
            api_key=api_key,
        )
        self.api_key = api_key
        self.model = "gemini-2.5-flash-preview-tts"
        self.backup_model = "gemini-2.5-pro-preview-tts"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def set_api_key(self, api_key: str):
        self.client = genai.Client(
            api_key=api_key,
        )
        self.api_key = api_key

    def get_available_voices(self, language_code: str) -> List[Voice]:
        # Genders are inferred from the names. This might not be 100% accurate.
        voice_data = {
            "Zephyr": self._SSML_FEMALE,
            "Puck": self._SSML_MALE,
            "Charon": self._SSML_MALE,
            "Kore": self._SSML_FEMALE,
            "Fenrir": self._SSML_MALE,
            "Leda": self._SSML_FEMALE,
            "Orus": self._SSML_MALE,
            "Aoede": self._SSML_FEMALE,
            "Callirrhoe": self._SSML_FEMALE,
            "Autonoe": self._SSML_FEMALE,
            "Enceladus": self._SSML_MALE,
            "Iapetus": self._SSML_MALE,
            "Umbriel": self._SSML_MALE,
            "Algieba": self._SSML_MALE,
            "Despina": self._SSML_FEMALE,
            "Erinome": self._SSML_FEMALE,
            "Algenib": self._SSML_MALE,
            "Rasalgethi": self._SSML_MALE,
            "Laomedeia": self._SSML_FEMALE,
            "Achernar": self._SSML_MALE,
            "Alnilam": self._SSML_MALE,
            "Schedar": self._SSML_MALE,
            "Gacrux": self._SSML_MALE,
            "Pulcherrima": self._SSML_FEMALE,
            "Achird": self._SSML_MALE,
            "Zubenelgenubi": self._SSML_MALE,
            "Vindemiatrix": self._SSML_FEMALE,
            "Sadachbia": self._SSML_FEMALE,
            "Sadaltager": self._SSML_MALE,
            "Sulafat": self._SSML_FEMALE,
        }

        voices = [
            Voice(name=name, gender=gender, region="")
            for name, gender in voice_data.items()
        ]

        logger().info(
            f"text_to_speech_gemini.get_available_voices: {voices} for language {language_code}"
        )

        return voices

    def _does_voice_supports_speeds(self):
        # Gemini API does not seem to support speed control in the provided example
        return False

    def _convert_text_to_speech_without_end_silence(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:
        """TTS add silence at the end that we want to remove to prevent increasing the speech of next
        segments if is not necessary."""

        dubbed_file = self._convert_text_to_speech(
            assigned_voice=assigned_voice,
            target_language=target_language,
            output_filename=output_filename,
            text=text,
            speed=speed,
        )

        dubbed_audio = AudioSegment.from_file(dubbed_file)
        pre_duration = len(dubbed_audio)

        FFmpeg().remove_silence(filename=dubbed_file)
        dubbed_audio = AudioSegment.from_file(dubbed_file)
        post_duration = len(dubbed_audio)
        if pre_duration != post_duration:
            logger().debug(
                f"text_to_speech._convert_text_to_speech_without_end_silence. File {dubbed_file} shorten from {pre_duration} to {post_duration}"
            )

        return dubbed_file

    def _get_cache_key(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        text: str,
        speed: float,
        backup: bool = False,
    ) -> str:
        """Generate a unique cache key based on input parameters."""
        cache_data = {
            "assigned_voice": assigned_voice,
            "target_language": target_language,
            "text": text,
            "speed": speed,
            "backup": backup,
            "model": self.backup_model if backup else self.model,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _get_cached_audio(self, cache_key: str) -> str | None:
        """Retrieve cached audio file if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if cache_file.exists():
            logger().debug(f"Cache hit for key: {cache_key}")
            return str(cache_file)
        return None

    def _save_to_cache(self, cache_key: str, audio_file: str) -> None:
        """Save audio file to cache."""
        cache_file = self.cache_dir / f"{cache_key}.wav"
        try:
            shutil.copy2(audio_file, cache_file)
            logger().debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger().warning(f"Failed to save to cache: {e}")

    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
        backup: bool = False,
    ) -> str:
        logger().debug(
            f"text_to_speech_gemini._convert_text_to_speech: assigned_voice: {assigned_voice}, output_filename: '{output_filename}'"
        )

        # Check cache first
        # cache_key = self._get_cache_key(
        #     assigned_voice=assigned_voice,
        #     target_language=target_language,
        #     text=text,
        #     speed=speed,
        #     backup=backup,
        # )

        # cached_file = self._get_cached_audio(cache_key)
        # if cached_file:
        #     # Copy cached file to output location
        #     try:
        #         shutil.copy2(cached_file, output_filename)
        #         logger().info(f"Using cached audio for: {output_filename}")
        #         return output_filename
        #     except Exception as e:
        #         logger().warning(
        #             f"Failed to copy cached file: {e}, generating new audio"
        #         )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"""
                        <style-instruction>
                        The following is a dub of a documentary in {target_language} language. 
                        Take pauses and intonate accordingly. 
                        Read aloud in a calm, soothing, enthusiastic tone like David Attenborough at a good pace:
                        </style-instruction>
                        {text}
                    """
                    ),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=[
                "audio",
            ],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=assigned_voice
                    )
                )
            ),
        )

        full_audio_data = bytearray()
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model if not backup else self.backup_model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                if (
                    chunk.candidates[0].content.parts[0].inline_data
                    and chunk.candidates[0].content.parts[0].inline_data.data
                ):
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)

                    if file_extension is None or file_extension != ".wav":
                        data_buffer = _convert_to_wav(
                            inline_data.data, inline_data.mime_type
                        )
                    full_audio_data.extend(data_buffer)
            try:
                with open(output_filename, "wb") as f:
                    f.write(bytes(full_audio_data))

                # Save to cache
                # self._save_to_cache(cache_key, output_filename)

                return output_filename

            except Exception as e:
                logger().error(
                    f"text_to_speech_gemini: Error writing audio file {output_filename}: {e}"
                )
                sleep(60)
                return self._convert_text_to_speech(
                    assigned_voice=assigned_voice,
                    target_language=target_language,
                    output_filename=output_filename,
                    text=text,
                    speed=speed,
                    backup=False,
                )

        except errors.ClientError as e:
            logger().error(
                f"text_to_speech_gemini: Client error occurred: {e.message} (code: {e.code} - key {self.api_key}). Retrying after 60 seconds."
            )
            sleep(60)
            return self._convert_text_to_speech(
                assigned_voice=assigned_voice,
                target_language=target_language,
                output_filename=output_filename,
                text=text,
                speed=speed,
                backup=False,
            )
        except Exception as e:
            logger().error(f"text_to_speech_gemini: An error occurred: {e}")
            raise

    def get_languages(self):
        languages = [
            "afr",
            "ara",
            "hye",
            "aze",
            "bel",
            "bos",
            "bul",
            "cat",
            "zho",
            "hrv",
            "ces",
            "dan",
            "nld",
            "eng",
            "est",
            "fin",
            "fra",
            "glg",
            "deu",
            "ell",
            "heb",
            "hin",
            "hun",
            "isl",
            "ind",
            "ita",
            "jpn",
            "kan",
            "kaz",
            "kor",
            "lav",
            "lit",
            "mkd",
            "msa",
            "mar",
            "mri",
            "nep",
            "nor",
            "fas",
            "pol",
            "por",
            "ron",
            "rus",
            "srp",
            "slk",
            "slv",
            "spa",
            "swa",
            "swe",
            "tgl",
            "tam",
            "tha",
            "tur",
            "ukr",
            "urd",
            "vie",
            "cym",
        ]

        languages = sorted(list(languages))

        logger().debug(f"text_to_speech_gemini.get_languages: {languages}")
        return languages
