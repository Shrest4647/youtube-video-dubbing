# Copyright 2024 Jordi Mas i Hernàndez <jmas@softcatala.org>
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

from time import sleep
from google import genai
from google.genai import types, errors
from logging import Logger

_BREAK_MARKER = "<break>"
STEPS = 20


def logger() -> Logger:
    return Logger("translation_gemini")


class TranslationGemini:
    def __init__(self, api_key=""):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def load_model(self):
        pass

    def _translate_text(
        self, source_language: str, target_language: str, text: str, model: str = None
    ) -> str:
        prompt = f'Strictly, respectfully, and concisely, Dub the following script from {source_language} to {target_language}. Keep all the markers "{_BREAK_MARKER}" intact in their proper place for sync purposes. Return only the translated script. \nScript: "{text}"'
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0.5,
        )

        try:
            response_chunks = self.client.models.generate_content_stream(
                model=model or self.model,
                contents=contents,
                config=generate_content_config,
            )
            translated_text = "".join(chunk.text for chunk in response_chunks)
            logger().info(
                f"translation_gemini: Translated text from {source_language} to {target_language} using Gemini."
            )
            translated_text = translated_text.strip()
            # check if the number of _BREAK_MARKER in the translated_text is the same as in the original text
            if translated_text.count(_BREAK_MARKER) != text.count(_BREAK_MARKER):
                logger().warning(
                    "translation_gemini: The number of break markers in the translated text does not match the original text. Retrying translation."
                )
                sleep(5)
                return self._translate_text(
                    source_language,
                    target_language,
                    text,
                    model="gemini-2.5-flash"
                    if model == "gemini-2.5-flash-lite"
                    else ("gemini-2.5-flash-lite" if model is None else None),
                )
            return translated_text
        except errors.ClientError as e:
            logger().error(
                f"translation_gemini: Client error occurred: {e.message} (code: {e.code}). Retrying after 60 seconds."
            )
            sleep(60)
            return self._translate_text(
                source_language,
                target_language,
                text,
            )
        except Exception as e:
            logger().error(f"translation_gemini: An error occurred: {e}")
            raise

    def get_language_pairs(self):
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
        results = set()
        for source in languages:
            for target in languages:
                if source != target:
                    results.add((source, target))
        logger().info(
            "translation_gemini: The Gemini API supports many language pairs. "
            f"Returning {len(results)} language pairs."
        )
        return results

    def _translate_script(
        self,
        *,
        script: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translates the provided transcript to the target language."""

        # find the nth occurance of the _BREAK_MARKER and split there
        parts = script.split(_BREAK_MARKER)
        script_splits = []

        for i in range(0, len(parts), STEPS):
            logger().warning(
                f"translation_gemini: Translating script part {i//STEPS + 1} / {(len(parts) + STEPS - 1)//STEPS}"
            )
            part = _BREAK_MARKER.join(parts[i : i + STEPS])

            translated_script = self._translate_text(
                source_language=source_language,
                target_language=target_language,
                text=part,
            )
            script_splits.append(translated_script)
        return _BREAK_MARKER.join(script_splits)
