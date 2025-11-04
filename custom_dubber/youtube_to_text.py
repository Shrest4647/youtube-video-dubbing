import re
from youtube_transcript_api import YouTubeTranscriptApi


class YoutubeToText:
    def __init__(self, youtube_id):
        self.youtube_id = youtube_id
        self.api = YouTubeTranscriptApi()
        self.transcript = None

    def get_transcript(self, language_codes):
        def filter_bracket_text(text):
            # Remove text within square brackets and parentheses
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"\(.*?\)", "", text)
            return text.strip()

        transcript_list = self.api.fetch(self.youtube_id, languages=language_codes)
        self.transcript = [
            {"start": item.start, "end": item.start + item.duration, "text": item.text}
            for item in transcript_list
            if filter_bracket_text(item.text) != ""
        ]
        return self.transcript
