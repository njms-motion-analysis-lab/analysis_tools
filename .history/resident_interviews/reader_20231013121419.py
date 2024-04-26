import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class Interview:

    def set_segments_transcript(self):
        with open("resident_one.txt", "r") as file:
            transcript = file.read()
        # Tokenize into sentences
        
        sentences = nltk.sent_tokenize(transcript)
        # Split into segments based on speaker
        segments = {}
        current_speaker = None
        for sentence in sentences:
            if "Elaine" in sentence or "Female resident 1" in sentence:
                current_speaker = sentence.split('\n')[0]
                segments[current_speaker] = []
            else:
                if current_speaker:
                    segments[current_speaker].append(sentence)

        print(segments)