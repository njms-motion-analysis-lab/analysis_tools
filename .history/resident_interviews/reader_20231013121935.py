import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class Interview:
    def __init__(self, transcript_path):
        self.path = transcript_path
        self.segments = None

    def set_segments(self):
        with open(self.path) as file:
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

        

        self.segments = segments
        print(self.segments)
        print("done!")


i1 = Interview("resident_one.txt")

i1.set_segments()
i2 = Interview("resident_two.txt")


