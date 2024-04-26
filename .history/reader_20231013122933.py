import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

nltk.download('punkt')

RAW_DATA_FOLDER = "resident_interviews"

class Interview:
    def __init__(self, transcript):
        self.transcript = transcript
        self.segments = None

    def set_segments(self):
        # Tokenize into sentences
        
        sentences = nltk.sent_tokenize(self.transcript)
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

ints = []

for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
    for filename in filenames:
        filepath = os.path.join(subdir, filename)  # Get the full path to the file
        
        with open(filepath, 'r') as file:
            transcript = file.read()
        
        inter = Interview(transcript)
        inter.set_segments()
        ints.append(inter)

        



i1 = Interview("resident_one.txt")

i1.set_segments()
i2 = Interview("resident_two.txt")


import pdb;pdb.set_trace()