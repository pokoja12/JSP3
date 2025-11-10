from urllib.request import urlopen
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
import yaml
import statistics


def load_text_from_web(url):
    response = urlopen(url)
    text = response.read().decode('utf-8')
    text = text.replace('\r\n', '\n')
    return text


def trim_text_edges(text, head, tail):
    """function removes a given number of lines from the start (head) and end (tail) of the text"""
    rows = text.splitlines()
    remaining = rows[head:len(rows) - tail] # [start:end] it gives smaller required list
    return "\n".join(remaining)


def vader_sentiment_per_sentence(text):
    """
    Splits text into sentences and returns VADER compound sentiment value for each sentence.
    
    Note: VADER calculates sentiment for the entire sentence, considering word sentiment,
    intensifiers, negations, and context—not just an average of individual word scores.
    """
    analyzer = SentimentIntensityAnalyzer() # analyzátor sentimentu
    sentences = sent_tokenize(text) 
    compound_values = [analyzer.polarity_scores(sentence)['compound'] 
                       for sentence in sentences]
    return compound_values


def load_yaml_file(yaml_path):
    """Načte YAML soubor a vrátí dictionary."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def new_wornet_sentiment_per_sentence(text):
    """
    Tokenizes text into sentences and words, finds all synsets of each word in a sentiment wordnet,
    gets their sentiment values from a new sentiment wordnet, and averages per word and per sentence.
    Returns a list of average sentiment values (compound_values) for each sentence.
    """
    senti_wordnet_dict = load_yaml_file("/Users/macbook/Desktop/programovani_uloziste/JSP3/sentiment_wordnet.yml")

    sentences = sent_tokenize(text)
    compound_values = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        word_values = []

        for word in words:
            if not word.isalpha():
                continue
            synsets = wn.synsets(word.lower())
            if not synsets:
                continue

            # Average sentiment of all synsets for this word
            synset_values = []
            for synset in synsets:
                synset_id = f"{synset.offset():08d}-{synset.pos()}"

                for dict_synset in senti_wordnet_dict:
                    if dict_synset.endswith(synset_id):  # only add synsets that exist in the new wordnet
                        synset_values.append(senti_wordnet_dict[dict_synset])
                        break  # stop searching after the first match

            if synset_values:
                word_values.append(statistics.mean(synset_values))

        if word_values:
            compound_values.append(statistics.mean(word_values))

    return compound_values


def main():
    text = load_text_from_web("https://www.gutenberg.org/files/64317/64317-0.txt")
    text = trim_text_edges(text,36,1)
    #print(text[10:])
    

# alternative_text simulates book, for quick results

    alternative_text = """
This interpretation of "A Day in the Life" by The Beatles is divided into chapters
based on the singer and musical section, highlighting the song’s narrative structure.
Each chapter offers a different perspective – the first and third chapters are sung
by John Lennon, providing reflective, sometimes surreal observations of everyday
events. The second chapter is Paul McCartney’s section, livelier and more humorous,
depicting the routines of daily life. The orchestral climax connects all parts,
closing the “day in the life” with a dramatic musical crescendo.

1)
I read the news today, oh boy
About a lucky man who made the grade.
And though the news was rather sad
Well, I just had to laugh.
I saw the photograph.
He blew his mind out in a car.
He didn't notice that the lights had changed.
A crowd of people stood and stared.
They'd seen his face before.
Nobody was really sure if he was from the House of Lords.
I saw a film today, oh boy
The English Army had just won the war.
A crowd of people turned away.
But I just had to look
Having read the book.
I'd love to turn you on.

2)
Woke up, fell out of bed
Dragged a comb across my head.
Found my way downstairs and drank a cup
And looking up, I noticed I was late.
Found my coat and grabbed my hat
Made the bus in seconds flat.
Found my way upstairs and had a smoke.
And somebody spoke and I went into a dream.

3)
I read the news today, oh boy
Four thousand holes in Blackburn, Lancashire.
And though the holes were rather small
They had to count them all.
Now they know how many holes it takes to fill the Albert Hall.
I'd love to turn you on
"""

    alternative_text = trim_text_edges(alternative_text,9,0)
    #print(alternative_text)

    sentiments_per_sentence = vader_sentiment_per_sentence(alternative_text)
    new_sentiments_per_sentence = new_wornet_sentiment_per_sentence(alternative_text)
    
    print(sentiments_per_sentence)
    print(new_sentiments_per_sentence)


if __name__ == "__main__":
    main()





