import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

def word_feats(words):
    return dict([(word, True) for word in words])

# We'll use some general sentences about fungi and mycology.
mycology_sentences = (
    "Fungi play a vital role in the decomposition process",
    "Mycology is the study of fungi",
    "Yeasts are a type of fungus",
    "Some fungi form symbiotic relationships with plants",
)

non_mycology_sentences = (
    "The tree is tall and sturdy",
    "The car is fast",
    "The computer is running quickly",
    "Basketball is a popular sport",
    "Dogs are known for their loyalty"
)

mycology_features = [(word_feats(f.split()), 'mycology') for f in mycology_sentences]
non_mycology_features = [(word_feats(f.split()), 'not mycology') for f in non_mycology_sentences]

train_set = mycology_features + non_mycology_features

classifier = NaiveBayesClassifier.train(train_set)

# Now let's test it out
print(classifier.classify(word_feats("Fungal diseases can affect crops".split())))  # this should say 'mycology'
print(classifier.classify(word_feats("The sun sets in the west".split())))  # this should say 'not mycology'
