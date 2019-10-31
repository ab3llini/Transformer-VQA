from nltk import pos_tag
from nltk.corpus import wordnet as wn


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(tkn_sentence_1, tkn_sentence_2):
    """ compute the sentence similarity using Wordnet """

    tkn_sentence_1 = pos_tag(tkn_sentence_1)
    tkn_sentence_2 = pos_tag(tkn_sentence_2)

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in tkn_sentence_1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in tkn_sentence_2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        sim_words = [synset.path_similarity(ss) for ss in synsets2]
        if len(sim_words) > 0:
            best_score = max(sim_words)
            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1

    # Average the values
    score /= count if count > 0 else 1
    return score


if __name__ == '__main__':

    sentences = [
        "Dogs are awesome.",
        "Some gorgeous creatures are felines.",
        "Dolphins are swimming mammals.",
        "Cats are beautiful animals.",
    ]

    focus_sentence = "Cats are beautiful animals."

    for sentence in sentences:
        print
        "Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence))
        print
        "Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence))
        print

    # Similarity("Cats are beautiful animals.", "Dogs are awesome.") = 0.511111111111
    # Similarity("Dogs are awesome.", "Cats are beautiful animals.") = 0.666666666667

    # Similarity("Cats are beautiful animals.", "Some gorgeous creatures are felines.") = 0.833333333333
    # Similarity("Some gorgeous creatures are felines.", "Cats are beautiful animals.") = 0.833333333333

    # Similarity("Cats are beautiful animals.", "Dolphins are swimming mammals.") = 0.483333333333
    # Similarity("Dolphins are swimming mammals.", "Cats are beautiful animals.") = 0.4

    # Similarity("Cats are beautiful animals.", "Cats are beautiful animals.") = 1.0
    # Similarity("Cats are beautiful animals.", "Cats are beautiful animals.") = 1.0
