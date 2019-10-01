from utilities.vqa.dataset import *


def build_dataset(name, directory, tokenizer, tr_split=0.8, limit=None, q_len_range=None, a_len_range=None, seed=555):
    """
    Builds a dataset for training and testing a model.
    Data is saved as lists using pickle, hence you'll have to convert it to tensors at the right time
    Note: This code is optimized to be fast, not fancy.
    :param name: The name of the dataset
    :param directory: The directory in which to save the dataset, relative to resources/data/
    :param tokenizer: The tokenizer to use (BERT, GPT etc)
    :param tr_split: Amount of samples for the training set
    :param limit: Size of the training + validation sets combined
    :param a_len_range: the range in which the question length should be. Inclusive left, exclusive right [)
    :param q_len_range: the range in which the answer length should be. Inclusive left, exclusive right [)
    :param seed: random seed to replicate results
    :return: The built dataset (training + testing) and the image path directory
    """

    # Set the seed
    random.seed(seed)

    # Load in RAM the samples using the VQA helper tools and our path parser.
    q_path, a_path, i_path = get_data_paths()
    vqa_helper = VQA(a_path, q_path)

    # Get all the question/answer objects
    qa_objects = vqa_helper.loadQA(vqa_helper.getQuesIds())

    # Candidate objects prior to filtering
    candidates = {}  # Dictionary : <key> = answer length, <value> = id, question, answer, image
    selected = []

    print('Creating candidates..')
    for qa_object in tqdm(qa_objects):
        # Parse object
        obj_id, obj_q, obj_as, obj_i = get_qai(qa_object, vqa_helper)

        # Check RGB validity and skip if need to
        if not check_rgb(i_path, obj_i):
            continue

        # Embed the question
        q_embed = [tokenizer.cls_token_id] + embed(tokenizer, obj_q) + [tokenizer.sep_token_id]
        # Compute the length of the question
        q_embed_len = len(q_embed)

        # Performance booster. This really helps by avoiding multiple identical operations
        prev_answer = None
        prev_answer_emb = None

        # Every question has 10 answers
        for obj_a in obj_as:

            # Try to skip embedding if possible and use cached version
            if obj_a == prev_answer:
                a_embed = prev_answer_emb
            else:
                # Embed the question and answer
                a_embed = embed(tokenizer, obj_a) + [tokenizer.sep_token_id]
                prev_answer = obj_a
                prev_answer_emb = a_embed

            # Compute the lengths of the answer
            a_embed_len = len(a_embed)

            # Filter out depending on question / answer lengths
            if q_len_range is not None and q_embed_len not in q_len_range:
                continue
            if a_len_range is not None and q_embed_len not in a_len_range:
                continue

            # Generate candidates
            if a_embed_len not in candidates:
                candidates[a_embed_len] = [[obj_id, q_embed, a_embed, obj_i]]
            else:
                candidates[a_embed_len].append([obj_id, q_embed, a_embed, obj_i])

    print('Shuffling candidates..')
    # Shuffle all the arrays in the dictionary. This is very important to balance the dataset
    for a_embed_len, _ in tqdm(candidates.items()):
        random.shuffle(candidates[a_embed_len])

    print('Selecting candidates..', 'Limit set to {}'.format(limit) if limit is not None else 'Limit deactivated')

    # Keep first the questions whose answer is longest.
    ordered_lengths = sorted(list(candidates.keys()), reverse=True)

    # Reduce progressively to limit
    for q_embed_len in ordered_lengths:
        # Add all samples in this length range. Then check limit.
        selected += candidates[q_embed_len]
        if limit is not None:
            # Check limit reach
            if len(selected) > limit:
                # Truncate if need to
                selected = selected[:limit]
                # Break operation
                break

    longest_sequence = 0

    # Generate token type ids. 0 = Question, 1 = Answer
    print('Generating token type ids..')
    for sample in tqdm(selected):
        l_q = len(sample[1])
        l_a = len(sample[2])
        sample.append([0] * l_q + [1] * l_a)  # sample[4] type_ids
        sample.append([1] * (l_q + l_a))  # sample[5] att_mask

        # Concatenate question & answer for later padding
        sample.insert(1, sample[1] + sample[2])

        # Delete question and answer single entities
        del sample[2:4]

        # Update longest sequence
        if l_q + l_a > longest_sequence:
            longest_sequence = l_q + l_a
    # Each sample now is: ID, SEQUENCE, IMAGE_ID, TOKEN_TYPES, ATT_MASK
    # Switch to numpy to exploit better indexing
    selected = np.array(selected)

    # Pad sequences
    print('\nPadding sequences..')
    padded_seqs = pad_sequences(selected[:, 1], maxlen=longest_sequence, padding='post',
                                value=int(tokenizer.pad_token_id))
    for sample, padded_seq in zip(selected, padded_seqs):
        sample[1] = padded_seq

    # Pad sequences
    print('\nPadding token types..')
    token_types = pad_sequences(selected[:, 3], maxlen=longest_sequence, padding='post',
                                value=int(1))
    for sample, types in zip(selected, token_types):
        sample[3] = types

    # Pad sequences
    print('\nPadding attention mask..')
    att_mask = pad_sequences(selected[:, 4], maxlen=longest_sequence, padding='post',
                             value=int(0))
    for sample, mask in zip(selected, att_mask):
        sample[4] = mask

    tr_data, ts_data = dump_selected(selected, directory, name, tr_split)

    return tr_data, ts_data, i_path


class BertDataset(Dataset):
    """
    This is a dataset specifically crafted for BERT models
    """
    def __init__(self, directory, name, maxlen=None):
        try:
            with open(os.path.join(directory, name), 'rb') as fd:
                self.data = pickle.load(fd)
            # Get image path
            _, _, self.i_path = get_data_paths()
            self.maxlen = maxlen if maxlen is not None else len(self.data)
            print('Data loaded successfully.')
        except (OSError, IOError) as e:
            print('Unable to load data. Did you build it first?', str(e))

    def get_image(self, image_id):
        return load_image(self.i_path, image_id)

    def __getitem__(self, item):
        sample = self.data[item]

        identifier = sample[0]
        sequence = torch.tensor(sample[1]).long()
        image = transform_image(self.get_image(sample[2]))
        token_types = torch.tensor(sample[3]).long()
        att_mask = torch.tensor(sample[4]).long()

        return identifier, sequence, image, token_types, att_mask

    def __len__(self):
        return self.maxlen


if __name__ == '__main__':

    build_dataset(directory=resources_path('models', 'bert', 'data'),
                  name='bert_1M.pk',
                  tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                  tr_split=0.8,
                  limit=1000000)

    tr_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='tr_bert_1M.pk')
    ts_dataset = BertDataset(directory=resources_path('models', 'bert', 'data'), name='ts_bert_1M.pk')