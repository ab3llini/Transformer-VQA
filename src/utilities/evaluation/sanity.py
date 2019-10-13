from tqdm import tqdm


def cross_dataset_similarity(*datasets):
    data = []

    for dataset in datasets:
        data.append(dataset.data)
    for row, samples in tqdm(enumerate(zip(*data))):
        target_id = samples[0][0]
        for ds_idx, sample in enumerate(samples[1:]):
            if sample[0] != target_id:
                print('Cross similarity failed:')
                print('Row {}, Dataset #{} question id {} does not match Dataset #0 id {}'.format(row, ds_idx + 1,
                                                                                                  sample[0], target_id))
                return False
    return True
