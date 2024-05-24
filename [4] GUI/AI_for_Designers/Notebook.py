def check_product_name(Activity: str) -> None:
    """Checks whether the name of the Activity is valid (only contains alphanumerical characters, '_' and '-'.
    The result is printed

    Args:
        Activity (str): name of the activity
    """

    for letter in Activity:
        if not letter.isalpha() and not letter.isalnum() and letter != '_' and letter != '-':
            print('Name is not valid!')
            return

    print('Name is valid!')


def amount_of_samples(data_file: str) -> int:
    """Function to find the amount of samples from the preprocessed data

    Args:
        data_file (str): relative path seen from the notebook that contains the different processed data files

    Returns:
        int: amount of samples that can be used for training the model
    """

    max_id = 0
    with open(data_file) as f:
        for line in f:
            if line != '':
                max_id = int(line.strip().split(',')[2])

    return max_id
