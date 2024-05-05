import numpy as np
# Filters 100 words to be used for test set
def filter_words(input_file, output_file):
    with open(input_file, 'r') as file:
        words = file.read().split()

    # Filter words based on length
    filtered_words = np.random.choice(words, size = 100, replace = False)

    with open(output_file, 'w') as file:
        file.write(f'{len(filtered_words)}\n')
        for word in filtered_words:
            file.write(word + '\n')

for i in range(4, 9):
    input_file_path = f'words_{i}.txt'
    output_file_path = f'words_test{i}.txt' 
    filter_words(input_file_path, output_file_path)