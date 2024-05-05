import random
random.seed(0)
input_file = 'allowed_words.txt'
subset_length = 100
with open(input_file, 'r') as file:
    word_count = int(file.readline())
    words = [file.readline().strip() for _ in range(word_count)]

# take random subset
random_subset = random.sample(words, subset_length)
output_file = f'subset_{subset_length}.txt'
with open(output_file, 'w') as file:
    file.write(f'{subset_length}\n')
    for word in random_subset:
        file.write(word + '\n')