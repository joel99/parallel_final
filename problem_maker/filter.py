
def filter_words(input_file, output_file, lower_bound=5, upper_bound=7):
    with open(input_file, 'r') as file:
        words = file.read().split()

    # Filter words based on length
    filtered_words = [word for word in words if lower_bound <= len(word) <= upper_bound]

    with open(output_file, 'w') as file:
        file.write(f'{len(filtered_words)}\n')
        for word in filtered_words:
            file.write(word + '\n')

for i in range(4, 9):
    input_file_path = 'words_alpha.txt'
    output_file_path = f'words_{i}.txt' 
    filter_words(input_file_path, output_file_path, lower_bound=i, upper_bound=i)
    
# Example usage
# input_file_path = 'words_alpha.txt'  # Change this to your input file path
# output_file_path = 'words_5_7.txt'  # Change this to your desired output file path
# filter_words(input_file_path, output_file_path)