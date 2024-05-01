def filter_words(input_file, output_file):
    with open(input_file, 'r') as file:
        words = file.read().split()

    # Filter words based on length
    filtered_words = [word for word in words if 5 <= len(word) <= 7]

    with open(output_file, 'w') as file:
        for word in filtered_words:
            file.write(word + '\n')

# Example usage
input_file_path = 'words_alpha.txt'  # Change this to your input file path
output_file_path = 'words_5_7.txt'  # Change this to your desired output file path
filter_words(input_file_path, output_file_path)