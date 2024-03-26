import boto3
import re

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Configure AWS credentials and EMR cluster details
s3_client = boto3.client('s3', region_name='us-east-1')

# Define the input S3 bucket and input corpus path
input_bucket = 'index-corpus'
input_corpus_path = 's3://index-corpus'

# Define the output S3 bucket and output paths
output_bucket = 'jm-class-bucket'
output_inverted_index_path = 's3://jm-class-bucket'
output_word_frequency_path = f'{output_bucket}/word_frequency.txt'
output_word_cloud_path = f'{output_bucket}/word_cloud.jpg'
stopwords_file_key = 'stopwords.txt'

def process_text_file(filename):
    """
    Retrieves .txt files from S3 storage bucket and prepares data for inverted indexing, word frequency analysis, and word cloud generation.
    """

    word_counts = {}
    word_frequencies = {}

    # Download stop words list from S3
    stop_words = set()
    stopwords_data = s3_client.get_object(Bucket=output_bucket, Key=stopwords_file_key)
    for line in stopwords_data['Body'].read().decode("latin-1").splitlines():
        stop_words.add(line.strip().lower())

    for file in s3_client.list_objects_v2(Bucket=input_bucket)['Contents']:
        file_key = file['Key']
        file_path = f"s3://{input_bucket}/{file_key}"
        if not file_path.endswith('.txt'):
            continue

        data = s3_client.get_object(Bucket=input_bucket, Key=file_key)
        contents = data['Body'].read().decode("latin-1")  # This setting removed the decoding byte error

        for word in contents.split():
            word = word.strip().lower()
            word = re.sub(r"[^\w]+", "", word)  # Remove punctuation

            if word.isalpha() and word and word not in stop_words:  # Exclude stop words
                if word not in word_counts:
                    word_counts[word] = set()
                    word_frequencies[word] = 0
                word_counts[word].add(file_key)
                word_frequencies[word] += 1

    return word_counts, word_frequencies


def build_inverted_index(input_bucket, output_bucket, word_counts, word_frequencies):
    """
    Builds an inverted index from the processed text files in S3, generates a word cloud, and uploads the outputs to S3.
    """

    inverted_index = {}

    for word, filenames in word_counts.items():
        if word not in inverted_index:
            inverted_index[word] = set()
        inverted_index[word].update(filenames)

    # Merge and write inverted index to a single file
    inverted_index_content = ""
    for word, filenames in sorted(inverted_index.items()):
        inverted_index_content += f"{word} : {','.join(filenames)}\n"

    s3_client.put_object(
        Body=inverted_index_content.encode('utf-8'),
        Bucket=output_bucket,
        Key="inverted_index_output.txt",
    )

    # Write word frequencies to a separate file
    word_frequency_content = ""
    for word, frequency in sorted(word_frequencies.items()):
        word_frequency_content += f"{word}: {frequency}\n"

    s3_client.put_object(
        Body=word_frequency_content.encode('utf-8'),
        Bucket=output_bucket,
        Key="word_frequency.txt",
    )

    # Generate and upload word cloud
    word_cloud = WordCloud().generate_from_frequencies(word_frequencies)
    plt.figure(figsize=(10, 6))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("word_cloud.jpg", bbox_inches='tight')
    with open("word_cloud.jpg", "rb") as f:
        s3_client.upload_fileobj(f, output_bucket, output_word_cloud_path)


word_counts, word_frequencies = process_text_file(input_corpus_path)
build_inverted_index(input_bucket, output_bucket, word_counts, word_frequencies)