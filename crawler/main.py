import pandas as pd
from crawl_channel import crawl_channel
from audio_captions import download_add_audio_captions_csv

def main():

    # download_add_audio_captions_csv("youtube_shorts.csv")

    crawl_channel("UC0U1sgCZRpocI5EQ7fG3_zQ")

    # Specify the path to your CSV file
    csv_file = 'youtube_shorts.csv'

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Display the DataFrame
    print(df)


if __name__ == "__main__":
    main()