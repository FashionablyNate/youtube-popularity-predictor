import os
import pandas as pd
import subprocess
from tqdm import tqdm

def download_captions_and_audio(video_url):

    if not os.path.exists("audio"):
        os.makedirs("audio")
    if not os.path.exists("caps"):
        os.makedirs("caps")

    audio_file = 'audio/' + video_url.split('=')[-1] + '.wav'
    subtitle_file = 'caps/' + video_url.split('=')[-1] + '.en.vtt'

    # Skip download if both audio and subtitle files exist
    if not os.path.exists(audio_file) and not os.path.exists(subtitle_file):
        # Download subtitles
        if not os.path.exists(subtitle_file):
            result = subprocess.run(['yt-dlp', '--write-auto-sub', '--sub-lang', 'en', '--skip-download', video_url, '-o', subtitle_file[:-6]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                print(f"Error downloading subtitles for {video_url}. Return code: {result.returncode}")

        # Download audio
        if not os.path.exists(audio_file):
            result = subprocess.run(['yt-dlp', '-x', '--audio-format', 'wav', video_url, '-o', audio_file[:-4]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                print(f"Error downloading audio for {video_url}. Return code: {result.returncode}")

    # Read subtitle text
    subtitle_text = None
    if os.path.exists(subtitle_file):
        with open(subtitle_file, 'r') as f:
            subtitle_text = f.read()

    return subtitle_text, audio_file if os.path.exists(audio_file) else None


def download_add_audio_captions_csv(csv_file):
    # Read the csv file into a DataFrame
    df = pd.read_csv(csv_file)

    # Apply the function to each video URL with tqdm loading bar
    results = [download_captions_and_audio(url) for url in tqdm(df['video_url'], total=df.shape[0])]
    df['subtitle_text'], df['audio_file'] = zip(*results)

    # Write the DataFrame back to csv
    df.to_csv(csv_file, index=False)

    print(df)
