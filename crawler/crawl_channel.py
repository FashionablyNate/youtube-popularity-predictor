import os
import csv
import logging
import googleapiclient.discovery
import googleapiclient.errors
import re
from database import insert_item, item_exists
import config

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

def init():
    api_service_name = "youtube"
    api_version = "v3"
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=config.API_KEY)

    return youtube

def get_channel_subscriber_count(youtube, channel_id):
    try:
        request = youtube.channels().list(
            part="statistics",
            id=channel_id
        )
        response = request.execute()
        print("API Call: subscriber count")

        return response['items'][0]['statistics'].get('subscriberCount')
    except googleapiclient.errors.HttpError as e:
        print( f"An HTTP error {e.resp.status} occurred: {e.content}" )
        return None


def get_video_ids_from_channel_ID(youtube, channel_id, page_token=""):
    try:
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            pageToken=page_token,
            type="video",
            videoDuration="short"
        )
        response = request.execute()
        print("API Call: video ids")
        video_ids = [item['id']['videoId'] for item in response['items']]
        next_page_token = response.get('nextPageToken')

        return ",".join(video_ids), next_page_token
    except googleapiclient.errors.HttpError as e:
        print( f"An HTTP error {e.resp.status} occurred: {e.content}" )
        return None, None

# Helper function to parse ISO 8601 format
def parse_duration(duration):
    # Parsing ISO 8601 duration format
    match = re.match('PT(\d+M)?(\d+S)?', duration)
    minutes = int(match.group(1)[:-1]) if match.group(1) else 0
    seconds = int(match.group(2)[:-1]) if match.group(2) else 0
    return 60 * minutes + seconds

def get_video_data_from_videos(youtube, video_ids, subscriber_count):
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_ids
        )
        response = request.execute()
        print("API Call: video data")

        videos = []
        for item in response['items']:
            video_length = parse_duration(item['contentDetails']['duration'])
            title = item['snippet']['title']
            description = item['snippet']['description']

            if video_length > 60 or not re.search('rogan', title, re.IGNORECASE) and not re.search('rogan', description, re.IGNORECASE) \
                and not re.search('jre', title, re.IGNORECASE) and not re.search('jre', description, re.IGNORECASE):
                continue
                
            video = {
                'video_id': item['id'],
                'video_length': video_length,
                'publication_date_time': item['snippet']['publishedAt'],
                'number_of_likes': item['statistics'].get('likeCount'),
                'number_of_comments': item['statistics'].get('commentCount'),
                'number_of_views': item['statistics'].get('viewCount'),
                'video_description': item['snippet']['description'],
                'title': item['snippet']['title'],
                'video_tags': item['snippet'].get('tags', []),
                'thumbnail_url': item['snippet']['thumbnails']['default']['url'],
                'video_url': f"https://www.youtube.com/watch?v={item['id']}",
                'subscriber_count': subscriber_count # Use the subscriber count passed in
            }
            videos.append(video)

        return videos
    except googleapiclient.errors.HttpError as e:
        print( f"An HTTP error {e.resp.status} occurred: {e.content}" )
        return None



def write_to_csv(videos, file):
    if videos:
        keys = videos[0].keys()
        with open(file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, quoting=csv.QUOTE_ALL)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(videos)

def crawl_channel(channel_id):

    if not item_exists(channel_id):

        # Set up logging
        logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

        youtube = init()
        
        # Get subscriber count once per channel
        subscriber_count = get_channel_subscriber_count(youtube, channel_id)

        page_token = ""
        while page_token is not None:
            video_ids, page_token = get_video_ids_from_channel_ID(youtube, channel_id, page_token)
            if video_ids:
                videos = get_video_data_from_videos(youtube, video_ids, subscriber_count)
                write_to_csv(videos, "youtube_shorts.csv")
        insert_item(channel_id)
