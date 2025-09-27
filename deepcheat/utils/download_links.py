"""
Downloads youtube videos from a csv of URLs
"""
import argparse

import pandas as pd
from pytube import YouTube
from pytube.exceptions import VideoUnavailable, AgeRestrictedError, VideoRegionBlocked, MembersOnly
from pytube.exceptions import LiveStreamError, RecordingUnavailable, VideoPrivate, RegexMatchError
from urllib.parse import urlparse, parse_qs



def get_youtube_uuid(url):
    parsed_url = urlparse(url)
    if 'youtube.com' in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    elif 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.lstrip('/')
    else:
        return None

def download_best_video(row, min_res, max_res, output_path, max_retries=2):
    """Downloads the Highest resolution, Highest FPS stream of a given video URL from YouTube"""

    retry = 0
    while retry <= max_retries:
        print(f"Downloading URL: {row['url']}")

        # create YT object
        try:
            yt_obj = YouTube(row['url'])
        except RegexMatchError:
            print(f"Bad Video URL: {row['url']}")
            break

        # get list of all mp4 video streams
        try:
            all_streams = yt_obj.streams.filter(file_extension='mp4',type='video')
        except (VideoUnavailable, AgeRestrictedError, VideoRegionBlocked,
                LiveStreamError, RecordingUnavailable, MembersOnly, VideoPrivate) as error:
            print(f"{row['url']} is not available for download: {error}")
            break
        except KeyError:
            retry += 1
            print(f"Error Getting Video Streams, retrying... ({retry}/{max_retries})")
            continue

        # get set of all available resolutions
        available_resolutions = []
        for stream in all_streams:
            available_resolutions.append(stream.resolution.split('p')[0])
        available_resolutions = [*set(available_resolutions)]

        # get streams with resolution that meets criteria
        good_resolutions = [int(res) for res in available_resolutions if int(res) >= int(min_res)]
        good_resolutions = [res for res in good_resolutions if res <= int(max_res)]
        if not good_resolutions:
            print(f"No streams available matching resolution criteria for {row['url']}!")
            break

        # get highest resolution
        best_res = str(max(good_resolutions)) + 'p'

        # get highest fps stream at best res
        best_stream = yt_obj.streams \
                            .filter(file_extension='mp4', type='video', res=best_res) \
                            .order_by('fps') \
                            .last()

        # download video
        out_filename = get_youtube_uuid(row['url']+".mp4")
        best_stream.download(output_path=output_path,
                            filename=out_filename,
                            skip_existing=True,
                            max_retries=max_retries)

        return True

    print(f"Could not download {row['url']}!")
    return False

def download_videos_from_csv(input_path, output_path, min_res=360, max_res=1080):
    """Reads URLs into dataframe and downloads each one"""

    # read csv into dataframe
    print(f'Reading Input CSV: {input_path}')
    urls_df = pd.read_csv(input_path)
    urls_df.dropna(inplace=True)

    # download all videos
    print('Beginning Downloads...')
    urls_df.apply(download_best_video,
                  min_res=min_res,
                  max_res=max_res,
                  output_path=output_path,
                  axis=1)

    print('Downloads Complete')

def main():

    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Get URL's from API and store them locally")
    parser.add_argument("-i", "--input", help="Path to CSV of Youtube URLs to download",
                        required=True, type=str)
    parser.add_argument("-o", "--output", help='Folder To Store Downloaded Videos',
                        required=True, type=str)
    parser.add_argument("-n", "--min_resolution", help='The minimum resolution allowed',
                        required=False, type=str, default='360')
    parser.add_argument("-x", "--max_resolution", help='The maximum resolution allowed',
                        required=False, type=str, default='1080')
    args = vars(parser.parse_args())


    """Download all YouTube URLs from an input CSV"""
    # read csv into dataframe
    download_videos_from_csv(args['input'],
                             args['output'],
                             args['min_resolution'],
                             args['max_resolution'])

if __name__ == "__main__":
    main()
