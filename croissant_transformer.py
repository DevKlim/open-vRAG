import pandas as pd
import json
import os
import datetime
import sys

# input dataframe row
# output croissant json
def Croissant_JSON_formating(row):
    status_id = row['twitterlink'].split('/')[-1]

    croissant_JSON = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/1.0",
            "sc": "https://schema.org/"
        },
        "@type": "sc:Dataset",
        "name": f"Tweet-ID-{status_id}",
        "description": f"A single social media video post(Tweet ID: {status_id}) by the user, structured as a Croissant Dataset.",
        "url": row['twitterlink'],
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "datePublished": row['postdatetime'],
        "creator": {
            "@type": "sc:Organization",
            "name": "Social Media Manually Collect Source"
        },
        "distribution": [
            {
            "@type": "cr:FileObject",
            "name": f"source_tweet_url_{status_id}",
            "encodingFormat": "text/html",
            "contentUrl": row['twitterlink']
            }
        ],
        
        "recordSet": [
            {
            "@type": "cr:RecordSet",
            "name": "tweet_data_record",
            "description": "The specific data fields for this tweet.",
            

            "field": [
                {
                "@type": "cr:Field",
                "name": "internal_id",
                "description": "Internal record ID from the source CSV.",
                "data_type": "sc:Integer",
                "sc:value": int(row['id'])
                },
                {
                "@type": "cr:Field",
                "name": "tweet_text",
                "description": "The full text/caption of the tweet.",
                "data_type": "sc:Text",
                "sc:value": row['caption']
                },
                {
                "@type": "cr:Field",
                "name": "likes_count",
                "description": "Number of likes/favorites.",
                "data_type": "sc:Integer",
                "sc:value": int(row['likes'])
                },
                {
                "@type": "cr:Field",
                "name": "shares_count",
                "description": "Number of shares/reposts.",
                "data_type": "sc:Integer",
                "sc:value": int(row['shares'])
                },
                {
                "@type": "cr:Field",
                "name": "post_timestamp",
                "description": "When the tweet was posted (ISO 8601).",
                "data_type": "sc:DateTime",
                "sc:value": row['postdatetime']
                },
                {
                "@type": "cr:Field",
                "name": "collection_timestamp",
                "description": "When this data was collected (ISO 8601).",
                "data_type": "sc:DateTime",
                "sc:value": row['collecttime']
                },
                {
                "@type": "cr:Field",
                "name": "video_transcript_path",
                "description": "Path to the video transcription file.",
                "data_type": "sc:Text",
                "sc:value": row['videotranscriptionpath'] if pd.notna(row['videotranscriptionpath']) else ""
                }
            ]
            }
        ]
        }
    return croissant_JSON


# input twitterVideo.csv file path
# output folder "croissant_tweet_metadata" in current directory
# for every row in csv file
def croissant_transform(file_path):
    # set output directory
    OUTPUT_DIR = "croissant_tweet_metadata"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # read csv
    df = pd.read_csv(file_path)

    # columns with useful information, hard coded
    df = df.iloc[:,0:9]

    # loop
    for index, row in df.iterrows():
        croissant_json = Croissant_JSON_formating(row)
        
        # save JSON-LD file
        file_path = os.path.join(OUTPUT_DIR, f"tweet_metadata_{index+1}.json")
        with open(file_path, 'w') as f:
            json.dump(croissant_json, f, indent=2)
        print(f"{file_path} generated")

    print("Conversion Finished")
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python croissant_transformer.py <input_csv_path>")
        sys.exit(1)

    input_csv_path = sys.argv[1]
    croissant_transform(input_csv_path)

