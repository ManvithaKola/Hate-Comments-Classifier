import json
from csv import writer
from apiclient.discovery import build

def build_service(filename):
    with open(filename) as f:
        key = f.readline()

    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME,
                 YOUTUBE_API_VERSION,
                 developerKey=key)


def get_comments(part='snippet', 
                 maxResults=100, 
                 textFormat='plainText',
                 order='time',
                 #videoId='DsOVVqubBus'
                 videoId='jqcO5sp9JgU'):
       
    comments = []
       
    
    service = build_service('apikey.json')
    
    
    response = service.commentThreads().list(
        part=part,
        maxResults=maxResults,
        textFormat=textFormat,
        order=order,
        videoId=videoId
    ).execute()
                 
    print(response)
    while response: 
        print('enter')        
        for item in response['items']:
            
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']            
            comments.append(comment)
            print(comment)
            
            with open('hatecomments1.csv', 'a', encoding="utf-8") as f:
                csv_writer = writer(f)
                csv_writer.writerow([comment])
        
       
        if 'nextPageToken' in response:
            response = service.commentThreads().list(
                part=part,
                maxResults=maxResults,
                textFormat=textFormat,
                order=order,
                videoId=videoId,
                pageToken=response['nextPageToken']
            ).execute()
        else:
            break

    
    return {
        'Comments': comments
    }
  
comments = get_comments()
