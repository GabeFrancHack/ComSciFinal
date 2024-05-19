#import pandas as pd

#df = pd.read_csv('tiktok_collected_liked_videos.csv', delimiter=',')
#df.drop(['user_name','user_id','video_id','video_desc','video_time','video_link'],axis=1)
#df['userliked']=0
#df['engagement']=0.0
#list_of_rows=[list(row) for row in df.values]
#print(df.head())
#print(df.info())

import pandas as pd

#read code & check for compiling errors
try:
    df = pd.read_csv('tiktok_collected_liked_videos.csv', delimiter=',')
except FileNotFoundError:
    print('file not found')
except pd.errors.ParserError:
    print('error analyzing')

#throwing out unwanted
columns_to_drop = ['user_name', 'user_id', 'video_id', 'video_desc', 'video_time', 'video_link']
df.drop(columns=columns_to_drop, inplace=True)

#adding outputs
df['userliked'] = 0.0

#wuda hell (convert data to floats)
df = df.astype('float64')

#compile the list
list_of_rows = df.values.tolist()

#print(list_of_rows)
print(df.head())
print(df.info())