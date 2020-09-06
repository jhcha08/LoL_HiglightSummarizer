import os,shutil, sys
import numpy as np
import pandas as pd
import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips

def make(csv_dir, mp4_dir):
    # 모든 csv 불러와서 돌아가면서 시간값, 이름 읽기
    all_csv_name_list = sorted(glob.glob(os.path.join(csv_dir,'*.csv')))
    for csv_name in all_csv_name_list:
        df = pd.read_csv(csv_name)
        print(df)
        start = np.array(df['start_sec'][df['pred'] == 1])
        end = np.array(df['end_sec'][df['pred'] == 1])
        name = os.path.split(os.path.splitext(csv_name)[0])[1]
        
        i=1
        while(i<len(end)):
            if end[i] - 3 == end[i-1]:
                end[i-1] = end[i]
                start = np.delete(start, i)
                end = np.delete(end, i)
            else:
                i+=1
        
        #Create temporary folder for storing subclips generated. This folder will be deleted later after highlights are generated.
        cwd=os.getcwd()
        sub_folder=os.path.join(cwd,"Subclips")
        if os.path.exists(sub_folder):
            shutil.rmtree(sub_folder)
            path=os.mkdir(sub_folder)
        else:
            path=os.mkdir(sub_folder)

        clip = VideoFileClip(os.path.join(mp4_dir,name+".mp4"))
        for i in range(len(start)):
            start_lim = start[i]
            end_lim = end[i]
            filename = "highlight" + str(i+1) + ".mp4"
            subclip=clip.subclip(start_lim, end_lim)
            subclip.write_videofile(os.path.join(sub_folder,filename))
        subclip.close()
        clip.close()
        
        files = sorted(glob.glob(os.path.join(sub_folder,'*.mp4')))
        videoclips = [VideoFileClip(i) for i in files]
        final_clip=concatenate_videoclips(videoclips)
        final_clip.write_videofile("./Highlights" + name + ".mp4") #Enter the desired output highlights filename.
        shutil.rmtree(sub_folder) #Delete the temporary file.
        for i in videoclips:
            i.close()

def main():
    if len(sys.argv) < 2:
        print('csv_dir, mp4_dir 입력해주세요.')
        return
    
    csv_dir = sys.argv[1]
    mp4_dir = sys.argv[2]
    
    make(csv_dir, mp4_dir)

if __name__ == "__main__":
    main()
