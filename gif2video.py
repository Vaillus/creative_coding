from moviepy.editor import VideoFileClip, concatenate_videoclips
clip = VideoFileClip("test.gif")
clip = clip.resize(width=1080, height=1080)
final_clip = concatenate_videoclips([clip] * 5)
final_clip.write_videofile("gallery/new.mp4",
                           codec='libx264', 
                           audio=False,
                           preset='ultrafast',
                           ffmpeg_params=["-pix_fmt", "yuv420p"]
)

