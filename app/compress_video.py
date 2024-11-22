# Simplified version and explanation at: https://stackoverflow.com/a/64439347/12866353
import logging
import os
import ffmpeg

def get_bitrate(input_path):

    probe = ffmpeg.probe(input_path)
    
    #print(probe['streams'])

    audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
    audio_bitrate = int(audio_stream['bit_rate']) if audio_stream else 0

    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    video_bitrate = int(video_stream['bit_rate']) if video_stream else 0

    return video_bitrate, audio_bitrate

def calculate_bit_rate(input_path, size_upper_bound):

    # Adjust them to meet your minimum requirements (in bps), or maybe this function will refuse your video!
    total_bitrate_lower_bound = 11000
    min_audio_bitrate = 32000
    max_audio_bitrate = 256000
    min_video_bitrate = 100000
    
    # Bitrate reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    probe = ffmpeg.probe(input_path)
    # Video duration, in s.
    duration = float(probe['format']['duration'])
    logging.info(f"{input_path} duration {duration} sec.")
    # print( probe["streams"] )

    # Target total bitrate, in bps.
    target_total_bitrate = (size_upper_bound * 1024 * 8) / (1.073741824 * duration)
    if target_total_bitrate < total_bitrate_lower_bound:
        logging.error('Bitrate is extremely low! Stop compress!')
        return False , False

    # Best min size, in kB.
    best_min_size = (min_audio_bitrate + min_video_bitrate) * (1.073741824 * duration) / (8 * 1024)
    if size_upper_bound < best_min_size:
        print('Quality not good! Recommended minimum size:', '{:,}'.format(int(best_min_size)), 'KB.')
        # return False

    audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

    if audio_stream:
        # Audio bitrate, in bps.
        audio_bitrate = float(audio_stream['bit_rate'])
        
        # target audio bitrate, in bps
        if 10 * audio_bitrate > target_total_bitrate:
            audio_bitrate = target_total_bitrate / 10
        
        if audio_bitrate < min_audio_bitrate < target_total_bitrate:
            audio_bitrate = min_audio_bitrate
        elif audio_bitrate > max_audio_bitrate:
            audio_bitrate = max_audio_bitrate
    else :
        audio_bitrate = 0

    # Target video bitrate, in bps.
    video_bitrate = target_total_bitrate - audio_bitrate if audio_stream else target_total_bitrate

    if video_bitrate < 1000:
        print('Bitrate {} is extremely low! Stop compress.'.format(video_bitrate))
        return False, False

    return video_bitrate, audio_bitrate

def compress_video_to_bitrate(input_path, output_path, video_bitrate, audio_bitrate, two_pass=True ):

    logging.info(f"compress_video_to_bitrate : {input_path} >> {output_path} - {video_bitrate} b/sec")
    
    try:
        i = ffmpeg.input(input_path)
        if two_pass:
            ffmpeg.output(i, os.devnull,
                          **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                          ).global_args('-nostats', '-loglevel', 'warning').overwrite_output().run()

            ffmpeg.output(i, output_path,
                          **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                          ).global_args('-nostats', '-loglevel', 'warning').overwrite_output().run()
        else:
            ffmpeg.output(i, output_path,
                          **{'c:v': 'libx264', 'b:v': video_bitrate, 'c:a': 'aac', 'b:a': audio_bitrate}
                          ).global_args('-nostats', '-loglevel', 'warning').overwrite_output().run()

        return output_path

    except FileNotFoundError as e:
        print('You do not have ffmpeg installed!', e)
        print('You can install ffmpeg by reading https://github.com/kkroening/ffmpeg-python/issues/251')
        return False


def compress_video_to_size(input_path, output_path, size_upper_bound):
    """
    """

    video_bitrate, audio_bitrate = calculate_bit_rate(input_path, size_upper_bound)
    
    compress_video_to_bitrate(input_path, output_path, video_bitrate, audio_bitrate, two_pass=True )

    if os.path.getsize(output_path) <= size_upper_bound * 1024:
        return output_path
    else:
        return False

