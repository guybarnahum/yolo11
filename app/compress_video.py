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

def compress_video_to_bitrate(input_path, output_path, video_bitrate, audio_bitrate, two_pass=True, add_timecode=True):
    
    logging.info(f"compress_video_to_bitrate : {input_path} >> {output_path} - {video_bitrate} b/sec")
    
    try:
        i = ffmpeg.input(input_path)
        
        # Base output arguments
        base_output_args = {'c:v': 'libx264', 'b:v': video_bitrate}
        
        # Add timecode metadata if requested
        if add_timecode:
            # Get input frame rate to calculate timecode
            probe = ffmpeg.probe(input_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if video_stream:
                # Get framerate from the input
                if 'r_frame_rate' in video_stream:
                    fps_str = video_stream['r_frame_rate']
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        fps = num / den if den != 0 else 0
                    else:
                        fps = float(fps_str)
                
                    # Add timecode metadata - start at 00:00:00:00
                    base_output_args['timecode'] = '00:00:00:00'
                    base_output_args['r'] = fps  # Ensure the same frame rate
        
        if two_pass:
            # First pass
            pass1_args = base_output_args.copy()
            pass1_args['pass'] = 1
            pass1_args['f'] = 'mp4'
            
            ffmpeg.output(i, os.devnull, **pass1_args
                         ).global_args('-nostats', '-loglevel', 'warning').overwrite_output().run()
            
            # Second pass
            pass2_args = base_output_args.copy()
            pass2_args['pass'] = 2
            pass2_args['c:a'] = 'aac'
            pass2_args['b:a'] = audio_bitrate
            
            ffmpeg.output(i, output_path, **pass2_args
                         ).global_args('-nostats', '-loglevel', 'warning').overwrite_output().run()
        else:
            # Single pass
            single_pass_args = base_output_args.copy()
            single_pass_args['c:a'] = 'aac'
            single_pass_args['b:a'] = audio_bitrate
            
            ffmpeg.output(i, output_path, **single_pass_args
                         ).global_args('-nostats', '-loglevel', 'warning').overwrite_output().run()
        
        return output_path
        
    except Exception as e:
        loggin.error(f'compress_video_to_bitrate - {str(e)}')
        return False


def compress_video_to_size(input_path, output_path, size_upper_bound):

    '''
    Compresses a video file to stay under a specified file size limit.
    
    This function calculates the appropriate video and audio bitrates to achieve
    the target file size, then compresses the video accordingly using a two-pass
    encoding process. It verifies the final file size and logs an error if the
    size limit is exceeded.
    
    Parameters:
        input_path (str): Path to the input video file to be compressed
        output_path (str): Path where the compressed video will be saved
        size_upper_bound (float): Maximum file size in KB
    
    Returns:
        str: Path to the compressed video file (same as output_path)
    
    Raises:
        Logs an error if the compressed video exceeds the specified size limit
    
    Note:
        This function depends on calculate_bit_rate() and compress_video_to_bitrate()
        functions above.
    '''
    video_bitrate, audio_bitrate = calculate_bit_rate(input_path, size_upper_bound)
    
    compress_video_to_bitrate(input_path, output_path, video_bitrate, audio_bitrate, two_pass=True )

    actual_size = os.path.getsize(output_path)
    size_limit  = size_upper_bound * 1024

    if  actual_size > size_limit:
        logging.error(f'video {output_path} size {actual_size} is bigger than size limit {size_limit}')

    return output_path

