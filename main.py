import pprint
import itertools
import os
import subprocess
import json
import re
import shutil
import ffmpeg
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import asyncio
console = Console()

max_worker = 6
max_retries = 10
gpu_devices = itertools.cycle([0, 1])

def progressbar(label, total):
    pbar = tqdm(total=total, desc=label, unit='B', unit_scale=True, unit_divisor=1024)
    return pbar

def get_video_info(input_path):
    command = [
        'ffprobe',
        '-v', 'error',
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    result = json.loads(result.stdout)
    return result

def process_bar(process):
    pbar = tqdm(total=None)
    for line in process.stdout:
       # 출력에서 "Duration: 00:00:
       # .04"와 같은 라인을 찾아서 전체 동영상 길이를 가져옵니다.
       if "Duration" in line:
           match = re.search("Duration: (.*?),", line)
           if match:
               time_str = match.group(1)
               try:
                   hours, minutes, seconds = map(float, re.split(':', time_str))
                   total_seconds = hours * 3600 + minutes * 60 + seconds
                   pbar.total = total_seconds
                   pbar.refresh()
               except ValueError:
                   # 유효하지 않은 시간 형식을 만났을 때의 처리
                   print(f"유효하지 않은 시간 형식: {time_str}")
                   continue  # 다음 라인으로 넘어갑니다.

           # 출력에서 "time=00:00:10.00"과 같은 라인을 찾아서 현재 진행 시간을 가져옵니다.
       if "time=" in line:
           match = re.search("time=(.*?) ", line)
           if match:
               time_str = match.group(1)
               try:
                   hours, minutes, seconds = map(float, re.split(':', time_str))
                   elapsed_seconds = hours * 3600 + minutes * 60 + seconds
                   pbar.n = elapsed_seconds
                   pbar.refresh()
               except ValueError:
                   # 유효하지 않은 시간 형식을 만났을 때의 처리
                   print(f"유효하지 않은 시간 형식: {time_str}")
                   continue  # 다음 라인으로 넘어갑니다.
    pbar.close()

def process_encoding(input_path, output_path):
    print("pre-encoding")
    command = [
        'ffmpeg',
        '-i', input_path,
        '-filter_complex', '[0:a]aresample=async=1:first_pts=0[a]',
        '-map', '0:v',
        '-map', '[a]',
        '-c:v', 'copy',
        '-c:a', 'pcm_s16le',
        '-strict', 'experimental',
        '-y',
        output_path
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)

def reverse_video(pre_processing ,input_path, output_dir, temp_dir, reversed_dir, audio_reversed_dir, output_name):
    print("reverse video")
    # input_path = f"{output_dir}/merged.mp4"
    # output_path = f"{output_dir}/reversed.mp4"
    segment_duration = 10

    if os.path.exists(pre_processing):
        shutil.rmtree(pre_processing)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(reversed_dir):
        shutil.rmtree(reversed_dir)
    if os.path.exists(audio_reversed_dir):
        shutil.rmtree(audio_reversed_dir)
    if not os.path.exists(pre_processing):
        os.makedirs(pre_processing)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(reversed_dir):
        os.makedirs(reversed_dir)
    if not os.path.exists(audio_reversed_dir):
        os.makedirs(audio_reversed_dir)

    process_encoding(input_path=input_path, output_path=f"{pre_processing}/encoded.mov")
    split_video(input_path=f'{pre_processing}/encoded.mov', segment_duration=segment_duration, temp_dir=temp_dir)
    asyncio.run(reverse_segment(temp_dir, reverse_dir))
    #reverse_audio(input_path, temp_dir, audio_reversed_dir)
    #concatenate_segments(reversed_dir=reversed_dir, output_path=f'{output_dir}/{output_name}.mp4')
    concatenate_segments(reversed_dir=reversed_dir, output_path = f"{temp_dir}/reversed.mov")
    #combine_audio_video(video_path=f"{temp_dir}/reversed.mov", audio_path=f'{temp_dir}/reversed_audio.mov', output_path=f'{output_dir}/{output_name}.mp4')

def split_video(input_path, segment_duration, temp_dir):
    print("split video")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = [
        'ffmpeg',
        '-i', input_path,
        '-c', 'copy',
        #'-c:v', 'h264_nvenc',
        # '-c:a', 'pcm_s16le',
        # '-map', '0:v',
        # '-map', '0:a',
        '-map', '0',
        '-start_at_zero',
        '-segment_time', str(segment_duration),
        '-f', 'segment',
        '-segment_format', 'mov',
        '-avoid_negative_ts', 'make_zero',
        '-break_non_keyframes', '0',
        #'-reset_timestamps', '1',
        #'-write_empty_segments', '1',
        '-copyts',  # 타임스탬프 복사
        #'-force_key_frames', f"expr:gte(t,n_forced*{segment_duration})",  # 강제 키프레임 설정
        f'{temp_dir}/segment%010d.mov',
        '-y'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)

async def worker(name, queue, progress, task_id):
    while True:
        segment_file, temp_dir, reversed_dir, gpu_device = await queue.get()
        try:
            await retry_process_segment(
                os.path.join(temp_dir, segment_file),
                os.path.join(reversed_dir, segment_file),
                gpu_device
            )
            progress.update(task_id, advance=1)
            completed_count = progress.tasks[task_id].completed  # 완료된 작업 수 가져오기
            total_count = progress.tasks[task_id].total  # 전체 작업 수 가져오기
            print(f"{name} 작업 완료: {segment_file} | 할당 GPU: {gpu_device} | 완료된 작업 수: {completed_count}|{total_count}")
        except Exception as e:
            print(f"{segment_file} 처리 중 오류 발생 (GPU {gpu_device}): {e}")
        finally:
            queue.task_done()

async def retry_process_segment(input_path, output_path, gpu_device):
    for attempt in range(1, max_retries + 1):
        try:
            await process_segment(input_path, output_path, gpu_device)
            return
        except Exception as e:
            if attempt < max_retries:
                print(f"오류 발생, 재시도 중 ({attempt}/{max_retries}): {input_path}")
                await asyncio.sleep(2)  # 재시도 전 대기 시간
            else:
                print(f"최대 재시도 횟수 초과: {input_path}")
                raise


async def reverse_segment(temp_dir, reversed_dir):
    if not os.path.exists(reversed_dir):
        os.makedirs(reversed_dir)

    segment_files = sorted(os.listdir(temp_dir))
    total_segments = len(segment_files)

    queue = asyncio.Queue(maxsize=1)
    workers = []

    # Rich Progress 설정
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}% [{task.completed}/{task.total}]"),
        TimeElapsedColumn(),  # 소요 시간 표시 추가
        TimeRemainingColumn()  # 남은 시간 표시 추가
    )

    with progress:
        task_id = progress.add_task("Processing", total=total_segments)

        for i in range(max_worker):
            worker_task = asyncio.create_task(worker(f"Worker-{i + 1}", queue, progress, task_id))
            workers.append(worker_task)

        for segment_file in segment_files:
            gpu_device = next(gpu_devices)
            await queue.put((segment_file, temp_dir, reversed_dir, gpu_device))

        await queue.join()

        for worker_task in workers:
            worker_task.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

async def process_segment(input_path, output_path, gpu_device):
    """하나의 비디오 세그먼트를 역순으로 만듭니다."""

    #Nvidia GPU를 사용하여 비디오 세그먼트를 역순으로 만듭니다.
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-fflags', '+genpts',
        #'-hwaccel_output_format', 'cuda',
        #'-c:v', 'h264_cuvid',
        '-hwaccel_device', f'{gpu_device}',
        '-i', input_path,
        #'-map', '0:v',
        #'-map', '0:a',
        '-start_at_zero',
        '-c:v', 'h264_nvenc',
        #'-vf', 'hwdownload,format=nv12,reverse,setpts=PTS-STARTPTS, hwupload=extra_hw_frames=64',  # 비디오 프레임 역순 및 타임스탬프 조정
        #'-vf', 'hwdownload,format=nv12,reverse, hwupload=extra_hw_frames=64',  # 비디오 프레임 역순 및 타임스탬프 조정
        '-vf', 'reverse, setpts=PTS-STARTPTS',
        '-af', 'areverse,aresample=async=1:min_hard_comp=0.100000:first_pts=0',  # 오디오 프레임 역순 및 타임스탬프 조정
        '-c:a', 'pcm_s16le',
        #'-af', 'areverse, asetpts=PTS-STARTPTS, atrim=start=0',  # 오디오 프레임 역순 및 타임스탬프 조정
        #'-gpu', f'{gpu_deivce}',
        #'-channel_layout', 'stereo',
        '-avoid_negative_ts', 'make_zero',
        '-format', 'mov',
        #'-an',
        '-r', '60',
        '-copyts',  # 타임스탬프 복사
        '-b:v', '8000k',
        output_path,
        '-y'
    ]


    process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise Exception(f"Error during processing: {stdout} | {stderr}")


def reverse_audio(input_path, temp_dir, audio_reversed_dir):
    print("reverse audio")

    command = [
        'ffmpeg',
        '-i', input_path,
        '-start_at_zero',
        '-threads', '0',
        # '-map', '0:a',
        '-vn',
        '-progress', '-',  # 진행률 표시
        #'-af', 'areverse',  # 오디오 프레임 역순
        '-af', 'areverse, asetpts=PTS-STARTPTS, atrim=start=0',  # 오디오 프레임 역순 및 타임스탬프 조정
        '-c:a', 'pcm_s16le',
        '-avoid_negative_ts', 'make_zero',
        '-copyts',  # 타임스탬프 복사
        f'{temp_dir}/reversed_audio.mov',
        '-y'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)
    print("reverse audio Done")

# def reverse_audio(input_path, temp_dir, audio_reversed_dir):
#     print("reverse audio")
#     if not os.path.exists(audio_reversed_dir):
#         os.makedirs(audio_reversed_dir)
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)
#     if not os.path.exists(f"{temp_dir}/audio"):
#         os.makedirs(f"{temp_dir}/audio")
#
#     temp_dir_audio = f"{temp_dir}/audio"
#     segment_duration = 10
#     split_audio(input_path, segment_duration, temp_dir_audio)
#     reverse_segment_audio_process(temp_dir_audio, audio_reversed_dir)
#     concat_segments_audio(audio_reversed_dir, temp_dir)

def split_audio(input_path, segment_duration, temp_dir):

    print("split audio")
    command = [
        'ffmpeg',
        '-i', input_path,
        '-map', '0:a',
        '-vn',
        '-c', 'copy',
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-break_non_keyframes', '0',
        '-reset_timestamps', '1',
        '-channel_layout', 'stereo',
        '-y',
        f'{temp_dir}/audio_segment%010d.ts'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)

def reverse_segment_audio_process(temp_dir, reversed_dir):
    """분할된 모든 오디오 세그먼트를 역순으로 만듭니다."""
    if not os.path.exists(reversed_dir):
        os.makedirs(reversed_dir)

    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        count = 0
        segment_files = sorted(os.listdir(temp_dir))
        total_segments = len(segment_files)

        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            future_to_segment = {executor.submit(process_segment_audio, os.path.join(temp_dir, seg_file),
                                                 os.path.join(reversed_dir, seg_file)): seg_file for seg_file in
                                 segment_files}
            for i, future in enumerate(as_completed(future_to_segment), 1):
                segment = future_to_segment[future]
                try:
                    future.result()
                    count += 1
                    print(f"작업 완료: {segment} | 남은 작업 수: {total_segments - i} | 완료된 작업 수: {count} | 총 작업 수: {total_segments}")
                except Exception as exc:
                    print(f"{segment} 처리 중 에러 발생: {exc}")

def process_segment_audio(input_path, output_path):
    """하나의 오디오 세그먼트를 역순으로 만듭니다."""
    command = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-i', input_path,
        '-map', '0:a',
        '-start_at_zero',
        '-af', 'areverse',  # 오디오 프레임 역순 및 타임스탬프 조정
        '-c:a', 'aac',
        '-y',
        output_path
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)

def concat_segments_audio(reversed_dir, output_path):
    print("concatenate segments audio")
    with open('audio_segments.txt', 'w') as f:
        for segment_file in sorted(os.listdir(reversed_dir), reverse=True):
            f.write(f"file '{os.path.join(reversed_dir, segment_file)}'\n")

    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'audio_segments.txt',
        '-c', 'copy',
        f'{output_path}/reversed_audio.mp4',
        '-progress', '-',  # 진행률 표시
        '-y',
    ]

    process = subprocess.run(command, universal_newlines=True)
    #process_bar(process)
    print("concatenate segments audio Done")
    os.remove('audio_segments.txt')

def concatenate_segments(reversed_dir, output_path):
    print("concatenate segments")
    with open('segments.txt', 'w') as f:
        for segment_file in sorted(os.listdir(reversed_dir), reverse=True):
            f.write(f"file '{os.path.join(reversed_dir, segment_file)}'\n")

    command = [
        'ffmpeg',
        '-start_at_zero',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'segments.txt',
        '-fflags', '+genpts',
        '-c', 'copy',
        '-threads', '0',
        f'{output_path}',
        '-progress', '-',  # 진행률 표시
        '-y'
    ]

    subprocess.run(command, universal_newlines=True)
    print("reversed Done")
    os.remove('segments.txt')

def combine_audio_video(video_path, audio_path, output_path):
    print("combine audio video")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-threads', '0',
        f"{output_path}",
        '-progress', '-',
        '-y'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)
    print("combine audio video Done")

def run(dir_path, output_dir, divide_tmp, merge_tmp, temp_dir, reverse_dir, audio_reversed_dir):
    for root, dirs, files in os.walk(dir_path):
        for i, file in enumerate(files):
            if file.endswith(('.mp4', '.ts')):
                input_path = os.path.join(root, file)
                print(f"Processing: {input_path}")
                print(f"Processing remaining: {len(files)} : {len(files) - (i + 1)} |  {file}")
                # divide_2x2_with_progress(input_path, divide_tmp)
                # merge_tile_2x2(divide_tmp, f"{merge_tmp}/merged.mp4")
                reverse_video(input_path, output_dir, temp_dir, reverse_dir, audio_reversed_dir, file)


if __name__ == '__main__':
    input_path = 'reversed.mov'
    pre_processing = './output/pre_processing'
    temp_dir = './output/temp'
    divide_tmp = './output/divide_tmp'
    merge_tmp = './output/merge_tmp'
    reverse_dir = './output/reversed'
    audio_reversed_dir = './output/reversed_audio'
    dir_path = './test_dir'
    output_dir = './output'

    #run(pre_processing, dir_path, output_dir, divide_tmp, merge_tmp, temp_dir, reverse_dir, audio_reversed_dir)

    # divide_2x2_with_progress(input_path, output_dir)
    # merge_tile_2x2(output_dir, f"{output_dir}/merged.mp4")
    reverse_video(pre_processing, input_path, output_dir, temp_dir, reverse_dir, audio_reversed_dir, 'reversed_done')
    #reverse_audio(input_path, temp_dir, audio_reversed_dir)
    #concat_segments(filelist_path= f'{output_dir}/filelist.txt', output_path=f'{output_dir}/final_reversed.mp4')
    #concatenate_segments(reverse_dir, output_path = f"{output_dir}/reversed.mp4")
    #combine_audio_video(video_path=f"{output_dir}/reversed.mp4", audio_path=f'{temp_dir}/reversed_audio.mp4', output_path=f'{output_dir}/final_reversed.mp4')
