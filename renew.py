import os
import subprocess
import re
import shutil
import sys
import signal
import psutil
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import asyncio

console = Console()

timeout = 60

# GPU 설정
gpu_config = {
    0: 7,
}

# GPU 세마포어 초기화
gpu_semaphores = {gpu_id: asyncio.Semaphore(max_tasks) for gpu_id, max_tasks in gpu_config.items()}

max_worker = sum(gpu_config.values())  # 전체 GPU에서 동시에 실행 가능한 최대 작업 수
max_retries = 10


def get_subprocess_kwargs():
    """운영 체제에 따른 subprocess 인자를 반환합니다."""
    if sys.platform != "win32":
        return {"preexec_fn": os.setsid}
    return {}

async def terminate_process(process, timeout=5):
    """강제로 프로세스를 종료합니다."""
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Process did not terminate in time. Sending SIGKILL...")
        process.kill()  # SIGKILL 송신

def process_bar(process):
    pbar = tqdm(total=None)
    for line in process.stdout:
        # line이 이미 문자열이므로 디코딩할 필요가 없습니다.
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
                    print(f"유효하지 않은 시간 형식: {time_str}")
                    continue
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
                    print(f"유효하지 않은 시간 형식: {time_str}")
                    continue
    pbar.close()


async def get_available_gpu():
    while True:
        for gpu_id, semaphore in gpu_semaphores.items():
            if semaphore._value > 0:  # 세마포어에 여유가 있는 경우
                return gpu_id
        await asyncio.sleep(0.1)  # 잠시 대기 후 다시 확인


async def process_bar_async(process):
    while True:
        line = await process.stdout.readline()
        if not line:
            break


async def worker(name, queue, progress, task_id):
    while True:
        segment_file, temp_dir, reversed_dir = await queue.get()
        try:
            gpu_device = await get_available_gpu()
            async with gpu_semaphores[gpu_device]:
                await retry_process_segment(
                    os.path.join(temp_dir, segment_file),
                    os.path.join(reversed_dir, segment_file),
                    gpu_device
                )
            progress.update(task_id, advance=1)
            completed_count = progress.tasks[task_id].completed
            total_count = progress.tasks[task_id].total
            print(f"{name} 작업 완료: {segment_file} | 할당 GPU: {gpu_device} | 완료된 작업 수: {completed_count}/{total_count}")
        except Exception as e:
            print(f"{segment_file} 처리 중 오류 발생 (GPU {gpu_device}): {e}")
        finally:
            queue.task_done()


async def retry_process_segment(input_path, output_path, gpu_device, max_retries=max_retries, timeout=timeout):
    for attempt in range(1, max_retries + 1):
        try:
            await process_segment(input_path, output_path, gpu_device, timeout)
            return
        except TimeoutError:
            print(f"Timeout occurred. Retrying ({attempt}/{max_retries})...")
        except Exception as e:
            print(f"Error occurred: {e}. Retrying ({attempt}/{max_retries})...")

        if attempt < max_retries:
            await asyncio.sleep(2)

    raise Exception(f"Failed to process segment after {max_retries} attempts")


async def process_segment(input_path, output_path, gpu_device, timeout=timeout):
    command = [
        'ffmpeg',
        #'-correct_ts_overflow', '0',  # 타임스탬프 오버플로우 자동 수정 비활성화
        '-hwaccel', 'cuda',
        '-hwaccel_device', f'{gpu_device}',
        '-i', input_path,
        '-start_at_zero',
        # '-filter_complex', '[0:v]reverse,setpts=PTS-STARTPTS[v];[0:a]areverse[a]',
        '-c:v', 'h264_nvenc',
        '-vf', 'reverse, setpts=PTS-STARTPTS',
        # '-af', 'aresample=async=1000',
        '-af', 'areverse,atrim=start=0,aresample=async=1:min_hard_comp=0.100000:first_pts=0',
        # '-af', 'areverse,aresample=async=1:min_hard_comp=0.100000:first_pts=0',
        #'-af', 'areverse,atrim=start=0,aresample=async=1:first_pts=0',
        # '-af', 'areverse,atrim=start=0,aresample=async=1:min_hard_comp=0.100000:first_pts=0:min_comp=0.001:max_soft_comp=0.01',
        # '-af', 'areverse,atrim=start=0,aresample=async=1000:min_hard_comp=0.100000:first_pts=0:min_comp=0.001:max_soft_comp=0.01',
        '-c:a', 'pcm_s16le',
        '-map', '0:v',
        '-map', '0:a',
        #'-avoid_negative_ts', 'make_zero',
        '-format', 'mkv',
        '-framerate', '60',
        '-r', '60',
        '-vsync', 'cfr',
        '-ar', '48000',
        '-copyts',
        # '-fflags', '+genpts',
        '-b:v', '12M',
        '-maxrate:v', '18M',
        f'{output_path}',
        '-y',
        '-progress', 'pipe:1'
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,  # stdout을 완전히 무시합니다.
            stderr=asyncio.subprocess.PIPE,
            **get_subprocess_kwargs()
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Process timed out after {timeout} seconds. Terminating...")
            await terminate_process(process)
            raise TimeoutError(f"FFmpeg process timed out after {timeout} seconds")

        if process.returncode != 0:
            raise Exception(f"Error during processing: {stderr.decode()}")

    except Exception as e:
        print(f"Error processing segment: {e}")
        raise

async def reverse_segment(temp_dir, reversed_dir):
    if not os.path.exists(reversed_dir):
        os.makedirs(reversed_dir)

    segment_files = sorted(os.listdir(temp_dir))
    total_segments = len(segment_files)

    queue = asyncio.Queue()
    workers = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}% [{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    )

    with progress:
        task_id = progress.add_task("Processing", total=total_segments)

        for i in range(max_worker):
            worker_task = asyncio.create_task(worker(f"Worker-{i + 1}", queue, progress, task_id))
            workers.append(worker_task)

        for segment_file in segment_files:
            await queue.put((segment_file, temp_dir, reversed_dir))

        await queue.join()

        for worker_task in workers:
            worker_task.cancel()

        await asyncio.gather(*workers, return_exceptions=True)


def reverse_video(pre_processing, input_path, output_dir, temp_dir, reversed_dir, audio_reversed_dir, output_name):
    print("reverse video")
    segment_duration = 10

    # 디렉토리 정리 및 생성
    for dir_path in [pre_processing, temp_dir, reversed_dir, audio_reversed_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_video(input_path=input_path, segment_duration=segment_duration, temp_dir=temp_dir)
    asyncio.run(reverse_segment(temp_dir, reversed_dir))
    concatenate_segments(reversed_dir=reversed_dir, output_path=f'{output_dir}/{output_name}')


def split_video(input_path, segment_duration, temp_dir):
    print("split video")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    input_format = os.path.splitext(input_path)[1].lower()

    if input_format == '.mp4':
        # MP4 입력 파일을 위한 명령
        command = [
            'ffmpeg',
            #'-correct_ts_overflow', '0',  # 타임스탬프 오버플로우 자동 수정 비활성화
            '-i', input_path,
            '-c', 'copy',
            '-map', '0',
            '-start_at_zero',
            '-segment_time', str(segment_duration),
            # '-segment_overlap', '1',  # 1초의 오버랩 추가
            '-f', 'segment',
            '-segment_format', 'mkv',
            '-reset_timestamps', '1',
            #'-avoid_negative_ts', 'make_zero',
            '-break_non_keyframes', '0',
            '-copyts',
            '-fflags', '+genpts',
            f'{temp_dir}/segment%010d.mkv',
            '-y'
        ]
    else:
        # 다른 형식의 입력 파일을 위한 기존 명령
        command = [
            'ffmpeg',
            #'-correct_ts_overflow', '0',  # 타임스탬프 오버플로우 자동 수정 비활성화
            '-i', input_path,
            '-c', 'copy',
            '-map', '0',
            '-start_at_zero',
            '-segment_time', str(segment_duration),
            # '-segment_overlap', '1',  # 1초의 오버랩 추가
            '-f', 'segment',
            '-segment_format', 'mkv',
            '-reset_timestamps', '1',
            #'-avoid_negative_ts', 'make_zero',
            '-break_non_keyframes', '0',
            '-copyts',
            '-fflags', '+genpts',
            f'{temp_dir}/segment%010d.mkv',
            '-y'
        ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    process_bar(process)


def concatenate_segments(reversed_dir, output_path):
    print("concatenate segments")
    with open('segments.txt', 'w') as f:
        for segment_file in sorted(os.listdir(reversed_dir), reverse=True):
            f.write(f"file '{os.path.join(reversed_dir, segment_file)}'\n")

    command = [
        'ffmpeg',
        #'-correct_ts_overflow', '0',  # 타임스탬프 오버플로우 자동 수정 비활성화
        '-f', 'concat',
        '-safe', '0',
        '-i', 'segments.txt',
        '-c', 'copy',
        '-c:v', 'copy',
        '-c:a', 'pcm_s16le',
        '-af', 'aresample=async=1:first_pts=0',
        # '-c:a', 'libopus',
        '-movflags', '+faststart',
        '-fflags', '+genpts',
        '-threads', '0',
        f'{output_path}-rec.mkv',
        '-progress', '-',  # 진행률 표시
        '-y'
    ]

    subprocess.run(command, universal_newlines=True)
    print("reversed Done")
    os.remove('segments.txt')


def run(pre_processing, dir_path, output_dir, divide_tmp, merge_tmp, temp_dir, reverse_dir, audio_reversed_dir):
    print("run")
    print(f"dir_path: {dir_path}")
    for root, dirs, files in os.walk(dir_path):
        print(f"Inside os.walk - root: {root}")
        sorted_file = sorted(files)
        for i, file in enumerate(sorted_file):
            print(i)
            if file.endswith(('.mov', 'mkv', '.mp4', '.ts')):
                input_path = os.path.join(root, file)
                print(f"Processing: {input_path}")
                print(f"Processing remaining: {len(files)} : {len(files) - (i + 1)} |  {file}")
                reverse_video(pre_processing, input_path, output_dir, temp_dir, reverse_dir, audio_reversed_dir, file)


if __name__ == '__main__':
    input_path = '439360392_20231113_040517.ts'
    pre_processing = './output/pre_processing'
    temp_dir = './output/temp'
    divide_tmp = './output/divide_tmp'
    merge_tmp = './output/merge_tmp'
    reverse_dir = './output/reversed'
    audio_reversed_dir = './output/reversed_audio'
    dir_path = './test_dir'
    output_dir = './output'

    run(pre_processing, dir_path, output_dir, divide_tmp, merge_tmp, temp_dir, reverse_dir, audio_reversed_dir)

    # divide_2x2_with_progress(input_path, output_dir)
    # merge_tile_2x2(output_dir, f"{output_dir}/merged.mp4")
    # reverse_video(input_path, output_dir, temp_dir, reverse_dir)
    # reverse_audio(input_path, temp_dir)
    # concat_segments(filelist_path= f'{output_dir}/filelist.txt', output_path=f'{output_dir}/final_reversed.mp4')
    # concatenate_segments(reverse_dir, output_path=f"{output_dir}")
    # combine_audio_video(video_path=f"{output_dir}/reversed.mp4", audio_path=f'{temp_dir}/reversed_audio.mp4', output_path=f'{output_dir}/final_reversed.mp4')
