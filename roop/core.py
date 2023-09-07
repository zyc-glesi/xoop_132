#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('-p', '--temp', help='select output_temp file or directory', dest='output_temp_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    #这一行代码解析命令行参数，并将解析后的参数值存储在 args 对象中。
    args = program.parse_args()
    #args = program.parse_args(['-s', 'source.jpg', '-t', 'target.jpg', '-o', 'output.mp4', '--frame-processor', 'face_swapper', '--keep-fps', '--keep-frames', '--skip-audio', '--many-faces', '--reference-face-position', '0', '--reference-frame-number', '0', '--similar-face-distance', '0.85', '--temp-frame-format', 'png', '--temp-frame-quality', '0', '--output-video-encoder', 'libx264', '--output-video-quality', '35', '--max-memory', '0', '--execution-provider', 'cpu', '--execution-threads', '0'])

    #余下的部分将解析后的参数值赋值给了全局变量 roop.globals 中的相应属性，以便在整个脚本中使用。
    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.output_temp_path = args.output_temp_path
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)


def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image 【图片换图片这段不用看】
    # 中文解释：如果目标文件是图片，那么就直接复制到输出路径，然后调用 get_frame_processors_modules 函数获取所有的帧处理器模块，然后遍历这些模块，调用 process_image 函数处理图片，最后调用 post_process 函数。
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        # validate image
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return

    # process image to videos  【视频切分成图片】 中文解释：如果目标文件是视频，那么就调用 predict_video 函数判断视频是否可以处理，如果可以处理，那么就调用 destroy 函数，然后调用 create_temp 函数创建临时文件夹，然后调用 extract_frames 函数提取视频帧，然后调用 get_temp_frame_paths 函数获取临时文件夹中的所有帧，然后调用 get_frame_processors_modules 函数获取所有的帧处理器模块，然后遍历这些模块，调用 process 函数处理帧，最后调用 post_process 函数。
    if predict_video(roop.globals.target_path):
        destroy()
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    # extract frames 中文解释：如果用户设置了 keep_fps 参数，那么就调用 detect_fps 函数获取视频的帧率，然后调用 extract_frames 函数提取视频帧，否则就调用 extract_frames 函数提取视频帧，帧率为 30。
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30 FPS...')
        extract_frames(roop.globals.target_path)

    # process frame 【图片批量替换成为人脸】中文解释：调用 get_temp_frame_paths 函数获取临时文件夹中的所有帧，然后调用 get_frame_processors_modules 函数获取所有的帧处理器模块，然后遍历这些模块，调用 process 函数处理帧，最后调用 post_process 函数。
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return

    # create video 【图片合成视频的代码】中文解释：如果用户设置了 keep_fps 参数，那么就调用 detect_fps 函数获取视频的帧率，然后调用 create_video 函数创建视频，否则就调用 create_video 函数创建视频，帧率为 30。
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(roop.globals.target_path)

    # handle audio 中文解释：如果用户设置了 skip_audio 参数，那么就调用 move_temp 函数将临时文件夹中的所有帧移动到输出路径，然后调用 update_status 函数更新状态，否则就调用 restore_audio 函数将音频复制到输出路径，然后调用 update_status 函数更新状态。
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    # clean temp 中文解释：调用 clean_temp 函数清理临时文件夹，然后调用 update_status 函数更新状态。
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)
    # validate video 中文解释：如果输出文件是视频，那么就调用 is_video 函数判断视频是否可以处理，如果可以处理，那么就调用 update_status 函数更新状态，否则就调用 update_status 函数更新状态。
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run() -> None:
    parse_args() #解析命令行参数
    if not pre_check(): #这是一个条件语句，检查是否通过了预检查（pre_check() 函数）。如果预检查失败，函数会提前返回，不执行后续操作。
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors): #这个循环遍历了配置的帧处理器模块列表，为每个模块执行以下操作
        if not frame_processor.pre_check():
            return
    limit_resources() #这是一个函数调用，用于限制脚本的资源使用，包括设置GPU内存限制和系统内存限制。
    if roop.globals.headless: #这是一个条件语句，检查是否启用了无头模式（headless mode）。如果启用了无头模式，函数会直接调用 start() 函数，不执行后续操作。
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()


#模仿 start 方法，写一个 start2 方法，用于批量图片对图片的换脸。
#仅仅需要完成 process frame 这个步骤
def start2() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return

    # process frame 【图片批量替换成为人脸】中文解释：调用 get_temp_frame_paths 函数获取临时文件夹中的所有帧，然后调用 get_frame_processors_modules 函数获取所有的帧处理器模块，然后遍历这些模块，调用 process 函数处理帧，最后调用 post_process 函数。
    temp_frame_paths = get_temp_frame_paths2(roop.globals.output_temp_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_frame(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        print(roop.globals.output_temp_path)
        return


def run2() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    start2()
