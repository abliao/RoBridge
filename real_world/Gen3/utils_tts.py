print('导入语音合成模块')

import os
import wave

import pyaudio
from aip import AipSpeech  # 导入api接口
from playsound import playsound  # 音频模块
import pygame
from API_KEY import *
APP_ID = '116199789'
API_KEY = 'h9NNpjBSkij1vRACbVnDb5s7'
SECRET_KEY = 'B5JLHtQD2yFpgsTDtghXzZksOVlWa3RX'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def tts(TEXT='我是你的语音助手', tts_wav_path = 'E:/code/agent_demo/temp/tts.mp3'):
    '''
    语音合成TTS，生成wav音频文件
    'per': 4  发声人选择，0为女声,1为男声,3为情感合成-度逍遥,4为情感合成-度丫丫，默认为普通女
    '''
    result = client.synthesis(TEXT, 'zh', 1, {
    'per': 1,
    'spd': 5,    # 速度
    'vol': 7   # 音量
    })
    if not isinstance(result, dict):
        with open(tts_wav_path, 'wb') as f:
            f.write(result)

def convert_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="wav")
    audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    print("✅ 音频转换完成:", output_path)

def play_audio(file_path):
    convert_wav(input_path=file_path, output_path=file_path)
    # 初始化pygame的混音器
    pygame.mixer.init()
    # 加载音频文件
    pygame.mixer.music.load(file_path)
    # 播放音频
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # 等待播放结束
        pygame.time.Clock().tick(10)

def play_wav(wav_path = 'temp/tts.wav'):
    '''
    播放wav音频文件
    '''

    playsound(wav_path)

# from pydub import AudioSegment

# # 指定 ffmpeg 路径（如果没加环境变量）
# AudioSegment.converter = r"E:/ffmpeg/ffmpeg-2025-02-06-git-6da82b4485-essentials_build/bin/ffmpeg.exe"
# AudioSegment.ffprobe = r"E:/ffmpeg/ffmpeg-2025-02-06-git-6da82b4485-essentials_build/bin/ffprobe.exe"
# # 读取原始文件并转换格式
# audio = AudioSegment.from_file("E:\\code\\agent_demo\\temp\\tts.wav", format="wav")
# audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # PCM 16-bit
# audio.export("E:/code/agent_demo/temp/tts_fixed.wav", format="wav")
# playsound('E:/code/agent_demo/temp/tts.mp3')
# def play_wav(wav_file='temp/tts.wav'):
#     '''
#     播放wav文件
#     '''
#     wf = wave.open(wav_file, 'rb')
 
#     # 实例化PyAudio
#     p = pyaudio.PyAudio()
 
#     # 打开流
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)

#     chunk_size = 1024
#     # 读取数据
#     data = wf.readframes(chunk_size)
 
#     # 播放音频
#     while data != b'':
#         stream.write(data)
#         data = wf.readframes(chunk_size)
 
#     # 停止流，关闭流和PyAudio
#     stream.stop_stream()
#     stream.close()
#     p.terminate()