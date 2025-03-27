from utils_asr import *  # 录音+语音识别
from utils_tts import *  # 语音合成模块

print('开始录音5秒')
record(DURATION=5)   # 录音
print('播放录音')
# play_wav('temp/speech_record.wav')
speech_result = speech_recognition()
print('开始语音合成')
tts(speech_result)
print('播放语音合成音频')
play_mp3('temp/tts.wav')

