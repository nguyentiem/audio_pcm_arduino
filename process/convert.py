# import wave
# import numpy as np
# from scipy.signal import resample

# def wav_to_8bit_8khz_header(wav_file_path, header_file_path, channel=0, array_name="hello"):
#     """
#     Đọc file WAV, lấy dữ liệu một kênh, resample xuống 8000Hz, chuyển đổi thành 8-bit và xuất ra file header C.
#     """
#     with wave.open(wav_file_path, 'rb') as wav_file:
#         # Lấy thông tin file WAV
#         num_channels = wav_file.getnchannels()
#         sample_width = wav_file.getsampwidth()
#         frame_rate = wav_file.getframerate()
#         num_frames = wav_file.getnframes()

#         if channel >= num_channels:
#             raise ValueError(f"Kênh không hợp lệ. File có {num_channels} kênh.")

#         # Đọc toàn bộ dữ liệu âm thanh
#         frames = wav_file.readframes(num_frames)
#         dtype = np.int16 if sample_width == 2 else np.uint8
#         audio_data = np.frombuffer(frames, dtype=dtype)

#         # Tách dữ liệu của kênh cụ thể
#         audio_data = audio_data[channel::num_channels]

#         # Resample xuống 8000 Hz
#         new_num_samples = int(len(audio_data) * 8000 / frame_rate)
#         audio_data_resampled = resample(audio_data, new_num_samples)

#         # Chuyển đổi sang 8-bit (scale từ -32768..32767 thành 0..255)
#         audio_data_8bit = ((audio_data_resampled - np.min(audio_data_resampled)) / 
#                            (np.max(audio_data_resampled) - np.min(audio_data_resampled)) * 255).astype(np.uint8)

#     # Ghi dữ liệu vào file header
#     with open(header_file_path, "w") as header_file:
#         # Viết phần đầu của file header
#         header_file.write(f"#ifndef {array_name.upper()}_H\n")
#         header_file.write(f"#define {array_name.upper()}_H\n\n")
#         header_file.write(f"const uint8_t {array_name}[] = {{\n")

#         # Ghi dữ liệu mảng byte thành các hàng
#         for i, byte in enumerate(audio_data_8bit):
#             if i % 12 == 0:  # Mỗi hàng chứa 12 byte
#                 header_file.write("\n    ")
#             header_file.write(f"0x{byte:02X}, ")

#         # Kết thúc mảng và file header
#         header_file.write("\n};\n\n")
#         header_file.write(f"#endif // {array_name.upper()}_H\n")

# # Đường dẫn file WAV và file header
# wav_file = "hello.wav"   # File WAV stereo ban đầu
# header_file = "hello.h"    # File header đầu ra
# channel = 0                # Chọn kênh 0 (trái) hoặc 1 (phải)

# # Chuyển đổi WAV sang header C với tần số 8000 Hz và mẫu 8-bit
# wav_to_8bit_8khz_header(wav_file, header_file, channel=channel)
# print(f"Header file đã được tạo: {header_file}")
import wave
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

def wav_to_progmem_header(wav_file_path, header_file_path, channel=0, array_name="hello"):
    """
    Chuyển đổi dữ liệu WAV thành định dạng PROGMEM để lưu trên flash Arduino.
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        # Lấy thông tin file WAV
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        if channel >= num_channels:
            raise ValueError(f"Kênh không hợp lệ. File có {num_channels} kênh.")

        # Đọc toàn bộ dữ liệu âm thanh
        frames = wav_file.readframes(num_frames)
        dtype = np.int16 if sample_width == 2 else np.uint8
        audio_data = np.frombuffer(frames, dtype=dtype)

        # Tách dữ liệu của kênh cụ thể
        audio_data = audio_data[channel::num_channels]

        # Resample xuống 8000 Hz
        new_num_samples = int(len(audio_data) * 8000 / frame_rate)
        audio_data_resampled = resample(audio_data, new_num_samples)

        # Chuyển đổi sang 8-bit (scale từ -32768..32767 thành 0..255)
        audio_data_8bit = ((audio_data_resampled - np.min(audio_data_resampled)) / 
                           (np.max(audio_data_resampled) - np.min(audio_data_resampled)) * 255).astype(np.uint8)


    # Tạo file header với PROGMEM
    with open(header_file_path, "w") as header_file:
        header_file.write(f"#include <avr/pgmspace.h>\n\n")
        header_file.write(f"const uint8_t {array_name}[] PROGMEM = {{\n")

        for i, byte in enumerate(audio_data_8bit):
            if i % 12 == 0:  # Mỗi hàng 12 byte
                header_file.write("\n    ")
            header_file.write(f"0x{byte:02X}, ")

        header_file.write("\n};\n")

# Chuyển đổi WAV sang PROGMEM
# wav_file = "hello.wav"
# header_file = "hello.h"
# wav_to_progmem_header(wav_file, header_file)
# print(f"File PROGMEM đã được tạo: {header_file}")

import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
import numpy as np

def plot_wav_waveforms(wav_file_path, header_file_path, output_channel=0, target_sample_rate=8000, array_name="hello"):
    """
    Đọc tín hiệu WAV, xử lý tín hiệu theo yêu cầu, và vẽ đồ thị đường sóng của tín hiệu gốc và tín hiệu đã cắt.
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        # Lấy thông tin file WAV
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        print(f"Số kênh: {num_channels}, Độ rộng mẫu: {sample_width * 8} bit, Tần số lấy mẫu: {frame_rate} Hz")

        # Đọc toàn bộ dữ liệu
        frames = wav_file.readframes(num_frames)
        dtype = np.int16 if sample_width == 2 else np.uint8
        audio_data = np.frombuffer(frames, dtype=dtype)

        # Tách dữ liệu của kênh cụ thể
        audio_data_channel = audio_data[output_channel::num_channels]

        # Resample tín hiệu xuống 8000 Hz
        new_num_samples = int(len(audio_data_channel) * target_sample_rate / frame_rate)
        audio_data_resampled = resample(audio_data_channel, new_num_samples)

        # Chuẩn hóa dữ liệu về phạm vi 8-bit (0 - 255)
        audio_data_8bit = ((audio_data_resampled - np.min(audio_data_resampled)) / 
                           (np.max(audio_data_resampled) - np.min(audio_data_resampled)) * 255).astype(np.uint8)

    # Vẽ đồ thị đường sóng
    time_original = np.linspace(0, len(audio_data_channel) / frame_rate, num=len(audio_data_channel))
    time_resampled = np.linspace(0, len(audio_data_8bit) / target_sample_rate, num=len(audio_data_8bit))
    with open(header_file_path, "w") as header_file:
        header_file.write(f"#include <avr/pgmspace.h>\n\n")
        header_file.write(f"const uint8_t {array_name}[] PROGMEM = {{\n")

        for i, byte in enumerate(audio_data_8bit):
            if i % 12 == 0:  # Mỗi hàng 12 byte
                header_file.write("\n    ")
            header_file.write(f"0x{byte:02X}, ")

        header_file.write("\n};\n")
    
    with wave.open("new.wav", 'wb') as out_wav:
        out_wav.setnchannels(1)  # Mono
        out_wav.setsampwidth(1)  # 8-bit
        out_wav.setframerate(8000)  # 8000 Hz
        out_wav.writeframes(audio_data_8bit.tobytes())

    plt.figure(figsize=(12, 6))

    # Đồ thị tín hiệu gốc
    plt.subplot(2, 1, 1)
    plt.plot(time_original, audio_data_channel, label="Tín hiệu gốc", color="blue")
    plt.title("Đường sóng tín hiệu gốc")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Biên độ")
    plt.legend()

    # Đồ thị tín hiệu sau xử lý
    plt.subplot(2, 1, 2)
    plt.plot(time_resampled, audio_data_8bit, label="Tín hiệu sau xử lý", color="green")
    plt.title("Đường sóng tín hiệu sau khi cắt bỏ (1 kênh, 8000Hz, 8-bit)")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Biên độ (8-bit)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def wav_to_header(wav_file_path, header_file_path, array_name="wav_data"):
    """
    Chuyển đổi file WAV thành mảng byte và lưu trong file header (.h) sử dụng PROGMEM.
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        # Lấy thông tin file WAV
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        print(f"Thông tin file WAV:")
        print(f"  - Số kênh: {num_channels}")
        print(f"  - Độ rộng mẫu: {sample_width * 8} bit")
        print(f"  - Tần số lấy mẫu: {frame_rate} Hz")
        print(f"  - Số mẫu: {num_frames}")

        # Đọc toàn bộ dữ liệu WAV
        frames = wav_file.readframes(num_frames)
        dtype = np.int16 if sample_width == 2 else np.uint8
        audio_data = np.frombuffer(frames, dtype=dtype)

        # Nếu là stereo, lấy chỉ 1 kênh
        if num_channels > 1:
            audio_data = audio_data[::num_channels]

        # Chuyển đổi tín hiệu 16-bit về 8-bit nếu cần
        if sample_width == 2:
            audio_data = ((audio_data - np.min(audio_data)) / 
                          (np.max(audio_data) - np.min(audio_data)) * 255).astype(np.uint8)

    # Ghi dữ liệu thành mảng byte vào file .h
    with open(header_file_path, "w") as header_file:
        header_file.write(f"#include <avr/pgmspace.h>\n\n")
        header_file.write(f"const uint8_t {array_name}[] PROGMEM = {{\n")

        for i, byte in enumerate(audio_data):
            if i % 12 == 0:  # Mỗi dòng 12 byte
                header_file.write("\n    ")
            header_file.write(f"0x{byte:02X}, ")

        header_file.write("\n};\n")

    print(f"Đã tạo file header: {header_file_path}")


# Sử dụng hàm để tạo file .h
wav_file = "hello.wav"  # Đường dẫn file WAV
header_file = "hello.h"  # Đường dẫn file header output
plot_wav_waveforms(wav_file, header_file, array_name="hello")