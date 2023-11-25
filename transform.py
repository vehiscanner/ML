import cv2

video_capture = cv2.VideoCapture("1.mp4")

frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

total_duration = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) / frame_rate

num_segments = 6

segment_duration = total_duration / num_segments


segment_number = 1
frame_count = 0
current_segment = None

# Buat direktori untuk menyimpan segmen-segmen video
output_path = "C:\\data train\\hasil transform\\segment_"

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_count == 0:
        segment_output_path = f"{output_path}{segment_number}.mp4"
        current_segment = cv2.VideoWriter(segment_output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    current_segment.write(frame)
    frame_count += 1

    if frame_count >= frame_rate * segment_duration:
        current_segment.release()
        frame_count = 0
        segment_number += 1

    if segment_number > num_segments:
        break

# Tutup video terakhir
if current_segment is not None:
    current_segment.release()

# Tutup video asli
video_capture.release()
cv2.destroyAllWindows()
