import os
import cv2

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_INSTALLED = True
except ImportError:
    MOVIEPY_INSTALLED = False


def extract_frames(video_path, output_dir, fps=1):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str): The path to the input video file.
        output_dir (str): The directory where the extracted frames will be saved.
        fps (int, optional): The desired frames per second for extraction. Defaults to 1.

    Raises:
        ImportError: If the moviepy library is not installed.
        FileNotFoundError: If the video file is not found at the specified path.
        Exception: For any other errors during video processing.
    """
    if not MOVIEPY_INSTALLED:
        raise ImportError("moviepy is not installed. Please install it to use video processing features.")

    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the video clip
        clip = VideoFileClip(video_path)
        duration = clip.duration

        frame_count = 0
        # Extract frames at specified intervals
        for t in range(0, int(duration), int(1 / fps) if fps != 0 else 1):
            frame = clip.get_frame(t)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame_bgr)
            frame_count += 1
            
        print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")
        clip.close()

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


# Example Usage:
# video_path = "path/to/your/video.mp4"  # Replace with the path to your video file
# output_dir = "path/to/output/frames"  # Replace with the desired output directory
# extract_frames(video_path, output_dir, fps=1)  # Extract frames at 1 frame per second