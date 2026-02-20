"""
Event-based simulator with compatibility fixes for numpy type deprecations.
Enhanced to process both .avi and .mp4 video files.
"""

from __future__ import absolute_import
import os
import time
import cv2
import argparse
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add compatibility layer for numpy deprecated types
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# Monkey patch scikit-video to use built-in int
import sys
import skvideo.io

skvideo.io.np.int = int

from metavision_core_ml.video_to_event.simulator import EventSimulator
from metavision_core.event_io.event_bufferizer import FixedCountBuffer
from metavision_core.event_io import DatWriter
from metavision_core_ml.preprocessing import viz_events
from metavision_core_ml.data.video_stream import TimedVideoStream
from metavision_core_ml.data.image_planar_motion_stream import PlanarMotionStream

# Supported video file extensions
SUPPORTED_VIDEO_EXTENSIONS = ['.avi', '.mp4']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def create_custom_event_visualization(events, width, height):
    """
    Create custom event visualization with ON events as green and OFF events as red.
    """
    # Create black background image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    if len(events) == 0:
        return image

    # Get event coordinates
    x_coords = events['x']
    y_coords = events['y']
    polarities = events['p']

    # Ensure coordinates are within bounds
    valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    polarities = polarities[valid_mask]

    # ON events (polarity = 1) -> Green (0, 255, 0)
    # OFF events (polarity = 0) -> Red (0, 0, 255) in BGR format
    on_mask = polarities == 1
    off_mask = polarities == 0

    if np.any(on_mask):
        image[y_coords[on_mask], x_coords[on_mask]] = [0, 255, 0]  # Green for ON events

    if np.any(off_mask):
        image[y_coords[off_mask], x_coords[off_mask]] = [0, 0, 255]  # Red for OFF events

    return image


def create_event_simulator(height, width, Cp, Cn, refractory_period, **kwargs):
    """Wrapper function to create EventSimulator with proper type conversion."""
    try:
        # Convert parameters to float
        Cp = float(Cp)
        Cn = float(Cn)
        refractory_period = float(refractory_period)
        cutoff_hz = float(kwargs.get('cutoff_hz', 0))
        sigma_threshold = float(kwargs.get('sigma_threshold', 0.001))
        shot_noise_rate_hz = float(kwargs.get('shot_noise_rate_hz', 10))

        # Convert dimensions to int
        height = int(height)
        width = int(width)

        return EventSimulator(
            height, width,
            Cp, Cn,
            refractory_period,
            cutoff_hz=cutoff_hz,
            sigma_threshold=sigma_threshold,
            shot_noise_rate_hz=shot_noise_rate_hz
        )
    except Exception as e:
        print(f"Error creating simulator: {str(e)}")
        raise


def is_supported_video_file(filepath):
    """Check if the file has a supported video extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_image_file(filepath):
    """Check if the file has a supported image extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run event based simulator on video/image files or directories (supports .avi and .mp4)')

    # Directory paths are now specified in the code, not as command line arguments
    parser.add_argument('--n_events', type=int, default=15000)
    parser.add_argument('--height_width', nargs=2, default=None, type=int)
    parser.add_argument('--crop_image', action="store_true")
    parser.add_argument("--display", action="store_true",
                        help="Enable frame display during processing (disabled by default)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('-o', "--output")
    parser.add_argument('-fps', '--override_fps', default=0, type=float)
    parser.add_argument('--save_frames', action="store_true",
                        help='Save RGB images, event images, and events as .npy files')
    parser.add_argument('--save_dir', default='saved_frames',
                        help='Directory to save frames and events')

    # Simulator parameters
    sim_group = parser.add_argument_group('Simulator parameters')
    sim_group.add_argument("--Cp", default=0.35, type=float)
    sim_group.add_argument("--Cn", default=0.35, type=float)
    sim_group.add_argument("--refractory_period", default=1, type=float)
    sim_group.add_argument("--sigma_threshold", default=0.1, type=float)
    sim_group.add_argument("--cutoff_hz", default=0, type=float)
    sim_group.add_argument("--leak_rate_hz", default=0, type=float)
    sim_group.add_argument("--shot_noise_rate_hz", default=10, type=float)

    args = parser.parse_args()
    return args


def init_video_stream(path, height, width, override_fps):
    """Initialize video stream with error handling for both AVI and MP4 files."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()

        # Handle image files
        if is_supported_image_file(path):
            return PlanarMotionStream(path, height, width, crop_image=False)

        # Handle video files (.avi and .mp4)
        if is_supported_video_file(path):
            # Try to open video file
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {path}")

            # Get video properties using cv2
            width = int(width) if width > 0 else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(height) if height > 0 else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(override_fps if override_fps > 0 else cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            print(f"Video info - Width: {width}, Height: {height}, FPS: {fps}, Format: {ext.upper()}")
            return TimedVideoStream(path, height, width, override_fps=fps)

        else:
            supported_formats = SUPPORTED_VIDEO_EXTENSIONS + SUPPORTED_IMAGE_EXTENSIONS
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {supported_formats}")

    except Exception as e:
        print(f"Error initializing video stream: {str(e)}")
        raise


def get_video_files_from_paths(paths):
    """Get all supported video files (.avi and .mp4) from the given paths, including subdirectories."""
    all_video_files = []
    path_info = {}

    for path in paths:
        video_files = []
        if os.path.isdir(path):
            print(f"Scanning directory recursively: {path}")
            # Walk through all subdirectories
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if is_supported_video_file(filename):
                        full_path = os.path.join(root, filename)
                        video_files.append(full_path)

            video_files.sort()  # Sort for consistent processing order
            path_info[path] = {'type': 'directory', 'files': video_files}

            # Show subfolder breakdown with file type counts
            subfolders = {}
            for video_file in video_files:
                relative_path = os.path.relpath(os.path.dirname(video_file), path)
                if relative_path not in subfolders:
                    subfolders[relative_path] = {'avi': [], 'mp4': []}

                ext = os.path.splitext(video_file)[1].lower()
                filename = os.path.basename(video_file)
                if ext == '.avi':
                    subfolders[relative_path]['avi'].append(filename)
                elif ext == '.mp4':
                    subfolders[relative_path]['mp4'].append(filename)

            if subfolders:
                print(f"  Found video files in {len(subfolders)} subfolder(s):")
                for subfolder, file_types in subfolders.items():
                    total_files = len(file_types['avi']) + len(file_types['mp4'])
                    if subfolder == '.':
                        print(
                            f"    📁 Root directory: {total_files} files (.avi: {len(file_types['avi'])}, .mp4: {len(file_types['mp4'])})")
                    else:
                        print(
                            f"    📁 {subfolder}: {total_files} files (.avi: {len(file_types['avi'])}, .mp4: {len(file_types['mp4'])})")

        elif os.path.isfile(path) and is_supported_video_file(path):
            video_files.append(path)
            path_info[path] = {'type': 'file', 'files': video_files}
        elif os.path.exists(path):
            print(f"Warning: {path} exists but contains no supported video files (.avi or .mp4)")
            path_info[path] = {'type': 'empty', 'files': []}
        else:
            print(f"Warning: Path {path} does not exist")
            path_info[path] = {'type': 'not_found', 'files': []}

        all_video_files.extend(video_files)

    return all_video_files, path_info


def get_video_files(directory_path):
    """Get all supported video files (.avi and .mp4) from the given directory."""
    video_files = []
    if os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            if is_supported_video_file(filename):
                video_files.append(os.path.join(directory_path, filename))
        video_files.sort()  # Sort for consistent processing order
    elif os.path.isfile(directory_path) and is_supported_video_file(directory_path):
        video_files.append(directory_path)

    return video_files


def process_single_video(video_path, args, base_directory=None):
    """Process a single video file (supports both .avi and .mp4)."""
    file_ext = os.path.splitext(video_path)[1].upper()
    print(f"\n{'=' * 60}")
    print(f"Processing {file_ext} video: {video_path}")
    print(f"{'=' * 60}")

    # Create output directory structure that mirrors the source structure
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if base_directory and os.path.commonpath([video_path, base_directory]) == base_directory:
        # Preserve folder structure relative to base directory
        relative_path = os.path.relpath(os.path.dirname(video_path), base_directory)
        if relative_path == '.':
            # Video is in the root of base directory
            video_output_dir = os.path.join('output_data', video_name)
        else:
            # Video is in a subdirectory
            folder_name = relative_path.replace(os.sep, '_')  # Replace path separators with underscores
            video_output_dir = os.path.join('output_data', folder_name, video_name)
    else:
        # Fallback: use just the video name
        video_output_dir = os.path.join('output_data', video_name)

    try:
        # Initialize dimensions
        height, width = args.height_width if args.height_width else (-1, -1)

        # Initialize video stream
        image_stream = init_video_stream(video_path, height, width, args.override_fps)

        if args.height_width is None:
            height, width = image_stream.get_size()
            height, width = int(height), int(width)

        print(f"Final dimensions: {width}x{height}")

        # Initialize components
        fixed_buffer = FixedCountBuffer(int(args.n_events))
        simu = create_event_simulator(
            height, width,
            args.Cp, args.Cn,
            args.refractory_period,
            cutoff_hz=args.cutoff_hz,
            sigma_threshold=args.sigma_threshold,
            shot_noise_rate_hz=args.shot_noise_rate_hz
        )

        # Initialize OpenCV video capture for original RGB frames
        original_cap = cv2.VideoCapture(video_path)
        if not original_cap.isOpened():
            print(f"Warning: Could not open {file_ext} video for original RGB display")
            original_cap = None
        else:
            print(f"Successfully opened {file_ext} video for RGB frame extraction")

        # Create save directories
        rgb_dir = os.path.join(video_output_dir, 'rgb_frames')
        events_viz_dir = os.path.join(video_output_dir, 'events_visualization')
        events_data_dir = os.path.join(video_output_dir, 'events_data')

        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(events_viz_dir, exist_ok=True)
        os.makedirs(events_data_dir, exist_ok=True)

        print(f"Saving frames and events to: {video_output_dir}")
        print(f"  - RGB frames: {rgb_dir}")
        print(f"  - Event visualizations (ON=Green, OFF=Red): {events_viz_dir}")
        print(f"  - Event data: {events_data_dir}")
        print(f"Display mode: {'ENABLED' if args.display else 'DISABLED'}")

        if args.display:
            cv2.namedWindow('events', cv2.WINDOW_NORMAL)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.namedWindow('original_rgb', cv2.WINDOW_NORMAL)

        # Set up output writer if specified
        output_file = None
        if args.output:
            output_file = os.path.join(video_output_dir, f"{video_name}.dat")
        writer = DatWriter(output_file, height=height, width=width) if output_file else None

        start = time.time()
        frame_count = 0
        for img, ts in image_stream:
            # Read original RGB frame
            original_rgb = None
            if original_cap is not None:
                ret, original_rgb = original_cap.read()
                if not ret:
                    original_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video if needed
                    ret, original_rgb = original_cap.read()

            total = simu.image_callback(img, ts)
            if total < args.n_events:
                continue

            events = simu.get_events()
            simu.flush_events()
            events = fixed_buffer(events)
            if not len(events):
                continue

            end = time.time()
            dt = float(events['t'][-1] - events['t'][0])

            # Use custom visualization function instead of viz_events
            image_rgb = create_custom_event_visualization(events, width=width, height=height)

            # Save frames and events
            frame_filename_base = f"frame_{frame_count:06d}"

            # Save original RGB image as PNG
            if original_rgb is not None:
                rgb_filename = f"{frame_filename_base}.png"
                cv2.imwrite(os.path.join(rgb_dir, rgb_filename), original_rgb)

            # Save event visualization image as PNG (already in BGR format)
            events_viz_filename = f"{frame_filename_base}.png"
            cv2.imwrite(os.path.join(events_viz_dir, events_viz_filename), image_rgb)

            # Save raw events data as NPY (only this stays as .npy)
            events_data_filename = f"{frame_filename_base}.npy"
            np.save(os.path.join(events_data_dir, events_data_filename), events)

            if args.verbose:
                print(f"Saved frame {frame_count} data to {video_output_dir}")

            if args.verbose:
                num_evs = len(events)
                max_evs = np.unique(
                    events["x"] * height + events['y'],
                    return_counts=True
                )[1].max()
                print(
                    f"Runtime: {(end - start) * 1000:.5f} ms, "
                    f"Max ev/pixel: {max_evs}, "
                    f"Total Mev: {num_evs * 1e-6:.5f}, "
                    f"dt: {dt} us"
                )

            if args.display:
                cv2.imshow('events', image_rgb)
                cv2.imshow('image', img)
                if original_rgb is not None:
                    cv2.imshow('original_rgb', original_rgb)
                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC key
                    print("ESC pressed. Stopping current video processing...")
                    break

            if writer:
                writer.write(events)

            start = time.time()
            frame_count += 1

        print(f"Completed processing {file_ext} video: {video_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Output saved to: {video_output_dir}")

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        raise

    finally:
        if args.display:
            cv2.destroyAllWindows()
        if 'writer' in locals() and writer:
            writer.close()
        if 'original_cap' in locals() and original_cap:
            original_cap.release()


def main():
    """Main function to run the event simulator on multiple videos (.avi and .mp4) from multiple paths."""

    # ====================================================================
    # CONFIGURE YOUR PATHS HERE - Edit this list to specify your directories
    # ====================================================================
    DIRECTORY_PATHS = [
        "path/to/directory"
    ]
    # ====================================================================

    try:
        args = parse_args()

        print(f"Processing configured paths (recursive search for .avi and .mp4 files): {DIRECTORY_PATHS}")
        print(f"Supported video formats: {SUPPORTED_VIDEO_EXTENSIONS}")
        print(f"Frame display: {'ENABLED (use --display flag to enable)' if not args.display else 'DISABLED (default)'}")

        # Get list of video files to process from all paths (including subdirectories)
        all_video_files, path_info = get_video_files_from_paths(DIRECTORY_PATHS)

        if not all_video_files:
            print(
                f"No supported video files ({SUPPORTED_VIDEO_EXTENSIONS}) found in any of the specified paths or their subdirectories.")
            print("\nPath summary:")
            for path, info in path_info.items():
                if info['type'] == 'directory':
                    print(f"  📁 {path}: {len(info['files'])} video files (including subdirectories)")
                elif info['type'] == 'file':
                    print(f"  📄 {path}: Single video file")
                elif info['type'] == 'empty':
                    print(f"  ❌ {path}: No supported video files found")
                else:
                    print(f"  ❌ {path}: Path not found")
            return

        # Display summary of found files organized by source folders and file types
        print(f"\n{'=' * 60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total video files found: {len(all_video_files)}")

        # Count file types
        avi_count = sum(1 for f in all_video_files if f.lower().endswith('.avi'))
        mp4_count = sum(1 for f in all_video_files if f.lower().endswith('.mp4'))
        print(f"  - AVI files: {avi_count}")
        print(f"  - MP4 files: {mp4_count}")

        # Group files by their immediate parent directory for better organization
        folder_groups = {}
        for video_path in all_video_files:
            parent_dir = os.path.dirname(video_path)
            if parent_dir not in folder_groups:
                folder_groups[parent_dir] = {'avi': [], 'mp4': []}

            filename = os.path.basename(video_path)
            if video_path.lower().endswith('.avi'):
                folder_groups[parent_dir]['avi'].append(filename)
            elif video_path.lower().endswith('.mp4'):
                folder_groups[parent_dir]['mp4'].append(filename)

        print(f"\nFiles organized by folder ({len(folder_groups)} folders):")
        for folder_path, file_types in sorted(folder_groups.items()):
            folder_name = os.path.basename(folder_path) if folder_path else "Root"
            total_files = len(file_types['avi']) + len(file_types['mp4'])
            print(
                f"  📁 {folder_name} ({folder_path}): {total_files} files (.avi: {len(file_types['avi'])}, .mp4: {len(file_types['mp4'])})")

            # List AVI files
            for file_name in sorted(file_types['avi']):
                print(f"    - {file_name} [AVI]")

            # List MP4 files
            for file_name in sorted(file_types['mp4']):
                print(f"    - {file_name} [MP4]")

        print(f"\n{'=' * 60}")

        # Process each video file
        total_videos = len(all_video_files)
        successful_processes = 0
        failed_processes = 0

        # Determine base directory for maintaining folder structure
        base_directory = DIRECTORY_PATHS[0] if len(DIRECTORY_PATHS) == 1 and os.path.isdir(DIRECTORY_PATHS[0]) else None

        for i, video_path in enumerate(all_video_files, 1):
            source_dir = os.path.dirname(video_path)
            video_name = os.path.basename(video_path)
            folder_name = os.path.basename(source_dir)
            file_ext = os.path.splitext(video_path)[1].upper()

            print(f"\n>>> Processing video {i}/{total_videos}")
            print(f"    Source folder: {folder_name}")
            print(f"    Source path: {source_dir}")
            print(f"    File: {video_name} [{file_ext}]")

            try:
                process_single_video(video_path, args, base_directory)
                print(f"✓ Successfully processed {video_name} [{file_ext}] from {folder_name}")
                successful_processes += 1
            except Exception as e:
                print(f"✗ Failed to process {video_name} [{file_ext}] from {folder_name}: {str(e)}")
                failed_processes += 1
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        # Final summary
        print(f"\n{'=' * 60}")
        print(f"BATCH PROCESSING COMPLETED!")
        print(f"{'=' * 60}")
        print(f"Total videos: {total_videos}")
        print(f"  - AVI files processed: {avi_count}")
        print(f"  - MP4 files processed: {mp4_count}")
        print(f"✓ Successful: {successful_processes}")
        print(f"✗ Failed: {failed_processes}")
        print(f"Output data saved in 'output_data' directory")
        print(f"Folder structure preserved: each source folder has its own output subdirectory")
        print(f"Each video has its own subdirectory with:")
        print(f"  - RGB frames (original video frames)")
        print(f"  - Event visualizations (ON=Green, OFF=Red)")
        print(f"  - Event data (.npy files)")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        if 'args' in locals() and args.verbose:
            import traceback
            traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise
