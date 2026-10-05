'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

_last_white_x = None
_last_lane = LANE_UNKNOWN
_gap_count = 0
_last_center_x = None

def detect_lane(frame):
    global _last_white_x, _last_lane, _gap_count, _last_center_x

    if frame is None:
        return {"center_x": -1, "lane": LANE_UNKNOWN}

    h, w = frame.shape[:2]
    if h < 100 or w < 100:
        return {"center_x": -1, "lane": LANE_UNKNOWN}

    vehicle_x = w / 2.0
    y_eval = int(0.75 * h)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yellow mask
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 60, 60], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8)
    )
    
    # White mask
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 150], dtype=np.uint8),
        np.array([180, 80, 255], dtype=np.uint8)
    )
    
    y0 = int(0.35 * h)
    y1 = int(0.95 * h)
    yellow_mask[:y0] = 0; yellow_mask[y1:] = 0
    white_mask[:y0] = 0; white_mask[y1:] = 0
    
    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    top_m = 50.0
    num_w, labels_w, stats_w, centroids_w = cv2.connectedComponentsWithStats(white_mask, 8)
    dash_pts = []
    for i in range(1, num_w):
        x, y, ww, hh, area = stats_w[i]
        cx, cy = centroids_w[i]
        t = max(0.0, min(1.0, (cy - y0) / float(y1 - y0)))
        cur_min_x = (1.0 - t) * top_m
        cur_max_x = w - cur_min_x
        if 15 <= area <= 25000 and ww < int(0.7*w) and hh < int(0.7*h):
            if cur_min_x <= cx <= cur_max_x:
                dash_pts.append((float(cx), float(cy), area))
            
    # Yellow boundaries
    num_y, labels_y, stats_y, centroids_y = cv2.connectedComponentsWithStats(yellow_mask, 8)
    yellow_left_x = None
    yellow_right_x = None
    
    for i in range(1, num_y):
        x, y, ww, hh, area = stats_y[i]
        if area > 30:
            cx, cy = centroids_y[i]
            sample_y = max(y0, min(y1 - 1, y_eval))
            if y <= sample_y <= y + hh:
                row_xs = np.where(labels_y[sample_y] == i)[0]
                if len(row_xs) > 0:
                    if cx < vehicle_x:
                        yl = float(np.max(row_xs))
                        if yellow_left_x is None or yl > yellow_left_x:
                            yellow_left_x = yl
                    else:
                        yr = float(np.min(row_xs))
                        if yellow_right_x is None or yr < yellow_right_x:
                            yellow_right_x = yr

    divider_x = None
    white_side = None
    
    if len(dash_pts) >= 1:
        dash_pts.sort(key=lambda p: p[1])
        left_dashes = [p for p in dash_pts if p[0] < vehicle_x]
        right_dashes = [p for p in dash_pts if p[0] >= vehicle_x]
        
        candidates = []
        for g, side in [(right_dashes, "right"), (left_dashes, "left"), (dash_pts, "all")]:
            if len(g) >= 1:
                g_ys = np.array([p[1] for p in g], dtype=np.float32)
                g_xs = np.array([p[0] for p in g], dtype=np.float32)
                if len(g) >= 2:
                    fit = np.polyfit(g_ys, g_xs, 1)
                    y_max = np.max(g_ys)
                    y_ref = min(float(y_eval), float(y_max))
                    x_at = fit[0] * y_ref + fit[1]
                    res = np.abs(g_xs - (fit[0] * g_ys + fit[1]))
                    err = float(np.mean(res))
                else:
                    x_at = float(g_xs[0])
                    err = 0.0
                if 0 <= x_at <= w:
                    tot_area = float(np.sum([p[2] for p in g]))
                    candidates.append((err, x_at, side, len(g), tot_area))
                    
        if candidates:
            candidates.sort(key=lambda c: (-c[4], -c[3], c[0]))
            err, x_at, side, count, tot_area = candidates[0]
            divider_x = x_at
            white_side = "right" if divider_x > vehicle_x else "left"

    half_lane = 120
    
    if divider_x is not None:
        if _last_white_x is not None:
            divider_x = 0.85 * divider_x + 0.15 * _last_white_x
        _last_white_x = divider_x
        _gap_count = 0
        if white_side == "right":
            lane = LANE_LEFT
            center_x = divider_x - 122
        else:
            lane = LANE_RIGHT
            center_x = divider_x + 101
            if yellow_right_x is not None and yellow_right_x > divider_x + 50:
                center_x = 0.50 * center_x + 0.50 * (yellow_right_x - 170)
        _last_lane = lane
    elif _last_white_x is not None and _gap_count < 12:
        _gap_count += 1
        divider_x = _last_white_x
        lane = _last_lane
        if lane == LANE_LEFT:
            center_x = divider_x - 122
        else:
            center_x = divider_x + 101
            if yellow_right_x is not None and yellow_right_x > divider_x + 50:
                center_x = 0.50 * center_x + 0.50 * (yellow_right_x - 170)
    else:
        # Yellow fallback
        if yellow_left_x is not None and yellow_right_x is not None:
            road_mid = (yellow_left_x + yellow_right_x) / 2.0
            lane = LANE_LEFT if vehicle_x < road_mid else LANE_RIGHT
            center_x = (yellow_left_x + road_mid) / 2.0 if lane == LANE_LEFT else (road_mid + yellow_right_x) / 2.0
        elif yellow_left_x is not None:
            lane = LANE_LEFT
            center_x = yellow_left_x + half_lane
        elif yellow_right_x is not None:
            lane = LANE_RIGHT
            center_x = yellow_right_x - half_lane
        else:
            lane = _last_lane if _last_lane != LANE_UNKNOWN else LANE_UNKNOWN
            center_x = _last_center_x if _last_center_x is not None else vehicle_x

    center_x = int(round(max(0, min(w - 1, center_x))))
    if _last_center_x is not None and lane == _last_lane:
        center_x = int(round(0.80 * center_x + 0.20 * _last_center_x))
    _last_center_x = center_x
    
    return {"center_x": center_x, "lane": lane}

##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
