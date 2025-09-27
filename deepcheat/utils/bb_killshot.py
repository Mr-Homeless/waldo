import cv2
import numpy as np
import sys
import os
import csv


def _get_bb_kills(filename):
    cap = cv2.VideoCapture(filename)
    DEATH_FRAME_COOLDOWN = 0
    DEATH_KILLFEED_PIXEL_AREA = 4700
    death = False
    death_counter = 0

    last_fifteen = [0]

    last_mov_med = 0
    frames_since_last_kill = 0
    total = 0
    cur_frame_count = 0
    kill_detection_frames = []
    while True:
        cur_frame_count+=1
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame = frame[0:height // 3, int(width / 1.3):width]
        red = cv2.inRange(frame, (0, 0, 100), (45, 45, 255))
        #cv2.imshow('red', red)
        contours, _ = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        total_contour_area = 0
        for contour in contours:
            total_contour_area += cv2.contourArea(contour)
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        for contour in contours:
            contour = cv2.convexHull(contour)
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

        #cv2.imshow('mask', mask)
        final_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.imshow('frame', frame)

        if death:
            death_counter += 1
        if death_counter > DEATH_FRAME_COOLDOWN and DEATH_FRAME_COOLDOWN!=0:
            print('Resuming detection')
            death = False
            death_counter = 0

        if not death:
            # Detect death
            red_pixels = np.count_nonzero(red)
            masked_pixels = np.count_nonzero(mask)

            if red_pixels > DEATH_KILLFEED_PIXEL_AREA and masked_pixels > DEATH_KILLFEED_PIXEL_AREA and DEATH_FRAME_COOLDOWN!=0:
                death = True
                print('Pausing detection because of potential death')
            last_fifteen.append(len(final_contours))
            if len(last_fifteen) > 15:
                last_fifteen.pop(0)
            moving_median = np.median(last_fifteen)
            if moving_median > last_mov_med:
                if frames_since_last_kill > 10:
                    print('Kill detected! on frame ', cur_frame_count)
                    kill_detection_frames.append(cur_frame_count)
                    total+=1
                    frames_since_last_kill = 0
            last_mov_med = moving_median
            frames_since_last_kill += 1

    print("total kills: ", total)
    return kill_detection_frames        

def write_bb_kills(infile, outfile):
    kill_detection_frames = _get_bb_kills(infile)
    with open(outfile, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(kill_detection_frames)  # Writes the entire list in one row

if __name__ =="__main__":
    outfilename= os.path.join("csg_videos_bb_labels", sys.argv[1].split("/")[-1] +".txt")
    write_bb_kills(sys.argv[1], outfilename)



