#!/usr/bin/env python3
import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2
from opcua import Server
from collections import deque
from datetime import datetime
import threading

# ---- pymodbus imports ----
from pymodbus.server import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.transaction import ModbusRtuFramer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSparseDataBlock, ModbusSequentialDataBlock


# ------------------ Modbus TCP Server Setup ------------------
# 1) PREDEFINE HR BLOCK FOR ADDRESSES 0 AND 1
#    Both registers start at 0. 
#    (If you eventually need more registers, extend the list [0,0] to [0,0,0,0,...].)
hr_block = ModbusSequentialDataBlock(0, [0, 0])

store = ModbusSlaveContext(
    hr=hr_block,    # Use the SequentialDataBlock, not a dict!
    zero_mode=True  # Keep address decoding in zero-mode (i.e. registers start at 0)
)
context = ModbusServerContext(slaves=store, single=True)

# Server identity (optional)
identity = ModbusDeviceIdentification()
identity.VendorName = 'ReactorSensor'
identity.ProductCode = 'RL'
identity.VendorUrl = 'http://reactor.local'
identity.ProductName = 'Interfacial Level Sensor'
identity.ModelName = 'ModbusTCP'
identity.MajorMinorRevision = '1.0'

# Start the Modbus TCP server in a background thread
def run_modbus_server():
    # Note: context, identity and address must all be passed by keyword
    StartTcpServer(
        context=context,
        identity=identity,
        address=('0.0.0.0', 5020),
    )

modbus_thread = threading.Thread(target=run_modbus_server, daemon=True)
modbus_thread.start()
print("Modbus TCP server started on port 5020")


def cluster_levels(levels, tolerance=10):
    """
    Clusters a sorted list of y-coordinate values that are within 'tolerance' pixels.
    Returns a list of the averaged cluster values.
    """
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lvl in levels[1:]:
        if lvl - clusters[-1][-1] <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    # Compute average of each cluster
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]

def weighted_determine_interfacial_level(candidates, threshold=5, top_surface=None, min_sep_px=0):
    """
    Determines the interfacial level using a weighted average when the candidate levels are close.
    Weights: Zone2 droplet > Zone1 droplet > Zone2 line. Zone1 line candidates are ignored.
    """

    # remove meniscus‐adjacent lines:
    if top_surface is not None and min_sep_px > 0:
        candidates = [c for c in candidates if not (c['type']=='line' and abs(c['level'] - top_surface) < min_sep_px)]

    # if any droplet remains, drop all lines
    if any(c['type']=='droplet' for c in candidates):
        candidates = [c for c in candidates if c['type']=='droplet']

    filtered_candidates = [c for c in candidates if not (c['zone'] == 1 and c['type'] == 'line')]
    if not filtered_candidates:
        return None

    for candidate in filtered_candidates:
        if candidate['zone'] == 2 and candidate['type'] == 'droplet':
            candidate['weight'] = 7
        elif candidate['zone'] == 1 and candidate['type'] == 'droplet':
            candidate['weight'] = 3
        elif candidate['zone'] == 2 and candidate['type'] == 'line':
            candidate['weight'] = 1
        else:
            candidate['weight'] = 0

    levels = [c['level'] for c in filtered_candidates]
    if max(levels) - min(levels) <= threshold:
        numerator = sum(c['level'] * c['weight'] for c in filtered_candidates)
        denominator = sum(c['weight'] for c in filtered_candidates)
        return numerator / denominator
    else:
        best_candidate = max(filtered_candidates, key=lambda c: c['weight'])
        return best_candidate['level']


# ---------------- Picamera2 Setup and ROI/Zone Selection ----------------
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (4056, 3040)})
picam2.configure(preview_config)
picam2.start()
time.sleep(1)  # Allow camera to warm up

# Capture the first frame for manual ROI and zone selection
first_frame = picam2.capture_array().copy()

# User to select the overall ROI on the first frame
roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")
if roi == (0, 0, 0, 0):
    picam2.close()
    exit("No ROI selected.")
x_roi, y_roi, w_roi, h_roi = roi
roi_first = first_frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

# Physical dimensions for calibration (you must measure these to calibrate)
H_ROI_phys_cm     = 4.3   # e.g. the visible reactor height in ROI in cm
H_ROI_pix         = h_roi  # pixel height of your ROI (from the selectROI call)
OFFSET_phys_cm    = 20.7   # distance from bottom of ROI to reactor bottom in cm

cm_per_pixel = H_ROI_phys_cm / H_ROI_pix

# Let the user select Zone 1 (within the ROI)
zone1_rect = cv2.selectROI("Select Zone 1 (within ROI)", roi_first, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select Zone 1 (within ROI)")
if zone1_rect == (0, 0, 0, 0):
    picam2.close()
    exit("No Zone 1 selected.")
zx1, zy1, zw1, zh1 = zone1_rect

# Let the user select Zone 2 (within the ROI)
zone2_rect = cv2.selectROI("Select Zone 2 (within ROI)", roi_first, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select Zone 2 (within ROI)")
if zone2_rect == (0, 0, 0, 0):
    picam2.close()
    exit("No Zone 2 selected.")
zx2, zy2, zw2, zh2 = zone2_rect

print("Selections complete. Processing video stream...")

# ---------------- Create a folder for saving images ----------------
save_folder = "saved_images"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Define parameters for preprocessing and edge detection
lower_thresh = 50
upper_thresh = 100
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

# Define kernels for morphological operations
kernel_erosion_zone1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_closing_1_zone1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
kernel_opening_zone1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_closing_2_zone1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))

kernel_erosion_zone2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_closing_1_zone2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
kernel_opening_zone2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_closing_2_zone2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

def run_detection(adap_thresh1, adap_thresh2):
    # Morphological operations for Zone 1
    edges_eroded_zone1 = cv2.erode(adap_thresh1, kernel_erosion_zone1, iterations=1)
    edges_closed_1_zone1 = cv2.morphologyEx(edges_eroded_zone1, cv2.MORPH_CLOSE, kernel_closing_1_zone1, iterations=1)
    edges_opened_zone1 = cv2.morphologyEx(edges_closed_1_zone1, cv2.MORPH_OPEN, kernel_opening_zone1, iterations=2)
    edges_closed_2_zone1 = cv2.morphologyEx(edges_opened_zone1, cv2.MORPH_CLOSE, kernel_closing_2_zone1, iterations=1)

    # Morphological operations for Zone 2
    edges_eroded_zone2 = cv2.erode(adap_thresh2, kernel_erosion_zone2, iterations=1)
    edges_closed_1_zone2 = cv2.morphologyEx(edges_eroded_zone2, cv2.MORPH_CLOSE, kernel_closing_1_zone2, iterations=1)
    edges_opened_zone2 = cv2.morphologyEx(edges_closed_1_zone2, cv2.MORPH_OPEN, kernel_opening_zone2, iterations=2)
    edges_closed_2_zone2 = cv2.morphologyEx(edges_opened_zone2, cv2.MORPH_CLOSE, kernel_closing_2_zone2, iterations=1)

    inverted_edges_zone1 = cv2.bitwise_not(edges_closed_2_zone1)
    inverted_edges_zone2 = cv2.bitwise_not(edges_closed_2_zone2)

    # ---- Line Detection ----
    linesP_zone1 = cv2.HoughLinesP(inverted_edges_zone1, rho=1, theta=np.pi/180, threshold=100, minLineLength=30, maxLineGap=50)
    levels_zone1 = []
    if linesP_zone1 is not None:
        for line in linesP_zone1:
            x1, y1, x2, y2 = line[0]
            if abs(y1-y2) < 20:  # nearly horizontal
                level_y = int(round((y1+y2)/2))
                levels_zone1.append(level_y)
    clustered_zone1 = cluster_levels(levels_zone1, tolerance=5)
    if clustered_zone1:
        clustered_zone1_sorted = sorted(clustered_zone1)
        top_level_zone1 = clustered_zone1_sorted[0] + zy1  # free surface
        interfacial_candidate_line_zone1 = clustered_zone1_sorted[1] + zy1 if len(clustered_zone1_sorted) >= 2 else None
    else:
        top_level_zone1 = None
        interfacial_candidate_line_zone1 = None

    linesP_zone2 = cv2.HoughLinesP(inverted_edges_zone2, rho=1, theta=np.pi/180, threshold=100, minLineLength=20, maxLineGap=30)
    levels_zone2 = []
    if linesP_zone2 is not None:
        for line in linesP_zone2:
            x1, y1, x2, y2 = line[0]
            if abs(y1-y2) < 10:
                level_y = int(round((y1+y2)/2))
                levels_zone2.append(level_y)
    clustered_zone2 = cluster_levels(levels_zone2, tolerance=5)
    interfacial_candidate_line_zone2 = min(clustered_zone2) + zy2 if clustered_zone2 else None

    # ---- Droplet Detection ----
    contours_zone1, _ = cv2.findContours(inverted_edges_zone1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    droplet_levels_zone1 = []
    for cnt in contours_zone1:
        area = cv2.contourArea(cnt)
        if 10 < area < 80:
            x, y, w, h = cv2.boundingRect(cnt)
            droplet_levels_zone1.append(y + h // 2)
    if droplet_levels_zone1:
        clustered_droplets_zone1 = cluster_levels(droplet_levels_zone1, tolerance=10)
        if clustered_droplets_zone1:
            interfacial_candidate_droplet_zone1 = clustered_droplets_zone1[len(clustered_droplets_zone1) // 2] + zy1
        else:
            interfacial_candidate_droplet_zone1 = None
    else:
        interfacial_candidate_droplet_zone1 = None

    contours_zone2, _ = cv2.findContours(inverted_edges_zone2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    droplet_levels_zone2 = []
    for cnt in contours_zone2:
        area = cv2.contourArea(cnt)
        if 100 < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            droplet_levels_zone2.append(y + h // 2)
    if droplet_levels_zone2:
        clustered_droplets_zone2 = cluster_levels(droplet_levels_zone2, tolerance=10)
        if clustered_droplets_zone2:
            interfacial_candidate_droplet_zone2 = clustered_droplets_zone2[len(clustered_droplets_zone2) // 2] + zy2
        else:
            interfacial_candidate_droplet_zone2 = None
    else:
        interfacial_candidate_droplet_zone2 = None

    return (top_level_zone1, interfacial_candidate_line_zone1, interfacial_candidate_line_zone2,
            interfacial_candidate_droplet_zone1, interfacial_candidate_droplet_zone2,
            edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1,
            inverted_edges_zone1, edges_eroded_zone2, edges_closed_1_zone2, edges_opened_zone2,
            edges_closed_2_zone2, inverted_edges_zone2)

try:
    while True:
        # Capture a frame for detection
        frame = picam2.capture_array()
        if frame is None:
            continue

        # Extract the ROI from the captured frame
        roi_img = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        result_roi = roi_img.copy()
        H_roi, W_roi = roi_img.shape[:2]

        # Extract Zone 1 and Zone 2 from the ROI
        zone1_img = roi_img[zy1:zy1+zh1, zx1:zx1+zw1]
        zone2_img = roi_img[zy2:zy2+zh2, zx2:zx2+zw2]

        # Preprocessing
        filtered_zone1 = cv2.medianBlur(zone1_img, 5)
        filtered_zone2 = cv2.medianBlur(zone2_img, 5)
        zone1_gray = cv2.cvtColor(filtered_zone1, cv2.COLOR_BGR2GRAY) if len(zone1_img.shape)==3 else zone1_img
        zone2_gray = cv2.cvtColor(filtered_zone2, cv2.COLOR_BGR2GRAY) if len(zone2_img.shape)==3 else zone2_img

        enhanced_zone1 = clahe.apply(zone1_gray)
        enhanced_zone2 = clahe.apply(zone2_gray)

        # Use adaptive thresholding on the original gray images
        adaptive_thresh_zone1 = cv2.adaptiveThreshold(zone1_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
        adaptive_thresh_zone2 = cv2.adaptiveThreshold(zone2_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

        # Optionally, you could also use the enhanced images if needed.
        (top_level_zone1, interfacial_candidate_line_zone1, interfacial_candidate_line_zone2,
         interfacial_candidate_droplet_zone1, interfacial_candidate_droplet_zone2,
         edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1,
         inverted_edges_zone1, edges_eroded_zone2, edges_closed_1_zone2, edges_opened_zone2,
         edges_closed_2_zone2, inverted_edges_zone2) = run_detection(adaptive_thresh_zone1, adaptive_thresh_zone2)

        # If no levels detected, try using enhanced images
        if top_level_zone1 is None:
            adaptive_thresh_zone1 = cv2.adaptiveThreshold(enhanced_zone1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
            adaptive_thresh_zone2 = cv2.adaptiveThreshold(enhanced_zone2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
            (top_level_zone1, interfacial_candidate_line_zone1, interfacial_candidate_line_zone2,
             interfacial_candidate_droplet_zone1, interfacial_candidate_droplet_zone2,
             edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1,
             inverted_edges_zone1, edges_eroded_zone2, edges_closed_1_zone2, edges_opened_zone2,
             edges_closed_2_zone2, inverted_edges_zone2) = run_detection(adaptive_thresh_zone1, adaptive_thresh_zone2)

        # Gather candidate levels
        candidates = []
        if interfacial_candidate_droplet_zone2 is not None:
            candidates.append({'level': interfacial_candidate_droplet_zone2, 'zone': 2, 'type': 'droplet'})
        if interfacial_candidate_line_zone2 is not None:
            candidates.append({'level': interfacial_candidate_line_zone2, 'zone': 2, 'type': 'line'})
        if interfacial_candidate_droplet_zone1 is not None:
            candidates.append({'level': interfacial_candidate_droplet_zone1, 'zone': 1, 'type': 'droplet'})
        if interfacial_candidate_line_zone1 is not None:
            candidates.append({'level': interfacial_candidate_line_zone1, 'zone': 1, 'type': 'line'})

        # filter and weight: avoid meniscus lines, prefer droplets
        new_interfacial_level = weighted_determine_interfacial_level(candidates, threshold=5, top_surface=top_level_zone1, min_sep_px=10)
        # Only proceed if we have a numeric pixel-level
        if new_interfacial_level is not None:
            # Round to integer pixel row
            interfacial_level = int(round(new_interfacial_level))

            # Convert pixel row to physical height in cm
            # (pixels above bottom of ROI → cm)
            pix_above = H_ROI_pix - interfacial_level
            h_cm = OFFSET_phys_cm + pix_above * cm_per_pixel

            # Write physical height into Modbus holding register #0
            # scale as needed (e.g. multiply by 100 for two decimal precision)
            reg_value = int(round(h_cm * 100))
            # directly update the slave context:
            context[0].setValues(3, 0, [reg_value])

            print(f"[Modbus] Updated register 0 with {reg_value} (x0.01 cm)")

            # 3) Annotate on image if you still want to see it visually
            #ytxt = max(0, interfacial_level - 25)
            #cv2.putText(result_roi, f"{h_cm:.1f} cm", (7, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        else:
            # No valid detection this frame—skip publishing
            print("No interfacial level detected this frame.")

        if top_level_zone1 is not None:
            cv2.line(result_roi, (0, top_level_zone1), (W_roi, top_level_zone1), (0, 255, 0), 3)
            cv2.putText(result_roi, "Top Level", (7, top_level_zone1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
            # compute and annotate top-level height in cm
            pix_above_top = H_ROI_pix - top_level_zone1
            top_height_cm = OFFSET_phys_cm + pix_above_top * cm_per_pixel

            # Write physical height into Modbus holding register #1
            # scale as needed (e.g. multiply by 100 for two decimal precision)
            reg_value_top = int(round(top_height_cm * 100))
            # directly update the slave context:
            context[0].setValues(3, 1, [reg_value_top])

            print(f"[Modbus] Updated register 1 with {reg_value_top} (x0.01 cm)")

            cv2.putText(result_roi, f"{top_height_cm:.1f} cm", (W_roi - 100, top_level_zone1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
             
            if interfacial_candidate_line_zone1 is not None and interfacial_candidate_line_zone1 > top_level_zone1:
                cv2.line(result_roi, (0, interfacial_candidate_line_zone1), (W_roi, interfacial_candidate_line_zone1), (255, 0, 0), 2)
                cv2.putText(result_roi, "Line Cand. Z1", (7, interfacial_candidate_line_zone1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if interfacial_candidate_line_zone2 is not None and interfacial_candidate_line_zone2 > top_level_zone1:
                cv2.line(result_roi, (0, interfacial_candidate_line_zone2), (W_roi, interfacial_candidate_line_zone2), (0, 0, 255), 2)
                cv2.putText(result_roi, "Line Cand. Z2", (7, interfacial_candidate_line_zone2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if interfacial_candidate_droplet_zone1 is not None and interfacial_candidate_droplet_zone1 > top_level_zone1:
                cv2.line(result_roi, (0, interfacial_candidate_droplet_zone1), (W_roi, interfacial_candidate_droplet_zone1), (0, 255, 255), 2)
                cv2.putText(result_roi, "droplet Cand. Z1", (7, interfacial_candidate_droplet_zone1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if interfacial_candidate_droplet_zone2 is not None and interfacial_candidate_droplet_zone2 > top_level_zone1:
                cv2.line(result_roi, (0, interfacial_candidate_droplet_zone2), (W_roi, interfacial_candidate_droplet_zone2), (255, 255, 0), 2)
                cv2.putText(result_roi, "droplet Cand. Z2", (7, interfacial_candidate_droplet_zone2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if interfacial_level is not None and interfacial_level > top_level_zone1:
                cv2.line(result_roi, (0, interfacial_level), (W_roi, interfacial_level), (255, 255, 255), 3)
                cv2.putText(result_roi, "Final Interfacial", (7, interfacial_level - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # --- compute and display the height in cm ---
                # pixels from bottom of ROI up to the detected level
                pix_above_if = H_ROI_pix - interfacial_level
                # convert to cm with your offset
                h_if_cm = OFFSET_phys_cm + pix_above_if * cm_per_pixel
                # choose a y-position for the text just above the line
                y_txt_if = max(0, interfacial_level - 25)
                cv2.putText(result_roi, f"{h_if_cm:.1f} cm", (W_roi - 100, y_txt_if), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save the annotated image with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_folder, f"result_{timestamp}.png")
        cv2.imwrite(filename, result_roi)
        print(f"Saved: {filename}")

        # Sleep for 1 second before capturing the next frame
        time.sleep(1)


except KeyboardInterrupt:
    print("Shutting down")
    print("Process terminated by user.")

# Cleanup
picam2.close()
cv2.destroyAllWindows()

