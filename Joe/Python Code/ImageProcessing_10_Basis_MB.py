import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def cluster_levels(levels, tolerance=10):
    """
    Clusters a sorted list of y-coordinate values that are within 'tolerance' pixels.
    Returns a list of the averaged cluster values.
    """
    if not levels:
        return []
    levels.sort()  # Ensure ascending order
    clustered = []
    current_cluster = [levels[0]]
    
    for lvl in levels[1:]:
        if lvl - current_cluster[-1] <= tolerance:
            current_cluster.append(lvl)
        else:
            clustered.append(int(round(sum(current_cluster) / len(current_cluster))))
            current_cluster = [lvl]
    if current_cluster:
        clustered.append(int(round(sum(current_cluster) / len(current_cluster))))
    return clustered


def weighted_determine_interfacial_level(candidates, threshold=5, top_surface=None, min_sep_px=0):
    """
    Determines the interfacial level using a weighted average when the candidate levels are close,
    giving more weight to the Droplet candidate of Zone 2 (weight 3), then Droplet candidate of Zone 1 (weight 2),
    and then the line candidate of Zone 2 (weight 1). It ignores any Zone 1 line candidate.
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

    # Assign weights according to type and zone
    for candidate in filtered_candidates:
        if candidate['zone'] == 2 and candidate['type'] == 'Droplet':
            candidate['weight'] = 7
        elif candidate['zone'] == 1 and candidate['type'] == 'Droplet':
            candidate['weight'] = 3
        elif candidate['zone'] == 2 and candidate['type'] == 'line':
            candidate['weight'] = 1
        else:
            candidate['weight'] = 0  # Should not occur because we filtered out Zone 1 lines

    # Check if candidate levels are close to each other
    levels = [c['level'] for c in filtered_candidates]
    if max(levels) - min(levels) <= threshold:
        # Compute weighted average
        numerator = sum(c['level'] * c['weight'] for c in filtered_candidates)
        denominator = sum(c['weight'] for c in filtered_candidates)
        return numerator / denominator
    else:
        # If not close, return the candidate with the highest weight
        best_candidate = max(filtered_candidates, key=lambda c: c['weight'])
        return best_candidate['level']
    

# ---------------- Step 1: Select the Image and Overall ROI ----------------

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
if not file_path:
    exit("No file selected.")

img = cv2.imread(file_path)
if img is None:
    exit("Failed to load image.")

cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
roi = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")
if roi == (0, 0, 0, 0):
    exit("No ROI selected.")

x_roi, y_roi, w_roi, h_roi = roi
roi_img = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

# ---------------- Step 2: Manually Select the Two Zones within the ROI ----------------

# Zone 1 (top zone for free surface)
cv2.namedWindow("Select Zone 1", cv2.WINDOW_NORMAL)
zone1_rect = cv2.selectROI("Select Zone 1", roi_img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()
if zone1_rect == (0, 0, 0, 0):
    exit("No Zone 1 selected.")
zx1, zy1, zw1, zh1 = zone1_rect
zone1_img = roi_img[zy1:zy1+zh1, zx1:zx1+zw1]

# Zone 2 (bottom zone for interfacial level)
cv2.namedWindow("Select Zone 2", cv2.WINDOW_NORMAL)
zone2_rect = cv2.selectROI("Select Zone 2", roi_img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()
if zone2_rect == (0, 0, 0, 0):
    exit("No Zone 2 selected.")
zx2, zy2, zw2, zh2 = zone2_rect
zone2_img = roi_img[zy2:zy2+zh2, zx2:zx2+zw2]

# ---------------- Preprocessing ----------------

# Blurring and grayscale conversion
blurred_zone1 = cv2.GaussianBlur(zone1_img, (5, 5), 0)
blurred_zone2 = cv2.GaussianBlur(zone2_img, (5, 5), 0)

filtered_zone1 = cv2.medianBlur(zone1_img, 5)
filtered_zone2 = cv2.medianBlur(zone2_img, 5)

zone1_gray = cv2.cvtColor(filtered_zone1, cv2.COLOR_BGR2GRAY) if len(zone1_img.shape)==3 else zone1_img
zone2_gray = cv2.cvtColor(filtered_zone2, cv2.COLOR_BGR2GRAY) if len(zone2_img.shape)==3 else zone2_img

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
enhanced_zone1 = clahe.apply(zone1_gray)
enhanced_zone2 = clahe.apply(zone2_gray)

# Apply Canny edge detection
lower = 50  # Example fixed lower threshold
upper = 100  # Example fixed upper threshold
edges_zone1 = cv2.Canny(enhanced_zone1, lower, upper)
edges_zone2 = cv2.Canny(enhanced_zone2, lower, upper)

# ---------------- Adaptive Thresholding ----------------

# First try: using zoneX_gray as input.
adaptive_thresh_zone1 = cv2.adaptiveThreshold(zone1_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
adaptive_thresh_zone2 = cv2.adaptiveThreshold(zone2_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

# ---------------- Morphological Operations Kernels ----------------

kernel_erosion_zone1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_closing_1_zone1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
kernel_opening_zone1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_closing_2_zone1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))

kernel_erosion_zone2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_closing_1_zone2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
kernel_opening_zone2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_closing_2_zone2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))


# ---------------- Candidate Detection Function ----------------
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
    # Zone 1: Lines (for free surface and candidate interfacial line)
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
        interfacial_candidate_line_zone1 = clustered_zone1_sorted[1] + zy1 if len(clustered_zone1_sorted)>=2 else None
    else:
        top_level_zone1 = None
        interfacial_candidate_line_zone1 = None

    # Zone 2: Lines
    linesP_zone2 = cv2.HoughLinesP(inverted_edges_zone2, rho=1, theta=np.pi/180, threshold=100 , minLineLength=20, maxLineGap=30)
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
    # Zone 1:
    contours_zone1, _ = cv2.findContours(inverted_edges_zone1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    droplet_levels_zone1 = []
    for cnt in contours_zone1:
        area = cv2.contourArea(cnt)
        if 100 < area < 200:
            x, y, w, h = cv2.boundingRect(cnt)
            droplet_levels_zone1.append(y + h//2)
    if droplet_levels_zone1:
        clustered_droplets_zone1 = cluster_levels(droplet_levels_zone1, tolerance=10)
        interfacial_candidate_droplet_zone1 = int(round(np.median(clustered_droplets_zone1))) + zy1
    else:
        interfacial_candidate_droplet_zone1 = None

    # Zone 2:
    contours_zone2, _ = cv2.findContours(inverted_edges_zone2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    droplet_levels_zone2 = []
    for cnt in contours_zone2:
        area = cv2.contourArea(cnt)
        if 100 < area < 500:
            x, y, w, h = cv2.boundingRect(cnt)
            droplet_levels_zone2.append(y + h//2)
    if droplet_levels_zone2:
        clustered_droplets_zone2 = cluster_levels(droplet_levels_zone2, tolerance=10)
        interfacial_candidate_droplet_zone2 = int(round(np.median(clustered_droplets_zone2))) + zy2
    else:
        interfacial_candidate_droplet_zone2 = None

    return (top_level_zone1, interfacial_candidate_line_zone1, interfacial_candidate_line_zone2, interfacial_candidate_droplet_zone1, 
            interfacial_candidate_droplet_zone2, edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1, 
            inverted_edges_zone1, edges_eroded_zone2, edges_closed_1_zone2, edges_opened_zone2, edges_closed_2_zone2, inverted_edges_zone2)


# Run candidate detection using the default adaptive threshold (zoneX_gray input)
(top_level_zone1, interfacial_candidate_line_zone1, interfacial_candidate_line_zone2, interfacial_candidate_droplet_zone1, interfacial_candidate_droplet_zone2,
 edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1, inverted_edges_zone1, edges_eroded_zone2, edges_closed_1_zone2, 
 edges_opened_zone2, edges_closed_2_zone2, inverted_edges_zone2) = run_detection(adaptive_thresh_zone1, adaptive_thresh_zone2)


# If no candidate levels were detected, re-run adaptive thresholding using the enhanced images.
if (top_level_zone1 is None):     #if (top_level_zone1 is None and interfacial_candidate_line_zone1 is None and interfacial_candidate_line_zone2 is None and interfacial_candidate_droplet_zone1 is None and interfacial_candidate_droplet_zone2 is None):
    print("No levels detected using the default (gray) images. Reprocessing using enhanced images.")
    adaptive_thresh_zone1 = cv2.adaptiveThreshold(enhanced_zone1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    adaptive_thresh_zone2 = cv2.adaptiveThreshold(enhanced_zone2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

(top_level_zone1, interfacial_candidate_line_zone1, interfacial_candidate_line_zone2, interfacial_candidate_droplet_zone1, interfacial_candidate_droplet_zone2,
edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1, inverted_edges_zone1, edges_eroded_zone2, edges_closed_1_zone2, 
edges_opened_zone2, edges_closed_2_zone2, inverted_edges_zone2) = run_detection(adaptive_thresh_zone1, adaptive_thresh_zone2)

# ---------------- Final Decision for the Interfacial Level ----------------
# Preference order:
# 1. Bubbles in Zone 2
# 2. Lines in Zone 2
# 3. Bubbles in Zone 1
# 4. (Optional) Lines in Zone 1

# Gather candidates into a list with their properties
candidates = []
if interfacial_candidate_droplet_zone2 is not None:
    candidates.append({'level': interfacial_candidate_droplet_zone2, 'zone': 2, 'type': 'Droplet'})
if interfacial_candidate_line_zone2 is not None:
    candidates.append({'level': interfacial_candidate_line_zone2, 'zone': 2, 'type': 'line'})
if interfacial_candidate_droplet_zone1 is not None:
    candidates.append({'level': interfacial_candidate_droplet_zone1, 'zone': 1, 'type': 'Droplet'})
if interfacial_candidate_line_zone1 is not None:
    candidates.append({'level': interfacial_candidate_line_zone1, 'zone': 1, 'type': 'line'})

# Use the new weighted logic: if candidate levels are close, use the weighted average;
# otherwise, choose the candidate with the highest weight.
new_interfacial_level = weighted_determine_interfacial_level(candidates, threshold=5, top_surface=top_level_zone1, min_sep_px=10)
if new_interfacial_level is not None:
    interfacial_level = new_interfacial_level
else:
    # Fallback to the original preference order if needed
    if interfacial_candidate_droplet_zone2 is not None:
        interfacial_level = interfacial_candidate_droplet_zone2
    elif interfacial_candidate_line_zone2 is not None:
        interfacial_level = interfacial_candidate_droplet_zone1
    elif interfacial_candidate_droplet_zone1 is not None:
        interfacial_level = interfacial_candidate_line_zone2
    elif interfacial_candidate_line_zone1 is not None:
        interfacial_level = interfacial_candidate_line_zone1
    else:
        interfacial_level = None


# ---------------- Drawing the Results on the Overall ROI ----------------
result_roi = roi_img.copy()
H_roi, W_roi = roi_img.shape[:2]

if top_level_zone1 is not None:
    cv2.line(result_roi, (0, top_level_zone1), (W_roi, top_level_zone1), (0, 255, 0), 3)
    cv2.putText(result_roi, "Top Level", (7, top_level_zone1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Now that we know top_level_zone1 is valid, we safely compare it.
    if interfacial_candidate_line_zone1 is not None and interfacial_candidate_line_zone1 > top_level_zone1:
        cv2.line(result_roi, (0, interfacial_candidate_line_zone1), (W_roi, interfacial_candidate_line_zone1), (255, 0, 0), 2)
        cv2.putText(result_roi, "Line Cand. Z1", (7, interfacial_candidate_line_zone1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if interfacial_candidate_line_zone2 is not None and interfacial_candidate_line_zone2 > top_level_zone1:
        cv2.line(result_roi, (0, interfacial_candidate_line_zone2), (W_roi, interfacial_candidate_line_zone2), (0, 0, 255), 2)
        cv2.putText(result_roi, "Line Cand. Z2", (7, interfacial_candidate_line_zone2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if interfacial_candidate_droplet_zone1 is not None and interfacial_candidate_droplet_zone1 > zy2:
        cv2.line(result_roi, (0, interfacial_candidate_droplet_zone1), (W_roi, interfacial_candidate_droplet_zone1), (0, 255, 255), 2)
        cv2.putText(result_roi, "Droplet Cand. Z1", (7, interfacial_candidate_droplet_zone1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if interfacial_candidate_droplet_zone2 is not None and interfacial_candidate_droplet_zone2 > top_level_zone1:
        cv2.line(result_roi, (0, interfacial_candidate_droplet_zone2), (W_roi, interfacial_candidate_droplet_zone2), (255, 255, 0), 2)
        cv2.putText(result_roi, "Droplet Cand. Z2", (7, interfacial_candidate_droplet_zone2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if interfacial_level is not None and interfacial_level > top_level_zone1:
        cv2.line(result_roi, (0, interfacial_level), (W_roi, interfacial_level), (255, 255, 255), 3)
        cv2.putText(result_roi, "Final Interfacial", (7, interfacial_level - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ---------------- Displaying the Results ----------------
titles = ["Original", "Gaussian Blur", "Median Blur", "Grayscale", "CLAHE", "Adaptive Threshold",
          "Canny Edges", "Erosion", "First Closing", "Opening", "Second Closing", "Inverted Edges"]

zone1_images = [zone1_img, blurred_zone1, filtered_zone1, zone1_gray, enhanced_zone1, adaptive_thresh_zone1, 
    edges_zone1, edges_eroded_zone1, edges_closed_1_zone1, edges_opened_zone1, edges_closed_2_zone1, inverted_edges_zone1]

zone2_images = [zone2_img, blurred_zone2, filtered_zone2, zone2_gray, enhanced_zone2, adaptive_thresh_zone2,
    edges_zone2, edges_eroded_zone2, edges_closed_1_zone2, edges_opened_zone2, edges_closed_2_zone2, inverted_edges_zone2]

n_steps = len(titles)
total_cols = n_steps + 1

fig = plt.figure(figsize=(30, 20))
gs = gridspec.GridSpec(2, total_cols, width_ratios=[1]*n_steps + [1.5])

def display_image(ax, img):
    if img is None:
        ax.text(0.5, 0.5, "None", fontsize=10, ha="center")
    else:
        if len(img.shape)==3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
    ax.axis("off")

for i in range(n_steps):
    ax = fig.add_subplot(gs[0, i])
    display_image(ax, zone1_images[i])
    ax.set_title("Zone 1:\n" + titles[i], fontsize=10)
for i in range(n_steps):
    ax = fig.add_subplot(gs[1, i])
    display_image(ax, zone2_images[i])
    ax.set_title("Zone 2:\n" + titles[i], fontsize=10)

ax_overall = fig.add_subplot(gs[:, -1])
display_image(ax_overall, result_roi)
ax_overall.set_title("Detected Levels", fontsize=10)

fig_roi = plt.figure("ROI with Detected Levels", figsize=(10, 10))
ax_roi = fig_roi.add_subplot(111)
display_image(ax_roi, result_roi)
ax_roi.set_title("ROI with Detected Levels", fontsize=10)
ax_roi.axis("off")

plt.tight_layout()
plt.show()
