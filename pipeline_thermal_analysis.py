import cv2
import numpy as np
import os

# -------------------------------------------------
# Image path
# -------------------------------------------------
image_path = r"C:\your file path here.png"

if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
    exit()

img = cv2.imread(image_path)
if img is None:
    print("Could not read image.")
    exit()

display = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

H, W = gray.shape

# -------------------------------------------------
# 1) HOT MASK
# -------------------------------------------------
lower_hot1 = np.array([15, 80, 220])
upper_hot1 = np.array([40, 255, 255])

lower_hot2 = np.array([0, 0, 245])
upper_hot2 = np.array([179, 70, 255])

hot_mask1 = cv2.inRange(hsv, lower_hot1, upper_hot1)
hot_mask2 = cv2.inRange(hsv, lower_hot2, upper_hot2)
hot_mask = cv2.bitwise_or(hot_mask1, hot_mask2)

hot_mask = cv2.morphologyEx(
    hot_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
)
hot_mask = cv2.morphologyEx(
    hot_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
)

# -------------------------------------------------
# 2) CONNECT HOT SPOTS ALONG THE SAME PIPE
# -------------------------------------------------
pipe_mask = hot_mask.copy()

kernel_pipe = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
pipe_mask = cv2.morphologyEx(pipe_mask, cv2.MORPH_CLOSE, kernel_pipe, iterations=2)

# -------------------------------------------------
# 3) FIND PIPE REGIONS
# -------------------------------------------------
contours, _ = cv2.findContours(pipe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pipe_boxes = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    if w < 10 or h < 10:
        continue

    pipe_boxes.append((x, y, w, h, area))

# Keep the 10 largest detected pipe regions
pipe_boxes = sorted(pipe_boxes, key=lambda b: b[4], reverse=True)[:10]

# Sort left to right
pipe_boxes = sorted(pipe_boxes, key=lambda b: b[0])

# -------------------------------------------------
# 4) CALCULATE WHITE PERCENT / HOT-SPOT LEVEL FOR EACH PIPE
# -------------------------------------------------
print(f"Detected pipes: {len(pipe_boxes)}")

for i, (x, y, w, h, area) in enumerate(pipe_boxes, start=1):
    roi_hot = hot_mask[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    total_pixels = roi_hot.shape[0] * roi_hot.shape[1]
    white_pixels = cv2.countNonZero(roi_hot)

    white_percent = 100.0 * white_pixels / total_pixels if total_pixels > 0 else 0.0

    # Mean grayscale intensity only where hot mask is white
    hot_pixels = roi_gray[roi_hot > 0]
    mean_hot_intensity = float(np.mean(hot_pixels)) if hot_pixels.size > 0 else 0.0

    # Draw red box
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 4)

    # Red text
    cv2.putText(display, f"Pipe Issue {i}", (x, max(y - 35, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.putText(display, f"Hot Spot: {white_percent:.1f}%", (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    print(
        f"Pipe {i}: white pixels={white_pixels}, total pixels={total_pixels}, "
        f"white percent={white_percent:.2f}%, mean hot intensity={mean_hot_intensity:.2f}"
    )

# -------------------------------------------------
# 5) SHOW Segmented Objects with Areas of Interest
# -------------------------------------------------
cv2.namedWindow("FLIR Pipe Hotspot Analysis", cv2.WINDOW_NORMAL)
cv2.namedWindow("Hot Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Pipe Mask", cv2.WINDOW_NORMAL)

print("Press q or ESC to quit.")

while True:
    cv2.imshow("FLIR Pipe Hotspot Analysis", display)
    cv2.imshow("Hot Mask", hot_mask)
    cv2.imshow("Pipe Mask", pipe_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break

    if cv2.getWindowProperty("FLIR Pipe Hotspot Analysis", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
for _ in range(5):
    cv2.waitKey(1)
