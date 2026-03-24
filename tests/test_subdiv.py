import cv2

try:
    print("Testing (int, int, int, int):")
    cv2.Subdiv2D((0, 0, 100, 100))
    print("Success")
except Exception as e:
    print("Failed:", e)

try:
    print("Testing (float, float, float, float):")
    cv2.Subdiv2D((0.0, 0.0, 100.0, 100.0))
    print("Success")
except Exception as e:
    print("Failed:", e)

try:
    print("Testing (int, int, int, int) representing (x, y, x2, y2):")
    cv2.Subdiv2D((10, 10, 110, 110))
    print("Success")
except Exception as e:
    print("Failed:", e)
