import cv2, os
d = r'C:\Users\TZY\Downloads\DeltaForce-OBS-Locker-main\DeltaForce-OBS-Locker-main\yolo\data'
for f in sorted(os.listdir(d)):
    if f.endswith(('.jpg','.png','.jpeg')):
        img = cv2.imread(os.path.join(d, f))
        if img is not None:
            print(f'{f}: {img.shape[1]}x{img.shape[0]}')
