import cv2
import yaml
import numpy as np
drawing = False
roi_drawing = False
poly_drawing = False
point1 = ()
point2 = ()
lines = []
roi_points = []
poly_data = []
polygons = [] # Change this to hold multiple polygons
current_line = None
current_roi = None
current_poly = None
# add a new flag to monitor the current mode
drawing_mode = 'line'  # start with line mode as default

def draw_line(event, x, y, flags, param):
    global drawing, roi_drawing, point1, point2, current_line, current_roi, drawing_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_mode == 'line':
            drawing = True
            roi_drawing = False
        elif drawing_mode == 'roi':
            drawing = False
            roi_drawing = True
        elif drawing_mode == 'poly':
            drawing = False
            roi_drawing = False
            poly_drawing = True

        point1 = (x, y)

        if poly_drawing:
            if not polygons:
                polygons.append([])
            if polygons[-1] and abs(x - polygons[-1][0][0]) < 10 and abs(y - polygons[-1][0][1]) < 10: # 10 is a threshold for point distance
                polygons[-1].append(polygons[-1][0]) # close the polygon
                poly_drawing = False # reset the poly_drawing
                polygons.append([]) # start a new polygon
            else:
                polygons[-1].append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            point2 = (x, y)
            current_line = {'id': len(lines)+1, 'start': [point1[0], point1[1]], 'end': [point2[0], point2[1]]}
            

        elif roi_drawing:
            point2 = (x, y)
            current_roi = [point1[0], point1[1], point2[0], point2[1]]

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing and point1 and point2:
            lines.append(current_line)
            current_line = None
        elif roi_drawing and point1 and point2:
            roi_points.append(current_roi)
            current_roi = None
        point1, point2 = (), ()
        
with open('data.yaml', 'r') as file:
    data = yaml.safe_load(file)

video_stream_links = data.get('video_stream_link', [])
arquivos = data.get('arquivo')

if video_stream_links:
    sources = video_stream_links
elif arquivos:
    sources = arquivos
data['lines'] = []
data['roi_points'] = []
data['polygons'] = []
for i, source in enumerate(sources):
    lines = []
    roi_points = []
    polygons = []
    current_line = None
    current_roi = None

    cap = cv2.VideoCapture(arquivos)

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_line)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if drawing and point1 and point2:
            cv2.line(frame, point1, point2, (0, 255, 0), 2)
            midpoint = ((point1[0]+point2[0])//2, (point1[1]+point2[1])//2)  # Calculate the midpoint of the line
            normal = (-(point2[1]-point1[1]), point2[0]-point1[0])  # Calculate the normal vector
            # Scale the normal for better visibility
            if normal != (0, 0):
                normal = (normal[0]*50//max(abs(normal[0]), abs(normal[1])), normal[1]*50//max(abs(normal[0]), abs(normal[1])))
                # Draw the normal from the midpoint to the endpoint
                cv2.line(frame, midpoint, (midpoint[0]+normal[0], midpoint[1]+normal[1]), (255, 0, 0), 2)

        for poly_points in polygons:
            if poly_points:
                if poly_drawing and poly_points == polygons[-1]:
                    cv2.polylines(frame, [np.array(poly_points)], False, (255, 255, 0), 2)
                else:
                    cv2.polylines(frame, [np.array(poly_points)], True, (255, 255, 0), 2)

        for line in lines:
            cv2.line(frame, tuple(line['start']), tuple(line['end']), (0, 255, 0), 2)

        if roi_drawing and point1 and point2:
            cv2.rectangle(frame, point1, point2, (0, 0, 255), 2)

        for roi in roi_points:
            cv2.rectangle(frame, tuple(roi[:2]), tuple(roi[2:]), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(60)
        if key == ord('s'):
            line_data = [line for line in lines]
            data['lines'].append(line_data)  # append this video's lines to the global list
            roi_data = []
            for roi in roi_points:
                roi_data.extend(roi)  # add the coordinates of the roi to roi_data
            roi_data[2] = roi_data[2] - roi_data[0]
            roi_data[3] = roi_data[3] - roi_data[1]
            data['roi_points'].append(roi_data)  # append this video's roi to the global list
            print(line_data, roi_points)

            for i in range(len(polygons)):
                poly_data.append({'id': i + 1, 'points': polygons[i]})
            data['polygons'].append(poly_data)
            with open('data.yaml', 'w') as file:
                yaml.safe_dump(data, file, sort_keys=False)
            poly_data = []
        elif key == ord('d'):
            if lines:
                lines.pop()
            if roi_points:
                roi_points.pop()
            if polygons:
                polygons.pop()
        elif key == ord('l'):  # switch to line drawing mode when 'r' is pressed
            drawing_mode = 'line'
        elif key == ord('r'):  # switch to roi drawing mode when 'l' is pressed
            drawing_mode = 'roi'
        elif key == ord('n'):  # switch to next video when 'n' is pressed
            break
        if key == ord('p'):  # switch to polygon drawing mode when 'p' is pressed
            drawing_mode = 'poly'
            if not polygons or polygons[-1]: # if polygons is empty or the last polygon is not empty
                polygons.append([]) # start a new polygon
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
