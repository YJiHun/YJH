from MyLibrary import myopencv as my
import cv2
import numpy as np

def binary(thresold, binary_img):
    binary = np.zeros_like(binary_img)
    binary[(binary_img >= thresold[0]) & (binary_img <= thresold[1])] = 255
    return binary
    pass

def babo(img,src,dst):
    height, width = img.shape[:2]
    dst_size = (width, height)
    src = src * np.float32([width, height])  ## width, height 비율 값
    dst = dst * np.float32(dst_size)  ## 이미지를 적용할 화면 비율
    M = cv2.getPerspectiveTransform(src, dst)  ## 자를 이미지 좌표값
    img_src = cv2.warpPerspective(img, M, dst_size)  ## 잘라낼 이미지, 잘라낼 이미지 영역값, 잘라낼 이미지를 붙일 영역 사이즈
    return img_src
    pass

def Yolo(img, score, nms):
    height, width = img.shape[:2]
    yolo_net = cv2.dnn.readNet('YOLO/yolov3.weights',
                               'YOLO/yolov3.cfg')
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in
                     yolo_net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
                                 True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 검출 신뢰도
            if confidence > 0.1:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                dw = int(detection[2] * width)
                dh = int(detection[3] * height)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score, nms)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            # 경계상자와 클래스 정보 투영
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(img, f'{label} {score:.2f}', (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2)
    return img
    pass

# with open('YOLO/coco.names', 'r') as f:
#     classes = [line.strip() for line in f.readline()]
#     pass

classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

margin = 150
nwindows = 9
minpix = 1

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

left_fit_ = np.empty(3)
right_fit_ = np.empty(3)

# name = 'project_video'
name = 'move_3'
Video = cv2.VideoCapture(f'videos/{name}.mp4')

lens = int(Video.get(cv2.CAP_PROP_FRAME_COUNT))

img_ = []
count = 0
A = 0
_ = True

pass
while _:
    _, img = Video.read()

    a = Yolo(img, 0.1, 0.4)

    cv2.imshow('result', a)
    key = cv2.waitKey(30)
    if key == 27:
        break
    # print(_)
    # img_.append(img)
    pass

# a = cv2.hconcat(img_)
# cv2.imshow('result', a)
# cv2.waitKey(0)
cv2.destroyAllWindows()
pass

pass
# while True:
#     _, img = Video.read()
#
#     height, width = img.shape[:2]
#
#     color_img = np.zeros_like(img)
#
#     if A == 0:
#         A = 1
#         print(img.shape)
#
#     img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#
#     img_hls_h, img_hls_l, img_hls_s = cv2.split(img_hls)
#
#     img_sobel_x = cv2.Sobel(img_hls_l, cv2.CV_64F, 1, 1)
#     img_sobel_x_abs = abs(img_sobel_x)
#     img_sobel_scaled = np.uint8(img_sobel_x_abs * 255 / np.max(img_sobel_x_abs))
#
#     sx_thresold = (15, 255)
#     s_thresold = (100, 255)
#
#     a = binary(sx_thresold, img_sobel_scaled)
#     b = binary(s_thresold, img_hls_s)
#
#     img_binary_added = cv2.addWeighted(a, 1., b, 1., 0)
#
#     height, width = img_binary_added.shape[:2]
#
#     dst_size = (1280, 720)
#
#     TopLeft = (0.43, 0.65)
#     TopRight = (0.58, 0.65)
#     BottomLeft = (0.1, 1.)
#     BottomRight = (1., 1.)
#
#     src = np.float32([TopLeft, TopRight, BottomLeft, BottomRight])  # TopLeft, TopRight , BottomLeft, BottomRight
#     dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]) ## 이미지를 적용할 화면 비율
#
#     img_warp = babo(img_binary_added,src,dst)
#
#     ## 이미지 자른 영역표시
#     B = np.float32([TopLeft, TopRight, BottomRight, BottomLeft])  # TopLeft, TopRight , BottomLeft, BottomRight
#     B = B * np.float32([width, height])  ## width, height 비율 값
#     B = np.int32(B)
#     cv2.polylines(img_binary_added, [B], True, color=(255, 0, 255), thickness=5)
#     ## 이미지 자른 영역표시
#
#
#     img_A = cv2.merge([img_warp,img_warp,img_warp])
#
#     height_1, width_1 = img_warp.shape[:2]
#
#     histogram = np.sum(img_warp[height_1 // 2:, :], axis=0)
#
#     midpoint = int(histogram.shape[0] / 2)
#
#     leftx_base = np.argmax(histogram[:midpoint])
#     rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#
#     window_height = int(height_1 / nwindows)
#
#     nonzero = img_warp.nonzero()
#     nonzero_y = np.array(nonzero[0])
#     nonzero_x = np.array(nonzero[1])
#
#     leftx_current = leftx_base
#     rightx_current = rightx_base
#
#     left_lane_inds = []
#     right_lane_inds = []
#
#     for window in range(nwindows):
#         win_y_low = height - (window + 1) * window_height
#         win_y_high = height - window * window_height
#
#         win_xleft_low = leftx_current - margin
#         win_xleft_high = leftx_current + margin
#
#         win_xright_low = rightx_current - margin
#         win_xright_high = rightx_current + margin
#
#         cv2.rectangle(color_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
#                       (100, 255, 255), 3)
#         cv2.rectangle(color_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
#                       (100, 255, 255), 3)
#
#         good_left_inds = ((nonzero_y >= win_y_low) &
#                           (nonzero_y < win_y_high) &
#                           (nonzero_x >= win_xleft_low) &
#                           (nonzero_x < win_xleft_high)).nonzero()[0]
#
#         good_right_inds = ((nonzero_y >= win_y_low) &
#                            (nonzero_y < win_y_high) &
#                            (nonzero_x >= win_xright_low) &
#                            (nonzero_x < win_xright_high)).nonzero()[0]
#
#         left_lane_inds.append(good_left_inds)
#         right_lane_inds.append(good_right_inds)
#
#         if len(good_left_inds) > minpix:
#             leftx_current = int(np.mean(nonzero_x[good_left_inds]))
#         if len(good_right_inds) > minpix:
#             rightx_current = int(np.mean(nonzero_x[good_right_inds]))
#
#     left_lane_inds = np.concatenate(left_lane_inds)
#     right_lane_inds = np.concatenate(right_lane_inds)
#
#     leftx = nonzero_x[left_lane_inds]
#     lefty = nonzero_y[left_lane_inds]
#     rightx = nonzero_x[right_lane_inds]
#     righty = nonzero_y[right_lane_inds]
#
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)
#
#     left_a.append(left_fit[0])
#     left_b.append(left_fit[1])
#     left_c.append(left_fit[2])
#     right_a.append(right_fit[0])
#     right_b.append(right_fit[1])
#     right_c.append(right_fit[2])
#
#     left_fit_[0] = np.mean(left_a[-10:])
#     left_fit_[1] = np.mean(left_b[-10:])
#     left_fit_[2] = np.mean(left_c[-10:])
#     right_fit_[0] = np.mean(right_a[-10:])
#     right_fit_[1] = np.mean(right_b[-10:])
#     right_fit_[2] = np.mean(right_c[-10:])
#
#     ploty = np.linspace(0, height_1 - 1, height_1)
#
#     left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
#     right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]
#
#     color_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
#     color_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 100, 255]
#
#     left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#     right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#     points = np.hstack((left, right))
#
#     mid_fitx = (left_fitx + right_fitx) // 2
#     mid_fitxy = np.vstack([mid_fitx, ploty])
#     mid_fitxy_t = np.transpose(mid_fitxy)
#     mid = np.array([mid_fitxy_t])
#
#     left_fitx_mean = np.int32(np.mean(left_fitx))
#     right_fitx_mean = np.int32(np.mean(right_fitx))
#     mid_fitx_mean = np.int32(np.mean(mid_fitx))
#
#     line_color = (0, 255, 0)
#     line_tick = 10
#     line_len = 20
#
#     cv2.fillPoly(color_img, np.int_(points), line_color)
#     cv2.polylines(color_img, np.int_(points), False, (0,255,255), 10)
#     cv2.polylines(color_img, np.int_(mid), False, (0, 255, 255), 10)
#     cv2.line(color_img, (left_fitx_mean, height_1 // 2 - line_len),
#              (left_fitx_mean, height_1 // 2 + line_len),
#              line_color, line_tick)
#     cv2.line(color_img, (right_fitx_mean, height_1 // 2 - line_len),
#              (right_fitx_mean, height_1 // 2 + line_len),
#              line_color, line_tick)
#     cv2.line(color_img, (mid_fitx_mean, height_1 // 2 - line_len),
#              (mid_fitx_mean, height_1 // 2 + line_len),
#              line_color, line_tick)
#     cv2.line(color_img, (width_1 // 2, height_1 // 2),
#              (width_1 // 2, height_1),
#              (255, 0, 0), line_tick)
#     cv2.line(color_img, (width_1 // 2, height_1 // 2),
#              (mid_fitx_mean, height_1 // 2),
#              (255, 255, 255), line_tick * 2)
#
#     road_width = 2.5 ## 도로 폭
#     road_width_pixel = road_width / (right_fitx_mean - left_fitx_mean) ## pixel 1 = m
#     error = width_1 // 2 - mid_fitx_mean ## 도로중심과 이미지 중심이 떨어진 픽셀 거리
#     dis_error = error * road_width_pixel ## 도로중심과 이미지 중심이 떨어진 거리
#
#     img_D = babo(color_img, dst, src)
#     img_A = cv2.addWeighted(img, 1., img_D, 0.4, 0)
#
#     img_A = Yolo(img_A, 0.5, 0.4)
#
#     text_size = 0.5
#     text_color = (0, 255, 0)
#     text_tick = 1
#     text_org = (width_1 // 2 + 40, height_1 // 2 + 150)
#
#     if dis_error > 0: # right
#         cv2.putText(img_A, 'right : {:.2f}m'.format(abs(dis_error)),
#                     text_org, cv2.FONT_ITALIC,
#                     text_size, text_color, text_tick)
#         pass
#     elif dis_error < 0: # left
#         cv2.putText(img_A, 'left : {:.2f}m'.format(abs(dis_error)),
#                     text_org, cv2.FONT_ITALIC,
#                     text_size, text_color, text_tick)
#         pass
#     else: # 중심
#         cv2.putText(img_A, 'center', text_org, cv2.FONT_ITALIC, text_size,
#                     (0, 0, 255), text_tick)
#         pass
#     #
#
#     if Video.get(cv2.CAP_PROP_POS_FRAMES) ==\
#             Video.get(cv2.CAP_PROP_FRAME_COUNT):
#         Video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#     cv2.imshow('c4', img_A)
#
#     key = cv2.waitKey(30)
#
#     if key == 27:
#         cv2.destroyAllWindows()
#         break
pass