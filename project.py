from darknet import Darknet
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

model = Darknet('../yolov3.cfg')
model.load_weights('../yolov3.weights')

cap = cv2.VideoCapture('../video.mp4')


def draw_square(img, x, y, w, h):
	w, h = int(w), int(h)
	cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 0), thickness=2)


while cap.isOpened():
	ret, frame = cap.read()
	
	h, w, _ = frame.shape
	diff = w - h
	pad = diff // 2
	frame = frame[:, pad:w - pad, :]
	frame = cv2.resize(frame, (608, 608))
	real_frame = frame.copy()
	
	frame = frame.transpose((2, 0, 1))
	tensor = torch.Tensor(frame)
	tensor = tensor.unsqueeze(0)
	output = model(tensor, CUDA=False).numpy()
	
	# print(output.size())
	# input(output)
	for i in range(output.shape[1]):
		if output[0, i, 4] > 0.3:
			coords = output[0, i, :4]
			draw_square(real_frame, *[int(x) for x in coords])
	
	cv2.imshow('frame', real_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
