from darknet import Darknet
import cv2
import torch
import numpy as np
import os
from util import write_results

CUDA = False

model = Darknet('../yolov3.cfg')
if CUDA:
	model = model.cuda()
model.load_weights('../yolov3.weights')

frames_path = '../Istanbul_traffic_annotated/Istanbul_traffic_annotated/images'

for file in os.listdir(frames_path):
	file_path = os.path.join(frames_path, file)
	frame = cv2.imread(file_path)
	real_frame = frame.copy()
	h, w, _ = frame.shape
	# Making it square by padding
	frame = cv2.copyMakeBorder(frame, 0, max(0, w - h), 0, max(0, h - w), cv2.BORDER_CONSTANT, value=0)
	
	frame = frame.transpose((2, 0, 1))
	tensor = torch.Tensor(frame)
	if CUDA:
		tensor = tensor.cuda()
	tensor = tensor.unsqueeze(0)
	with torch.no_grad():
		output = model(tensor, CUDA=False)
	
	output = write_results(output, 0.2, 80)
	if CUDA:
		output = output.cpu()
	output = output.numpy()
	
	for out in output:
		color = cv2.cvtColor(np.uint8([[[(out[-1] / 80) * 256, 255, 255]]]), cv2.COLOR_HSV2BGR)
		color = tuple(map(int, color[0, 0]))
		cv2.rectangle(real_frame, (out[1], out[2]), (out[3], out[4]), color, thickness=2)
	
	# cv2.imwrite('../outputs/' + '.'.join(file.split('.')[:-1]) + '.png', real_frame)
	cv2.imshow('frame', real_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
