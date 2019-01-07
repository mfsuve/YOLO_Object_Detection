from darknet import Darknet
import cv2
import torch
import numpy as np
import os
from util import write_results, load_classes

### Model initialization
CUDA = False

model = Darknet('yolov3.cfg')
if CUDA:
	model = model.cuda()
model.load_weights('yolov3.weights')
### End Model initialization

# These 2 lines are just for us to test our model
frames_path = '../Istanbul_traffic_annotated/Istanbul_traffic_annotated/images'
classes = load_classes('coco.names')

# Predict Function
def predict(frame_img):
	result = []
	h, w, _ = frame_img.shape
	
	# Making it square by padding
	frame_img = cv2.copyMakeBorder(frame_img, 0, max(0, w - h), 0, max(0, h - w), cv2.BORDER_CONSTANT, value=0)
	
	frame_img = frame_img.transpose((2, 0, 1))
	tensor = torch.Tensor(frame_img)
	if CUDA:
		tensor = tensor.cuda()
	tensor = tensor.unsqueeze(0)
	with torch.no_grad():
		output = model(tensor, CUDA=CUDA, inp_dim=int(max(h, w)))
	
	output = write_results(output, 0.2, 80)
	
	if CUDA:
		output = output.cpu()
	output = output.numpy()
	
	for out in output:
		cls = int(out[-1])
		if cls == 2 or cls == 7 or cls == 5 or cls == 3 or cls == 1 or cls == 0:
			d_up = {}
			d = {}
			name = 'Car'
			if cls == 7:
				name = 'Truck'
			elif cls == 5:
				name = 'Bus'
			elif cls == 3:
				name = 'Motorcycle'
			elif cls == 1:
				name = 'Bicycle'
			elif cls == 0:
				name = 'Pedestrian'
			d['xmax'] = max(out[3], out[1])
			d['xmin'] = min(out[3], out[1])
			d['ymax'] = max(out[2], out[4])
			d['ymin'] = min(out[2], out[4])
			d['name'] = name
			d['confidence'] = out[-2]
			d_up['bndbox'] = d
			result.append(d_up)
	return result


def draw_rect(frame, text, color, left, top, right, bottom):
	cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	textsize = cv2.getTextSize(text, font, 1, 2)[0]
	cv2.putText(frame, text, ((left + right - textsize[0]) // 2, top - textsize[1] + 10), font, 1, color,
	            thickness=2)


for file in os.listdir(frames_path):
	file_path = os.path.join(frames_path, file)
	frame = cv2.imread(file_path)
	real_frame = frame.copy()
	h, w, _ = frame.shape
	
	print('Output from predict:')
	print(predict(frame))
	
	# Making it square by padding
	frame = cv2.copyMakeBorder(frame, 0, max(0, w - h), 0, max(0, h - w), cv2.BORDER_CONSTANT, value=0)
	
	frame = frame.transpose((2, 0, 1))
	tensor = torch.Tensor(frame)
	if CUDA:
		tensor = tensor.cuda()
	tensor = tensor.unsqueeze(0)
	with torch.no_grad():
		output = model(tensor, CUDA=CUDA, inp_dim=int(max(h, w)))
	
	output = write_results(output, 0.2, 80)
	
	if CUDA:
		output = output.cpu()
	output = output.numpy()
	
	for out in output:
		color = cv2.cvtColor(np.uint8([[[(out[-1] / 80) * 256, 255, 255]]]), cv2.COLOR_HSV2BGR)
		color = tuple(map(int, color[0, 0]))
		draw_rect(real_frame, classes[int(out[-1])], color, *map(int, out[1:5]))
	
	# cv2.imwrite('../outputs/' + '.'.join(file.split('.')[:-1]) + '.png', real_frame)
	cv2.imshow('frame', real_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
