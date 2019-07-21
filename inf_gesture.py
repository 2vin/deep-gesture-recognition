#################################################
## Gesture Recognition With Hand Pose 
## Author: Vinay (July 5, 2019)
## Usage:
## 	import inf_gesture
##	Detector = gDetector(models,config,label")
## 	Detector.detect(image)
#################################################

from libraries import *

class gDetector:
	def __init__(self, model, config, label):
		self.model = model
		self.config = config 
		self.label = label
		self.classes = None

		with open(self.label, 'r') as f:
			self.classes = [line.strip() for line in f.readlines()]

		self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
		self.net = cv2.dnn.readNet(self.model, self.config)


	def get_output_layers(self, net):
		net = self.net
		layer_names = net.getLayerNames()
		output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		return output_layers

	def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
	    classes = self.classes
	    label = self.label
	    label = str(classes[class_id])
	    color = self.COLORS[class_id]
	    cv2.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), color, 2)
	    cv2.putText(img, label, (int(x-10),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	def detect(self, image):
		net = self.net
		Width = image.shape[1]
		Height = image.shape[0]
		scale = 0.00392
		blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

		net.setInput(blob)
		outs = net.forward(self.get_output_layers(net))

		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.4
		nms_threshold = 0.4

		for out in outs:
		    for detection in out:
		        scores = detection[5:]
		        class_id = np.argmax(scores)
		        confidence = scores[class_id]
		        if confidence > 0.4:
		            center_x = int(detection[0] * Width)
		            center_y = int(detection[1] * Height)
		            w = int(detection[2] * Width)
		            h = int(detection[3] * Height)
		            x = center_x - w / 2
		            y = center_y - h / 2

		            if class_id == 1 or class_id == 2 or class_id == 4:
			            class_ids.append(class_id)
			            confidences.append(float(confidence))
			            boxes.append([x, y, w, h])

		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

		# Uncomment for testing individual module
		# for i in indices:
		#     i = i[0]
		#     box = boxes[i]
		#     x = box[0]
		#     y = box[1]
		#     w = box[2]
		#     h = box[3]
		#     self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

		return boxes, indices, self.classes, class_ids

	def close():
		cv2.destroyAllWindows()




cap = cv2.VideoCapture(0)

while(cv2.waitKey(10)<0):
	ret, image = cap.read()
	image = cv2.resize(image,(int(720),int(405)))

	Detector = gDetector("./model/gesture", "./model/gesture.cfg","./model/gesture.names")
	boxes, indices, classes, class_ids = Detector.detect(image)
	
	# Draw detections
	for i in indices:
		ind = i[0]
		box = boxes[ind]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		cv2.rectangle(image, (int(x),int(y)), (int(x+w),int(y+h)), (255-50*class_ids[i[0]], 30*class_ids[i[0]], 50*class_ids[i[0]]), 2)
		cv2.putText(image, classes[class_ids[i[0]]], (int(x-10),int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255-50*class_ids[i[0]], 50*class_ids[i[0]], 50*class_ids[i[0]]), 2)


	cv2.imshow("Gesture", image)
	# cv2.waitKey()