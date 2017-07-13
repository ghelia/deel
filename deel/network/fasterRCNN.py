
from chainer import serializers
from chainer.cuda import to_gpu
from deel.model.librcnn.cpu_nms import cpu_nms as nms
from deel.model.faster_rcnn import FasterRCNN as FRCNN
import chainer
import cv2 as cv
import numpy as np
from deel.deel import *
from deel.tensor import *
from deel.network import *
from deel.model import *

CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


def get_model(gpu):
	model = FasterRCNN(gpu)
	model.train = False
	serializers.load_npz('misc/VGG16_faster_rcnn_final.model', model)

	return model


def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
	img = orig_img.astype(np.float32, copy=True)
	img -= pixel_means
	im_size_min = np.min(img.shape[0:2])
	im_size_max = np.max(img.shape[0:2])
	im_scale = float(scale) / float(im_size_min)
	if np.round(im_scale * im_size_max) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
					interpolation=cv.INTER_LINEAR)

	return img.transpose([2, 0, 1]).astype(np.float32), im_scale


def draw_result(out, im_scale, clss, bbox, nms_thresh, conf):
	CV_AA = 16
	print(clss.shape)
	print(bbox.shape)
	for cls_id in range(1, 21):
		_cls = clss[:, cls_id][:, np.newaxis]
		_bbx = bbox[:, cls_id * 4: (cls_id + 1) * 4]
		dets = np.hstack((_bbx, _cls))
		keep = nms(dets, nms_thresh)
		dets = dets[keep, :]

		inds = np.where(dets[:, -1] >= conf)[0]
		for i in inds:
			x1, y1, x2, y2 = map(int, dets[i, :4])
			cv.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2, CV_AA)
			ret, baseline = cv.getTextSize(
				CLASSES[cls_id], cv.FONT_HERSHEY_SIMPLEX, 0.8, 1)
			cv.rectangle(out, (x1, y2 - ret[1] - baseline),
						 (x1 + ret[0], y2), (0, 0, 255), -1)
			cv.putText(out, CLASSES[cls_id], (x1, y2 - baseline),
					   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, CV_AA)

	return out


def draw_rois(out,im_scale, rois,bbox,cls):
	CV_AA = 16
	print(bbox.shape)
	for i in range(len(rois)):
		n,x1,y1,x2,y2 = rois[i]
		canvas = out.copy()
		cv.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1, CV_AA)
		
		cls_id=np.argmax(cls[i])
		if cls[i][cls_id]>0.1 and cls_id != 0:
			x1 = int(x1)
			x2 = int(x2)
			y1 = int(x1)
			y2 = int(y2)
			ret, baseline = cv.getTextSize(
				CLASSES[cls_id], cv.FONT_HERSHEY_SIMPLEX, 0.8, 1)
			cv.rectangle(out, (x1, y2 - ret[1] - baseline),
						 (x1 + ret[0], y2), (0, 0, 255), -1)
			cv.putText(out, CLASSES[cls_id], (x1, y2 - baseline),
					   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, CV_AA)
		for j in range(0,84,4):
			x1,y1,x2,y2 = bbox[i][j:j+4]
			cv.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1, CV_AA)
		cv.imshow("res",canvas)
		cv.waitKey(0)		

	return out
'''
	FasterRCNN  
'''
class FasterRCNN(ImageNet):
	def __init__(self,modelpath='misc/VGG16_faster_rcnn_final.model',
					mean=[102.9801, 115.9465, 122.7717],
					in_size=224):
		super(FasterRCNN,self).__init__('FasterRCNN',in_size)
		self.func = FRCNN(Deel.gpu)
		self.func.train=False
		serializers.load_npz('misc/VGG16_faster_rcnn_final.model', self.func)

		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = mean[0]
		ImageNet.mean_image[1] = mean[1]
		ImageNet.mean_image[2] = mean[2]
		ImageNet.in_size = in_size

		self.labels = CLASSES

		self.batchsize = 1
		xp = Deel.xp
		self.x_batch = xp.ndarray((self.batchsize, 3, self.in_size, self.in_size), dtype=np.float32)

		if Deel.gpu >=0:
			self.func = self.func.to_gpu(Deel.gpu)
		self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.func)

	def forward(self,x,train=True):
		y = self.func(x,np.array([[224, 224, 1.0]]))
		return y

	def Input(self,x):
		xp = Deel.xp
		if isinstance(x,str):
			#img = Image.open(x).convert('RGB')
			orig_image = cv.imread(x)
			image, im_scale = img_preprocessing(orig_image, PIXEL_MEANS)
			t = ImageTensor(orig_image,filtered_image=image,
							h=image.shape[1],w=image.shape[2],
							in_size=self.in_size)
			t.im_scale = im_scale
		elif hasattr(x,'_Image__transformer'):
			t = ImageTensor(x,filtered_image=filter(np.asarray(x)),
							in_size=self.in_size)
		else:
			t = ImageTensor(x,filtered_image=filter(np.asarray(x)),
							in_size=self.in_size)
		t.use()
		return t

	def feature(self,x=None):
		if x is None:
			x=Tensor.context


		#if not isinstance(x,ImageTensor):
		#	x=self.Input(x)

		xp = Deel.xp
		x_data = xp.asarray([x.value],dtype=xp.float32)
		xv = chainer.Variable(x_data, volatile=True)

		h, w = xv.data.shape[2:]
		cls_score, bbox_pred  = self.func(xv,np.array([[h, w, 1.0]]))
		return self.func.score_fc7,self.func.deltas

	def classify(self,x=None):
		if x is None:
			x=Tensor.context

		if not isinstance(x,ImageTensor):
			x=Input(x)

		xp = Deel.xp
		x_data = xp.asarray(self.x_batch)
		xv = chainer.Variable(x.value, volatile=True)

		h, w = xv.data.shape[2:]
		cls_score, bbox_pred  = self.func(xv,np.array([[h, w, x.im_scale]]))
		draw_rois(x.content,x.im_scale,self.func.rois,bbox_pred,cls_score.data)

		if Deel.gpu >= 0:
			cls_score = chainer.cuda.cupy.asnumpy(cls_score)
			bbox_pred = chainer.cuda.cupy.asnumpy(bbox_pred)
		result = draw_result(x.content, 1.0, cls_score.data, bbox_pred,0.3,0.8)
		cv.imshow("res",result)
		cv.waitKey(0)







