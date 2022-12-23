// yzy
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>      // 深度学习模块
#include <opencv2/imgproc.hpp>	// 图像处理模块
#include <opencv2/highgui.hpp>  // 高层GUI图形用户界面

#include<time.h>


// 命名空间
using namespace cv;
using namespace dnn;
using namespace std;

// 自定义配置结构
struct Configuration
{
	public: 
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};

// 模型
class YOLOv5
{
public:
    // 初始化
	YOLOv5(Configuration config,bool isCuda);
	void detect(Mat& frame);  // 检测函数
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	int num_classes;
    string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
							"train", "truck", "boat", "traffic light", "fire hydrant",
							"stop sign", "parking meter", "bench", "bird", "cat", "dog",
							"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
							"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
							"skis", "snowboard", "sports ball", "kite", "baseball bat",
							"baseball glove", "skateboard", "surfboard", "tennis racket",
							"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
							"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
							"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
							"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
							"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
							"sink", "refrigerator", "book", "clock", "vase", "scissors",
							"teddy bear", "hair drier", "toothbrush"};
	//vector<string> class_names;
	const bool keep_ratio = true;
	Net net;   // dnn里的
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
};

// 初始化
YOLOv5::YOLOv5(Configuration config,bool isCuda=false)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	this->net = readNet(config.modelpath);  // 解析模型onnx权重。dnn.hpp
	 //cuda // https://blog.csdn.net/cxyhjl/article/details/125383555
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		cout<<"cuda"<<endl;
	}
	//cpu
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	this->num_classes = sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 640;
	this->inpWidth = 640;
}

Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;  // 输入高宽
	*newh = this->inpHeight;    // 指针变量指向输入yolo模型的宽高
	*neww = this->inpWidth;
	Mat dstimg;                 // 定义一个目标源
	if (this->keep_ratio && srch != srcw) {  // 高宽不等
		float hw_scale = (float)srch / srcw; // 保存比列
		if (hw_scale > 1) {     // 按照yolov5的预处理进行处理
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale); // 
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
            // 和yolov5的处理对应,没有进行32的取模运算,这个是用114像素填充到(640,640)了,最后输入还是640,640
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);	
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOv5::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->classes[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

// 预测
void YOLOv5::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;  
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);  // 预处理
	Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false); // return:4-dimensional Mat with NCHW dimensions order.
	this->net.setInput(blob);    // 设置输出
	vector<Mat> outs;   // 要给空的走一遍
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());  // [b,num_pre,(5+classes)]

	int num_proposal = outs[0].size[1]; // 25200
	int out_dim2 = outs[0].size[2];  // 
	if (outs[0].dims > 2)
	{
		outs[0] = outs[0].reshape(0, num_proposal);  // 一般都会大于二维的，所以展平二维[b,num_pre*(5+classes)]
	}
	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;    //  opencv里保存box的
	vector<int> classIds;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	float* pdata = (float*)outs[0].data;  // 定义浮点型指针，
	for(int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
	{
		int index = i * out_dim2;      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 置信度分数
		if (obj_conf > this->objThreshold)  // 大于阈值
		{
			//Mat scores = outs[0].row(row_ind).colRange(5, nout); // 相当于python里的切片操作，每类的预测类别分数
			Mat scores(1, this->num_classes, CV_32FC1, pdata+index + 5);     // 这样操作更好理解
			Point classIdPoint; //定义点
			double max_class_socre; // 定义一个double类型的变量保存预测中类别分数最大值
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);  // 求每类类别分数最大的值和索引
			max_class_socre *= obj_conf;   // 最大的类别分数*置信度
			if (max_class_socre > this->confThreshold) // 再次筛选
			{ 
				const int class_idx = classIdPoint.x;  // 类别索引,在yolo里就是表示第几类
				// float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx,映射回原图。对应yolov5里后处理部分
				// float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy，但是现在的yolov5导出的onnx已经将这个处理放在里面了
				// float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w，所以这里不需要后处理一下了，直接取输出就行
				// float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

				// 经过后处理的只需要直接取就行
				float cx = pdata[index];  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h

				int left = int((cx - padw - 0.5 * w)*ratiow);  // *ratiow，变回原图尺寸
				int top = int((cy - padh - 0.5 * h)*ratioh);

				confidences.push_back((float)max_class_socre);
				boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));  //（x,y,w,h）
				classIds.push_back(class_idx);  // 
			}
		}
	
	}    

	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classIds[idx]);
	}
}

int main()
{

    clock_t startTime,endTime; //计算时间
	// 自己定义的yolo一些配置超参
	Configuration yolo_nets = { 0.3, 0.5, 0.3, "yolov5s.onnx" };
	YOLOv5 yolo_model(yolo_nets,false);
	string imgpath = "bus.jpg";
	Mat srcimg = imread(imgpath);

	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	startTime = clock();//计时开始	
	yolo_model.detect(srcimg);
	endTime = clock();//计时结束
	cout << "clock_running time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "The whole run time is:" << (double)clock() /CLOCKS_PER_SEC<< "s" << endl;
	cout << "getTickCount_running time :" << nTime << "s" << endl;

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);  // 自适应调节窗口大小
	imwrite("restult_cpu.jpg",srcimg);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
    return 0;
}


 