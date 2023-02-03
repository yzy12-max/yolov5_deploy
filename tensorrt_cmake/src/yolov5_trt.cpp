// yzy
// yolov5进行tensorrt部署的源文件
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <glog/logging.h>
#include "yolov5_trt.h"


// 命名空间
using namespace cv;
using namespace nvinfer1;


// Logger for TRT info/warning/errors, https://github.com/onnx/onnx-tensorrt/blob/main/onnx_trt_backend.cpp
class TRT_Logger : public nvinfer1::ILogger
{
    nvinfer1::ILogger::Severity _verbosity;
    std::ostream* _ostream;

public:
    TRT_Logger(Severity verbosity = Severity::kWARNING, std::ostream& ostream = std::cout)
        : _verbosity(verbosity)
        , _ostream(&ostream)
    {
    }
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= _verbosity)
        {
            time_t rawtime = std::time(0);
            char buf[256];
            strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
            const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR
                        ? "  ERROR"
                        : severity == Severity::kWARNING ? "WARNING" : severity == Severity::kINFO ? "   INFO"
                                                                                                   : "UNKNOWN");
            (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
        }
    }
};

// 判断文件是否形成
static bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}


// 加载onnx文件
void YOLOv5::loadOnnx(const std::string strModelName)
{
    TRT_Logger gLogger;   // 日志
    //根据tensorrt pipeline 构建网络
    IBuilder* builder = createInferBuilder(gLogger);    // 
    builder->setMaxBatchSize(1);   // batchsize
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // 显式批处理
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);                      // 定义模型
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);              // 使用nvonnxparser 定义一个可用的onnx解析器
    parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));   // 解析onnx
	// 使用builder对象构建engine
    IBuilderConfig* config = builder->createBuilderConfig();   // 
	// 特别重要的属性是最大工作空间大小
    config->setMaxWorkspaceSize(1ULL << 30);                   // 分配内存空间
    m_CudaEngine = builder->buildEngineWithConfig(*network, *config);    // 来创建一个 ICudaEngine 类型的对象，在构建引擎时，TensorRT会复制权重

    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";  // 
    IHostMemory *gieModelStream = m_CudaEngine->serialize();    // 将引擎序列化
    std::string serialize_str;     // 
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size()); 
	// memcpy内存拷贝函数 ，从源内存地址的起始位置开始拷贝若干个字节到目标内存地址中
    memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size()); 
    serialize_output_stream.open(strTrtName.c_str());  
    serialize_output_stream<<serialize_str;     // 将引擎序列化数据转储到文件中
    serialize_output_stream.close();   
    m_CudaContext = m_CudaEngine->createExecutionContext();    //执行上下文用于执行推理
	// 使用一次，销毁parser，network, builder, and config 
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}


void YOLOv5::loadTrt(const std::string strName)
{
    TRT_Logger gLogger;
	// 序列化引擎被保留并保存到文件中
    m_CudaRuntime = createInferRuntime(gLogger);    
    std::ifstream fin(strName);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    { 
        std::stringstream buffer;   
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    m_CudaEngine = m_CudaRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr); // runtime对象反序列化
    m_CudaContext = m_CudaEngine->createExecutionContext();  //可以查询引擎获取有关网络的输入和输出的张量信息--维度/数据格式/数据类型
    m_CudaRuntime->destroy();
}


// 初始化
YOLOv5::YOLOv5(Configuration config)
{
	confThreshold = config.confThreshold;
	nmsThreshold = config.nmsThreshold;
	objThreshold = config.objThreshold;
	inpHeight = 640;
	inpWidth = 640;

	std::string model_path = config.modelpath;  // 模型权重路径
	// 加载模型
	std::string strTrtName = config.modelpath;      // 加载模型权重
	size_t sep_pos = model_path.find_last_of(".");
    strTrtName = model_path.substr(0, sep_pos) + ".engine"; // ".trt"
	if(ifFileExists(strTrtName.c_str()))
    {        
        loadTrt(strTrtName);
    }
    else
    {
        loadOnnx(config.modelpath);
    }

	// 利用加载的模型获取输入输出信息
	// 使用输入和输出blob名来获取输入和输出索引
    m_iInputIndex = m_CudaEngine->getBindingIndex("images");     // 输入索引
    m_iOutputIndex = m_CudaEngine->getBindingIndex("output");   // 输出  
	Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);  // 输入，
	int size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];   // 展平
	m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);   // 输入尺寸(W,H)
    Dims dims_o = m_CudaEngine->getBindingDimensions(m_iOutputIndex);  // 输出，维度[0,1,2,3]NHWC
	size = dims_o.d[0] * dims_o.d[1] * dims_o.d[2];   // 所有大小
    m_iClassNums = dims_o.d[2] - 5;    // [,,classes+5]
    m_iBoxNums = dims_o.d[1];    // [b,num_pre_boxes,classes+5]

	// 分配内存大小
    cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iInputIndex] = malloc(size * sizeof(float));
    m_ArraySize[m_iInputIndex] = size *sizeof(float);
    cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iOutputIndex] = malloc( size * sizeof(float));
    m_ArraySize[m_iOutputIndex] = size *sizeof(float);

    // bgr
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]); 
}

void YOLOv5::UnInit()
{

    for(auto &p: m_ArrayDevMemory)
    {      
        cudaFree(p);
        p = nullptr;            
    }        
    for(auto &p: m_ArrayHostMemory)
    {        
        free(p);
        p = nullptr;        
    }        
    cudaStreamDestroy(m_CudaStream);
    //m_CudaContext->destroy();    // 这个报错
    m_CudaEngine->destroy();

}

YOLOv5::~YOLOv5()
{
    UnInit();   
}



Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
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

void YOLOv5::nms(std::vector<BoxInfo>& input_boxes)
{
	
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	std::vector<bool> remove_flags(input_boxes.size(),false);
	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = max(box1.x1, box2.x1);
		float yy1 = max(box1.y1, box2.y1);
		float xx2 = min(box1.x2, box2.x2);
		float yy2 = min(box1.y2, box2.y2);
		// 交集
		float w = max(0.0f, xx2 - xx1 + 1);
		float h = max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = max(0.0f,box1.x2-box1.x1) * max(0.0f,box1.y2-box1.y1)
						   + max(0.0f,box2.x2-box2.x1) * max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

void YOLOv5::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	cv::cvtColor(dstimg, dstimg, cv::COLOR_BGR2RGB);   // 由BGR转成RGB
	cv::Mat m_Normalized;
	dstimg.convertTo(m_Normalized, CV_32FC3, 1/255.);
    cv::split(m_Normalized, m_InputWrappers);  // 通道分离[h,w,3] rgb
    //创建CUDA流,推理时TensorRT执行通常是异步的，因此将内核排入CUDA流
    cudaStreamCreate(&m_CudaStream); 
	auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream); 
	auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);    // TensorRT 执行通常是异步的，因此将内核排入 CUDA 流：
	ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream); //输出传回给CPU，数据从显存到内存
    ret = cudaStreamSynchronize(m_CudaStream);
	float* pdata = (float*)m_ArrayHostMemory[m_iOutputIndex];

	std::vector<BoxInfo> generate_boxes;  // BoxInfo自定义的结构体
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	for(int i = 0; i < m_iBoxNums; ++i) // 遍历所有的num_pre_boxes
	{
		int index = i * (m_iClassNums + 5);      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 置信度分数
		if (obj_conf > this->objThreshold)  // 大于阈值
		{
			float* max_class_pos = std::max_element(pdata + index + 5, pdata + index + 5 + m_iClassNums);   //
			(*max_class_pos) *= obj_conf;   // 最大的类别分数*置信度
			if ((*max_class_pos) > this->confThreshold) // 再次筛选
			{ 
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h

				float xmin = (cx - padw - 0.5 * w)*ratiow;
				float ymin = (cy - padh - 0.5 * h)*ratioh;
				float xmax = (cx - padw + 0.5 * w)*ratiow;
				float ymax = (cy - padh + 0.5 * h)*ratioh;

				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, (*max_class_pos), max_class_pos-(pdata + index + 5) });
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		std::string label = format("%.2f", generate_boxes[i].score);
		label = this->classes[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main(int argc,char *argv[])
{
	clock_t startTime,endTime; //计算时间
	Configuration yolo_nets = { 0.3, 0.5, 0.3,"/xxx/yolov5s.engine" };  // 注意路径
	YOLOv5 yolo_model(yolo_nets);
	std::string imgpath = "/xxx/bus.jpg";
	Mat srcimg = imread(imgpath);

	double timeStart = (double)getTickCount();
	startTime = clock();//计时开始	
	yolo_model.detect(srcimg);
	endTime = clock();//计时结束
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	std::cout << "clock_running time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
    std::cout << "The run time is:" << (double)clock() /CLOCKS_PER_SEC<< "s" << std::endl;
	std::cout << "getTickCount_running time :" << nTime << "sec\n" << std::endl;
	// static const string kWinName = "Deep learning object detection in ONNXRuntime";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	imwrite("/xxx/restult_trt.jpg",srcimg);
	// waitKey(0);
	// destroyAllWindows();
	return 0;
}