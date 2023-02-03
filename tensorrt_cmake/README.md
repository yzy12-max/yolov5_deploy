# 使用cmake编译步骤
+ 将一些头文件，要链接的依赖库用指令写在CMakeLists.txt里，这里都给了注释
+ 新建build和bin文件夹
```cpp
  mkdir build
  mkdir bin
  cd /xxx/build
  cmake ..
  make 
  cd /xxx/bin
  ./yolov5_trt
```
+ 最好写一个脚本一套流程下来直接执行生成的执行文件，因为cmake生成之后如果对代码进行了更改要将以前生成的CmakeCache.txt缓存删掉重新编译才行。
