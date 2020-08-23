#if __has_include("TFLTensorFlowLite.h")
  #import "TFLTensorFlowLite.h"
#endif
  @interface ObjectDetectorManager : NSObject
  typedef void(^postRecognitionBlock)(NSArray *textBlocks);

  - (instancetype)init;

  -(BOOL)isRealDetector;
  -(void)findObjects:(UIImage *)image completed:(postRecognitionBlock)completed;

  @end
