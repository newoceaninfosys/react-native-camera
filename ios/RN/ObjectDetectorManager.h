#if __has_include("TFLTensorFlowLite.h")
  #import "TFLTensorFlowLite.h"
#endif
@interface ObjectDetectorManager : NSObject
typedef void(^postRecognitionBlock)(NSArray *textBlocks);

- (instancetype)init:(NSDictionary *)options;
- (BOOL)isRealDetector;
- (void)run:(UIImage *)image completed:(postRecognitionBlock)completed;
- (void)destroy;

@end
