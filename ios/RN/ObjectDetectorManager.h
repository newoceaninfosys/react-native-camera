#if __has_include("TFLTensorFlowLite.h")
  #import "TFLTensorFlowLite.h"
#endif
@interface ObjectDetectorManager : NSObject
typedef void(^postRecognitionBlock)(NSArray *textBlocks);

- (instancetype)init;
- (BOOL)isRealDetector;
- (void)run:(UIImage *)image completed:(postRecognitionBlock)completed;
- (void)load:(NSDictionary *)options;
- (NSData *) copyDataFromUIImage:(UIImage *) uiImage shape:(int)shape;
- (UIImage *) cropImage:(UIImage *) uiImage;
- (UIImage *) resizeImage:(UIImage *) uiImage shape:(int)shape;

@end
