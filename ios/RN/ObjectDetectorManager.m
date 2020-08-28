#import "ObjectDetectorManager.h"
#if __has_include("TFLTensorFlowLite.h")
#import "Firebase/Firebase.h"

@interface ObjectDetectorManager ()
//@property(nonatomic, strong) FIRVisionTextRecognizer *textRecognizer;
//@property(nonatomic, assign) float scaleX;
//@property(nonatomic, assign) float scaleY;
@property(nonatomic, strong) TFLInterpreter *interpreter;
@property(nonatomic, strong) NSDictionary *options;
@end

@implementation ObjectDetectorManager

- (instancetype)init:(NSDictionary *)options
{
    if (self = [super init]) {
        NSLog(@"ObjectDetectorManager init");
        self.options = options;
        NSString* value = [options valueForKey:@"file"];
        NSLog(@"ObjectDetectorManager init options.file %@", value);
          
        NSString *modelPath = [NSBundle.mainBundle pathForResource:@"nsfw"
                                                            ofType:@"tflite"];
        NSError *tfliteError;

        self.interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                                  error:&tfliteError];
    }
    return self;
}

-(BOOL)isRealDetector
{
  return true;
}

- (void)destroy
{
    NSLog(@"ObjectDetectorManager destroy");
}

- (void)run:(UIImage *)uiImage completed: (void (^)(NSArray * result)) completed
{
    NSLog(@"ObjectDetectorManager run");
//    NSMutableArray *data = [[NSMutableArray alloc] init];
//    completed(data);

    CGImageRef image = uiImage.CGImage;
    long imageWidth = CGImageGetWidth(image);
    long imageHeight = CGImageGetHeight(image);
    CGContextRef context = CGBitmapContextCreate(nil,
                                                 imageWidth, imageHeight,
                                                 8,
                                                 imageWidth * 4,
                                                 CGColorSpaceCreateDeviceRGB(),
                                                 kCGImageAlphaNoneSkipFirst);
    CGContextDrawImage(context, CGRectMake(0, 0, imageWidth, imageHeight), image);
    UInt8 *imageData = CGBitmapContextGetData(context);

    NSMutableData *inputData = [[NSMutableData alloc] initWithCapacity:0];

    for (int row = 0; row < 224; row++) {
      for (int col = 0; col < 224; col++) {
        long offset = 4 * (col * imageWidth + row);
        // Normalize channel values to [0.0, 1.0]. This requirement varies
        // by model. For example, some models might require values to be
        // normalized to the range [-1.0, 1.0] instead, and others might
        // require fixed-point values or the original bytes.
        // (Ignore offset 0, the unused alpha channel)
        Float32 red = imageData[offset+1] / 255.0f;
        Float32 green = imageData[offset+2] / 255.0f;
        Float32 blue = imageData[offset+3] / 255.0f;

        [inputData appendBytes:&red length:sizeof(red)];
        [inputData appendBytes:&green length:sizeof(green)];
        [inputData appendBytes:&blue length:sizeof(blue)];
      }
    }
}

@end
#else

@interface ObjectDetectorManager ()
@end

@implementation ObjectDetectorManager

- (instancetype)init
{
  self = [super init];
  return self;
}

-(BOOL)isRealDetector
{
  return false;
}

-(void)run:(UIImage *)image completed:(postRecognitionBlock)completed;
{
  NSLog(@"ObjectDetector not installed, stub used!");
  NSArray *features = @[@"Error, Object Detector not installed"];
  completed(features);
}

@end
#endif
