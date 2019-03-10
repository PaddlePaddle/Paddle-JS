// conv2d的input数据

// 常量
// 卷积核
const int length_shape_filter = LENGTH_SHAPE_FILTER;
const int width_shape_filter = WIDTH_SHAPE_FILTER;
const int height_shape_filter = HEIGHT_SHAPE_FILTER;
const int width_texture_filter = WIDTH_TEXTURE_FILTER;
const int height_texture_filter = HEIGHT_TEXTURE_FILTER;
const int channel_filter = CHANNEL_FILTER;

// 输入数据
const int width_shape_origin = WIDTH_SHAPE_ORIGIN;
const int height_shape_origin = HEIGHT_SHAPE_ORIGIN;
const int length_shape_origin = LENGTH_SHAPE_ORIGIN;
const int width_texture_origin = WIDTH_TEXTURE_ORIGIN;
const int height_texture_origin = HEIGHT_TEXTURE_ORIGIN;
const int channel_origin = CHANNEL_ORIGIN;

// 输出数据
const int width_shape_out = WIDTH_SHAPE_OUT;
const int height_shape_out = HEIGHT_SHAPE_OUT;
const int width_texture_out = WIDTH_TEXTURE_OUT;
const int height_texture_out = HEIGHT_TEXTURE_OUT;
const int channel_out = CHANNEL_OUT;
const int length_shape_out = LENGTH_SHAPE_OUT;

// 计算相关
// 拆分步长
const int stride_h = STRIDE_HORIZONTAL;
const int stride_v = STRIDE_VERTICAL;
// padding的数目
const int padLeft = PAD_LEFT;
const int padTop = PAD_TOP;
// dilation膨胀系数
const int dilation_h = DILATION_HORIZONTAL;
const int dilation_v = DILATION_VERTICAL;
// 激活函数
const float multi_value = MULTI_VALUE;
const float bias_value = BIAS_VALUE;

// uniform变量
// 卷积核
uniform int numbers_shape_filter[LENGTH_SHAPE_FILTER];
uniform int numbers_shape_origin[LENGTH_SHAPE_ORIGIN];
uniform int numbers_shape_out[LENGTH_SHAPE_OUT];
uniform sampler2D texture_filter;

// 输入数据
uniform sampler2D texture_origin;