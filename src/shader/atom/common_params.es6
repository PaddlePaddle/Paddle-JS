export default `
// varying变量
// 顶点shader透传的材质坐标
varying vec2 vCoord;
// 扩展shader的ivec类型
struct ivec5 {
    int x;
    int y;
    int z;
    int w;
    int u;
};
struct ivec6 {
    int x;
    int y;
    int z;
    int w;
    int u;
    int v;
};
// dynamic的input数据
const float multi_value = float(MULTI_VALUE);
const float bias_value = float(BIAS_VALUE);
`
