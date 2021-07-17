import { Tensor } from './Tensor';
import { getIntArray, getInt } from './utils';

/* eslint-disable max-statements, max-depth */
function main(tensorMap: Map<string, Tensor>, attrs: Attrs, runtime: i32): f32[] {
    const origin = tensorMap.get('origin') as Tensor;
    const out = tensorMap.get('out_' + runtime) as Tensor;

    const originShape: i32[] = (origin as Tensor).shape;
    const outShape: i32[] = (out as Tensor).shape;

    const originData: f32[] = (origin as Tensor).data;

    const originReducedShape: i32[] = (origin as Tensor).shapeReduced;
    const outReducedShape: i32[] = (out as Tensor).shapeReduced;

    const strides = attrs.strides;
    const paddings = attrs.paddings;
    const ksizes = attrs.ksize;
    const pooling_type = attrs.pooling_type;
    
    const stride_v = strides[0] || 1;
    const stride_h = strides[1] || 1;
    const ksize_x = ksizes[0] || 1;
    const ksize_y = ksizes[1] || 1;
    const padTop = paddings[0] || 0;
    const padLeft = paddings[1] || 0;

    const outB = outShape[0];
    const outC = outShape[1];
    const outH = outShape[2];
    const outW = outShape[3];


    const originH = originShape[2];
    const originW = originShape[3];

    const originS0 = originReducedShape[0];
    const originS1 = originReducedShape[1];
    const originS2 = originReducedShape[2];

    const outS0 = outReducedShape[0];
    const outS1 = outReducedShape[1];
    const outS2 = outReducedShape[2];

    const result = new Array<f32>(out.total);

    for (let n = 0; n < outB; n++) {
        for (let c = 0; c < outC; c++) {
            for (let h = 0; h < outH; h++) {
                for (let w = 0; w < outW; w++) {
                    let res = 0.0;
                    let count_pool = 0;
                    const oy_base = h * stride_v - padTop;
                    const ox_base = w * stride_h - padLeft;
                    for (let fy = 0; fy < ksize_y; fy++) {
                        const oy = oy_base + fy;
                        if (oy >= originH) {
                            break;
                        }
                        if (oy < 0) {
                            continue;
                        }
                        for (let fx = 0; fx < ksize_x; fx++) {
                            const ox = ox_base + fx;
                            if (ox >= originW) {
                                break;
                            }
                            if (ox < 0) {
                                continue;
                            }
                            // origin数据
                            const curr = originData[n * originS0 + c * originS1 + oy * originS2 + ox];
                            /* eslint-disable-next-line */
                            if (pooling_type == 1) {
                                // max-pool update
                                if (count_pool == 0){
                                    res = curr;
                                    count_pool++;
                                }
                                if (curr > res) {
                                    res = curr;
                                }
                            }
                            else {
                                res += curr;
                                // 在平均池化模式忽略填充值(exclusive默认为true）
                                count_pool++;
                            }
                        }
                    }
                    /* eslint-disable-next-line */
                    if (pooling_type != 1) {
                        res = res / count_pool;
                    }
                    result[n * outS0 + c * outS1 + h * outS2 + w] = f32(res);
                }
            }
        }
    }

    return result;
}
/* eslint-enable max-statements */

class Attrs {
    strides: i32[] = [];
    paddings: i32[] = [];
    ksize: i32[] = [];
    pooling_type: i32;
    constructor(data: Obj) {
        this.strides = getIntArray('strides', data);
        this.paddings = getIntArray('paddings', data);
        this.ksize = getIntArray('dilations', data);
        this.pooling_type = getInt('groups', data);
    }
}

const behaviors = [
    'isMax',
    'setPacked',
    'isGlobalPooling'
];

export {
    main,
    Attrs,
    behaviors
};